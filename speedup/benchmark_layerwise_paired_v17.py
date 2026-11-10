#!/usr/bin/env python3
"""Balanced within-process comparison of the unchanged V12 and V16 backends.

One common model, predictor, prompt, allocator and CPU affinity. Each backend
gets one full warmup; six adjacent pairs alternate A/B and B/A. No profiling
is enabled in the timed runs. Output/logit/index parity is checked after time.
"""
import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from reviewer_offload_backend_v12 import OffloadEngine as BulkEngine, load_opt
from reviewer_offload_backend_v16 import OffloadEngine as LayerEngine


ROOT = Path(__file__).resolve().parents[1]
CLASSES = {'bulk_v12': BulkEngine, 'layer_v16': LayerEngine}


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, indent=2)
        stream.write('\n')


def smi(*args):
    return subprocess.check_output(['nvidia-smi', *args], text=True).strip()


def gpu_state(gpu):
    uuid = smi('-i', str(gpu), '--query-gpu=uuid', '--format=csv,noheader')
    processes = smi('--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                    '--format=csv,noheader')
    foreign = []
    for row in processes.splitlines():
        values = [part.strip() for part in row.split(',')]
        if values and values[0] == uuid and int(values[1]) != os.getpid():
            foreign.append(row)
    if foreign:
        raise RuntimeError('GPU has another compute process: ' + repr(foreign))
    return dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        gpu=smi('-i', str(gpu),
                '--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw',
                '--format=csv,noheader'),
        compute_processes=processes.splitlines(), load_average=os.getloadavg())


def digest_json(data):
    return hashlib.sha256(json.dumps(data, separators=(',', ':')).encode()).hexdigest()


def digest_tensor(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=6)
    parser.add_argument('--prompt-len', type=int, default=1920)
    parser.add_argument('--gen-len', type=int, default=128)
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    if args.pairs < 4 or args.pairs % 2:
        raise ValueError('Use an even number of pairs >=4 for balanced ordering')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible != '3' or len(os.sched_getaffinity(0)) > 10:
        raise RuntimeError('Use exactly GPU 3 and at most ten CPU cores')
    if os.environ.get('PYTORCH_CUDA_ALLOC_CONF') != 'backend:cudaMallocAsync':
        raise RuntimeError('Use the same cudaMallocAsync allocator as V13/V16')
    if args.prompt_len + args.gen_len > 2048:
        raise ValueError('OPT positional limit exceeded')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    initial = gpu_state(int(visible))
    torch.set_num_threads(10)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261026)
    torch.cuda.manual_seed_all(20261026)

    model_path = Path('/home/liushenyu/opt-13b')
    predictor_path = Path('/home/liushenyu/test/predictor_topk_0.3_iwl_epoch_20.pth')
    embedding_path = Path('/home/liushenyu/test/opt_13b_embed_weights.pth')
    model, loading = load_opt().OPTForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.float16, device_map={'': 0},
        local_files_only=True, output_loading_info=True)
    if any(loading.get(k) for k in
           ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError(loading)
    from predictor_model2A import PredictorTransformer
    predictor = PredictorTransformer(5120, 5, nhead=4, num_encoder_layers=1,
                                     d_hid=2048)
    predictor.load_state_dict(torch.load(predictor_path, map_location='cpu',
                                        weights_only=True), strict=True)
    predictor.cuda().eval()
    source_embedding = torch.load(embedding_path, map_location='cpu',
                                  weights_only=True).half().cuda()
    embedding = model.model.decoder.embed_tokens.weight
    assert torch.equal(source_embedding, embedding)
    del source_embedding
    assert next(predictor.parameters()).dtype == torch.float32
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False,
                                              local_files_only=True)
    templates = (
        'The reader opened the old archive and searched the following page.',
        'A scientist measured the temperature before beginning a new experiment.',
        'The committee reviewed the report and discussed its main conclusion.',
        'Several students walked across campus after the lecture ended.')
    rows = []
    for b in range(args.batch):
        row = tokenizer(' '.join([templates[b % 4]] * (args.prompt_len // 8 + 64)),
                        add_special_tokens=False, return_tensors='pt').input_ids[0, :args.prompt_len]
        if row.numel() != args.prompt_len:
            row = row.repeat((args.prompt_len + row.numel() - 1) // row.numel())[:args.prompt_len]
        rows.append(row)
    prompt = torch.stack(rows).cuda()
    manifest = dict(
        protocol='balanced adjacent A/B, B/A pairs in one common-model process',
        initial_gpu_state=initial, model_path=str(model_path),
        predictor_path=str(predictor_path), embedding_path=str(embedding_path),
        predictor_sha256=sha_file(predictor_path),
        embedding_sha256=sha_file(embedding_path),
        model_config_sha256=sha_file(model_path / 'config.json'),
        prompt_sha256=digest_tensor(prompt.cpu()), seed=20261026,
        predictor_config=dict(encoder_layers=1, heads=4, ffn=2048, dtype='float32'),
        batch=args.batch, prompt_len=args.prompt_len, gen_len=args.gen_len,
        budget=0.3, recent=20, strategy='neural_async', pairs=args.pairs,
        cpu_affinity=sorted(os.sched_getaffinity(0)), cpu_threads=10,
        torch_version=torch.__version__,
        allocator=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
        source_sha256={name: sha_file(ROOT / 'speedup' / name) for name in
                       ('benchmark_layerwise_paired_v17.py',
                        'reviewer_offload_backend_v12.py',
                        'reviewer_offload_backend_v16.py')})
    write_new(args.output_dir / 'manifest.json', manifest)
    reference = None
    all_runs = []

    def run_one(label, phase, pair, position):
        nonlocal reference
        before = gpu_state(int(visible))
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        engine = CLASSES[label](model, predictor, embedding)
        try:
            started = time.perf_counter()
            result, pre, logits = engine.generate(
                prompt, args.gen_len, 'neural_async', ratio=.3, recent=20)
            after = gpu_state(int(visible))
            validation = dict(
                output_sha256=digest_json(result['output_ids']),
                prefill_logits_sha256=digest_tensor(pre),
                decode_logits_sha256=digest_tensor(logits),
                index_hashes=[p['index_hash'] for p in result['plans']],
                h2d_kv_bytes=result['h2d_kv_bytes'])
            if reference is None:
                reference = dict(validation)
            if validation != reference:
                write_new(args.output_dir / ('PARITY_FAILURE_' + phase + '_' + str(pair) + '_' + label + '.json'),
                          dict(expected=reference, actual=validation))
                raise RuntimeError('Logit, output, index or transfer parity failed')
            result.pop('output_ids')
            document = dict(
                complete=True, phase=phase, pair=pair, position=position,
                backend=label, before=before, after=after,
                invocation_including_diagnostics_s=time.perf_counter()-started,
                validation=validation, result=result)
            filename = (f'warmup-{label}.json' if phase == 'warmup' else
                        f'pair{pair:02d}-{position}-{label}.json')
            write_new(args.output_dir / filename, document)
            print(json.dumps(dict(phase=phase, pair=pair, position=position,
                                  backend=label, decode_tps=result['decode_tokens_per_s'],
                                  allocated_gib=result['peak_allocated_bytes'] / 2**30,
                                  bitwise_logit_parity=True)), flush=True)
            if phase == 'measured':
                all_runs.append(document)
        finally:
            engine.close()
            del engine
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    for label in CLASSES:
        run_one(label, 'warmup', 0, 0)
    paired = []
    for pair in range(1, args.pairs + 1):
        order = tuple(CLASSES) if pair % 2 else tuple(reversed(CLASSES))
        for position, label in enumerate(order, 1):
            run_one(label, 'measured', pair, position)
        a, b = [next(r['result'] for r in all_runs
                     if r['pair'] == pair and r['backend'] == label)
                for label in CLASSES]
        paired.append(dict(
            pair=pair, order=list(order),
            bulk_tps=a['decode_tokens_per_s'], layer_tps=b['decode_tokens_per_s'],
            layer_divided_by_bulk=b['decode_tokens_per_s']/a['decode_tokens_per_s'],
            decode_time_saved_s=a['decode_s']-b['decode_s'],
            allocated_saved_gib=(a['peak_allocated_bytes']-b['peak_allocated_bytes'])/2**30))
    metrics = ('decode_tokens_per_s', 'decode_s', 'total_s', 'prefill_s',
               'peak_allocated_bytes', 'peak_reserved_bytes', 'gpu_staging_bytes',
               'actual_history_selection_ratio', 'h2d_kv_bytes')
    summary = {}
    for label in CLASSES:
        runs = [r['result'] for r in all_runs if r['backend'] == label]
        summary[label] = {
            key: dict(mean=statistics.mean(r[key] for r in runs),
                      stdev=statistics.stdev(r[key] for r in runs))
            for key in metrics}
    report = dict(complete=True, all_validation_bitwise_equal=True,
                  manifest=manifest, summary=summary, pairs=paired,
                  number_of_pairs_layer_faster=sum(p['layer_divided_by_bulk'] > 1 for p in paired),
                  ratio_of_mean_throughputs=summary['layer_v16']['decode_tokens_per_s']['mean']/
                                            summary['bulk_v12']['decode_tokens_per_s']['mean'],
                  median_paired_ratio=statistics.median(p['layer_divided_by_bulk'] for p in paired),
                  limitations='One workload on shared host. Balances time/order on one GPU; compares entire staging implementations, not individual micro-causes.')
    write_new(args.output_dir / 'summary.json', report)
    print(json.dumps({'complete': True, 'summary': summary, 'pairs': paired}, indent=2), flush=True)


if __name__ == '__main__':
    main()
