#!/usr/bin/env python3
"""Repeatable throughput run for one method/config on one visible GPU.

Correctness validation must pass first. Run exactly one process per GPU so
PCIe and pinned-memory bandwidth remain interpretable."""
import argparse
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from reviewer_offload_backend_v12 import OffloadEngine, load_opt

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path('/home/liushenyu/opt-13b')
PREDICTOR = Path('/home/liushenyu/test/predictor_topk_0.3_iwl_epoch_20.pth')
EMBEDDING = Path('/home/liushenyu/test/opt_13b_embed_weights.pth')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--predictor', type=Path, default=PREDICTOR)
    parser.add_argument('--embedding', type=Path, default=EMBEDDING)
    parser.add_argument('--strategy', required=True, choices=('full_gpu', 'full_offload', 'neural_serial', 'neural_async'))
    parser.add_argument('--prompt-len', type=int, required=True)
    parser.add_argument('--gen-len', type=int, required=True)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--budget', type=float, default=.3)
    parser.add_argument('--recent', type=int, default=20)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261026)
    parser.add_argument('--threads', type=int, default=10)
    parser.add_argument('--allocator-budget-gib', type=float, default=29.5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible not in ('0', '1', '2', '3') or not torch.cuda.is_available():
        raise RuntimeError('Expose exactly one authorized GPU (0, 1, 2 or 3)')
    if args.prompt_len < 16 or args.prompt_len + args.gen_len > 2048 or args.gen_len < 2:
        raise ValueError('Lengths must fit OPT-13B 2048 positions and include >=2 output tokens')
    if args.batch < 1 or args.repeats < 2 or args.warmups < 1:
        raise ValueError('Use batch>=1, at least 2 measured repetitions and 1 warmup')
    if args.strategy.startswith('neural') and (not args.predictor.is_file() or not args.embedding.is_file()):
        raise FileNotFoundError('Neural benchmark requires the original trained predictor and its embedding table')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_per_process_memory_fraction(args.allocator_budget_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory)

    from device_memory_monitor_v12 import DeviceMemoryMonitor
    monitor = DeviceMemoryMonitor(int(visible))

    opt = load_opt()
    model, loading = opt.OPTForCausalLM.from_pretrained(
        str(args.model), torch_dtype=torch.float16, device_map={'': 0},
        local_files_only=True, output_loading_info=True)
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError(f'Checkpoint mismatch: {loading}')
    predictor = embedding = None
    if args.strategy.startswith('neural'):
        from predictor_model2A import PredictorTransformer
        predictor = PredictorTransformer(5120, 5, nhead=4, num_encoder_layers=1, d_hid=2048)
        predictor.load_state_dict(torch.load(args.predictor, map_location='cpu', weights_only=True), strict=True)
        predictor.to('cuda').eval()
        embedding = torch.load(args.embedding, map_location='cpu', weights_only=True)
        if embedding.shape != (model.config.vocab_size, model.config.hidden_size):
            raise ValueError(f'Predictor embedding shape mismatch: {tuple(embedding.shape)}')
        # The source table is FP16. Match the accuracy runner's original table.
        embedding = embedding.to(device='cuda', dtype=torch.float16)
        native_embedding = model.model.decoder.embed_tokens.weight
        if not torch.equal(embedding, native_embedding):
            raise RuntimeError('Exact embedding sharing requires bitwise equal values')
        # Reuse the exact same FP16 table; predictor still casts lookups to FP32.
        embedding = native_embedding
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), use_fast=False, local_files_only=True)
    templates = ('The reader opened the old archive and searched the following page.',
                 'A scientist measured the temperature before beginning a new experiment.',
                 'The committee reviewed the report and discussed its main conclusion.',
                 'Several students walked across campus after the lecture ended.')
    prompt_rows = []
    for b in range(args.batch):
        row = tokenizer(' '.join([templates[b % len(templates)]] * (args.prompt_len // 8 + 64)),
                        add_special_tokens=False, return_tensors='pt').input_ids[0, :args.prompt_len]
        if row.numel() != args.prompt_len:
            row = row.repeat((args.prompt_len + row.numel() - 1) // row.numel())[:args.prompt_len]
        prompt_rows.append(row)
    prompt = torch.stack(prompt_rows).cuda()
    engine = OffloadEngine(model, predictor, embedding)
    try:
        runs = []
        for iteration in range(args.warmups + args.repeats):
            result, _, _ = engine.generate(prompt, args.gen_len, args.strategy,
                                           ratio=args.budget, recent=args.recent)
            if iteration < args.warmups:
                continue
            runs.append({k: v for k, v in result.items() if k != 'output_ids'})
            runs[-1]['output_sha256'] = hashlib.sha256(
                json.dumps(result['output_ids'], separators=(',', ':')).encode()).hexdigest()
            print(json.dumps({'iteration': len(runs), 'total_s': result['total_s'],
                              'decode_s': result['decode_s'],
                              'decode_tokens_per_s': result['decode_tokens_per_s']}), flush=True)
        fields = ('prefill_s', 'decode_s', 'total_s', 'decode_tokens_per_s',
                  'end_to_end_output_tokens_per_s', 'peak_allocated_bytes',
                  'peak_reserved_bytes', 'cpu_process_peak_rss_bytes',
                  'actual_history_selection_ratio', 'h2d_kv_bytes',
                  'prefill_peak_allocated_bytes', 'decode_peak_allocated_bytes',
                  'gpu_staging_bytes', 'pinned_staging_bytes', 'cpu_kv_home_bytes',
                  'request_setup_s', 'request_total_including_setup_s')
        summary = {field: {'mean': statistics.mean(run[field] for run in runs),
                           'stdev': statistics.stdev(run[field] for run in runs)}
                   for field in fields}
        document = {
            'complete': True, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'purpose': 'v14 compact staging and shared embedding; bounded default allocator',
            'embedding_shared_with_model': embedding is model.model.decoder.embed_tokens.weight,
            'predictor_embedding_additional_bytes': 0 if embedding is model.model.decoder.embed_tokens.weight else embedding.numel()*embedding.element_size(),
            'runner_sha256': sha256(Path(__file__)),
            'allocator_budget_gib': args.allocator_budget_gib,
            'allocator_config': os.environ.get('PYTORCH_CUDA_ALLOC_CONF', ''),
            'backend_sha256': sha256(ROOT / 'speedup/reviewer_offload_backend_v12.py'),
            'model_tensor_bytes': sum(x.numel()*x.element_size() for x in model.parameters()) + sum(x.numel()*x.element_size() for x in model.buffers()),
            'predictor_tensor_bytes': sum(x.numel()*x.element_size() for x in predictor.parameters()) + sum(x.numel()*x.element_size() for x in predictor.buffers()) if predictor is not None else 0,
            'predictor_embedding_bytes': embedding.numel()*embedding.element_size() if embedding is not None else 0,
            'gpu_visible_devices': visible, 'gpu_name': torch.cuda.get_device_name(0),
            'torch_version': torch.__version__, 'model_path': str(args.model),
            'model_config_sha256': sha256(args.model / 'config.json'),
            'predictor_path': str(args.predictor) if args.strategy.startswith('neural') else None,
            'predictor_sha256': sha256(args.predictor) if args.strategy.startswith('neural') else None,
            'embedding_path': str(args.embedding) if args.strategy.startswith('neural') else None,
            'embedding_sha256': sha256(args.embedding) if args.strategy.startswith('neural') else None,
            'strategy': args.strategy, 'prompt_len': args.prompt_len, 'generation_len': args.gen_len,
            'batch_size': args.batch, 'budget_ratio': args.budget, 'recent_union': args.recent,
            'warmups': args.warmups, 'repeats': args.repeats, 'cpu_threads': args.threads,
            'summary': summary, 'runs': runs,
            'interpretation': 'Prefill, decode and end-to-end are separate. decode_tokens_per_s excludes prefill. Peak GPU allocation and physical pinned CPU/GPU staging are reported separately.',
        }
        document['nvml_memory'] = monitor.finish()
        with args.output.open('x') as stream:
            stream.write(json.dumps(document, indent=2) + '\n')
        print(args.output)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
