#!/usr/bin/env python3
"""Isolated service measurements and serial/async CUDA traces (run after suite).

Uses the same predictor, packing path and offload engine as the timing matrix.
Microbenchmarks use a fixed immutable prefix to isolate costs; quality must be
assessed separately. No output from this script is an end-to-end throughput run.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from reviewer_offload_backend import OffloadEngine, load_opt, neural_indices, pad_indices
from benchmark_reviewer_offload import MODEL, PREDICTOR, EMBEDDING, sha256


def summarize(samples):
    return {'mean': statistics.mean(samples), 'stdev': statistics.stdev(samples), 'samples': samples}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=MODEL)
    p.add_argument('--predictor', type=Path, default=PREDICTOR)
    p.add_argument('--embedding', type=Path, default=EMBEDDING)
    p.add_argument('--contexts', nargs='+', type=int, default=[512, 1024, 1536, 1920])
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--trace', action='store_true')
    args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0', '1', '2', '3') or not torch.cuda.is_available():
        raise RuntimeError('Expose exactly one idle authorized GPU')
    if args.repeats < 2 or args.output_dir.exists() or args.batch < 1:
        raise ValueError('Need >=2 repeats, positive batch and a new output directory')
    if not all(16 <= x <= 2040 for x in args.contexts):
        raise ValueError('Contexts must leave room for trace decode')
    args.output_dir.mkdir(parents=True)
    torch.manual_seed(20261026)
    torch.set_num_threads(10)
    torch.set_num_interop_threads(1)
    model, loading = load_opt().OPTForCausalLM.from_pretrained(
        str(args.model), torch_dtype=torch.float16, device_map={'': 0},
        local_files_only=True, output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError(loading)
    from predictor_model2A import PredictorTransformer
    predictor = PredictorTransformer(5120, 5, nhead=4, num_encoder_layers=1, d_hid=2048)
    predictor.load_state_dict(torch.load(args.predictor, map_location='cpu', weights_only=True), strict=True)
    predictor.cuda().eval()
    embedding = torch.load(args.embedding, map_location='cpu', weights_only=True).cuda()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), use_fast=False, local_files_only=True)
    texts = ['The reader opened the old archive and searched the following page.',
             'A scientist measured the temperature before beginning a new experiment.']
    engine = OffloadEngine(model, predictor, embedding)
    results = []
    try:
        with torch.inference_mode():
            for length in args.contexts:
                prompt = torch.cat([tokenizer(' '.join([texts[b % 2]] * 300), return_tensors='pt',
                                              add_special_tokens=False).input_ids[:, :length]
                                    for b in range(args.batch)]).cuda()
                engine.allocate(args.batch, length+8, 'neural_serial', .3, 20)
                engine.prompt_len = length
                engine.prefill = True
                logits = engine.forward(prompt)
                decode_input = logits.argmax(-1)[:, None]
                torch.cuda.synchronize()
                engine.prefill = False
                engine.history = length
                samples = {key: [] for key in ('predictor_gpu_ms', 'prediction_selection_host_s',
                                               'packing_host_s', 'h2d_stream_ms',
                                               'prepare_wall_s', 'llm_plus_kv_store_wall_s',
                                               'llm_stream_ms')}
                for iteration in range(args.repeats+2):
                    ready = torch.cuda.Event()
                    ready.record()
                    started = time.perf_counter()
                    plan = engine.prepare(engine.slots[0], prompt, length, length, ready,
                                          tuple(engine.store_events))
                    plan['slot'].ready.synchronize()
                    prepare_s = time.perf_counter()-started
                    engine.current_plan = plan
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    started = time.perf_counter()
                    start.record()
                    engine.forward(decode_input, length)
                    end.record()
                    plan['slot'].consumed.record()
                    torch.cuda.synchronize()
                    llm_s = time.perf_counter()-started
                    if iteration >= 2:
                        a, b = plan['predictor_events']
                        c, d = plan['dma_events']
                        values = (a.elapsed_time(b), plan['prediction_selection_host_s'],
                                  plan['pack_s'], c.elapsed_time(d), prepare_s,
                                  llm_s, start.elapsed_time(end))
                        for key, value in zip(samples, values):
                            samples[key].append(value)
                results.append({'context': length, 'batch': args.batch,
                                'selected_history_ratio': plan['selected']/plan['candidates'],
                                'h2d_kv_bytes': plan['transferred_bytes'],
                                'isolated': {key: summarize(value) for key, value in samples.items()}})
                engine.current_plan = None
                engine.home = engine.resident = None
                engine.slots = []
                if args.trace:
                    # Profiling is explicitly separate from throughput results.
                    for strategy in ('neural_serial', 'neural_async'):
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA]) as profiler:
                            metrics, _, _ = engine.generate(prompt, 8, strategy)
                        profiler.export_chrome_trace(str(args.output_dir / f'{strategy}-p{length}.trace.json'))
                        results[-1][strategy+'_profiled_run'] = {k: v for k, v in metrics.items() if k != 'output_ids'}
                print(json.dumps({'complete_context': length, 'isolated': results[-1]['isolated']}), flush=True)
        (args.output_dir/'components.json').write_text(json.dumps({
            'complete': True, 'model': str(args.model), 'predictor_sha256': sha256(args.predictor),
            'embedding_sha256': sha256(args.embedding), 'results': results,
            'note': 'Isolated measurements have no concurrent predictor/LLM. KV store D2H remains part of the LLM path. Compare to per-step events in non-profiled throughput runs for co-execution; trace timing is instrumented and must not replace throughput. Fixed-prefix microbenchmarks do not measure generation quality.'
        }, indent=2)+'\n')
    finally:
        engine.close()


if __name__ == '__main__':
    main()
