#!/usr/bin/env python3
"""Measure memory using original FlexGen or the accepted v8 baseline adapters.

No source, symlink or checkpoint is rewritten. Selection/cache adapters for
InfiniGen and H2O are copied from v8; effective parameters are reported.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from reviewer_memory_v9 import snapshot, combined_memory, digest

ROOT = Path(__file__).resolve().parents[1]
FLEX = ROOT / 'speedup/flexgen'
WEIGHTS = Path('/home/liushenyu/opt-13b-flexgen-np')
TOKENIZER = Path('/home/liushenyu/opt-13b')
PROMPT = FLEX / 'pg19_firstbook.txt'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('flexgen', 'infinigen', 'h2o'), required=True)
    parser.add_argument('--gpu', type=int, choices=(2, 3), required=True)
    parser.add_argument('--prompt-len', type=int, required=True)
    parser.add_argument('--gen-len', type=int, default=128)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--threads', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        raise RuntimeError('Expose only the assigned GPU')
    if len(os.sched_getaffinity(0)) > 10 or not 1 <= args.threads <= 10:
        raise RuntimeError('At most ten CPU cores per experiment')
    if args.gen_len < 2 or args.prompt_len < 16 or args.prompt_len + args.gen_len > 2048:
        raise ValueError('Invalid OPT context/generation lengths')
    if args.output.exists():
        raise FileExistsError(args.output)
    if not (WEIGHTS / 'opt-13b-np/decoder.layers.0.self_attn.q_proj.weight').is_file():
        raise FileNotFoundError(WEIGHTS)

    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261027)
    torch.cuda.manual_seed_all(20261027)
    sys.path.insert(0, str(FLEX))
    sys.path.insert(0, str(FLEX / 'infinigen'))
    subdir, frontend_file, backend_file = {
        'flexgen': ('original', 'flex_opt.py', 'pytorch_backend.py'),
        'infinigen': ('infinigen', 'flex_opt_v7.py', 'pytorch_backend_v7.py'),
        'h2o': ('h2o', 'flex_opt.py', 'pytorch_backend.py'),
    }[args.method]
    backend_path = FLEX / subdir / backend_file
    frontend_path = FLEX / subdir / frontend_file
    load('flexgen.pytorch_backend', backend_path)
    native = load('flexgen.memory_v9_' + args.method, frontend_path)
    original_tokenizer = AutoTokenizer.from_pretrained
    AutoTokenizer.from_pretrained = lambda *a, **kw: original_tokenizer(
        str(TOKENIZER), use_fast=False, local_files_only=True,
        padding_side=kw.get('padding_side', 'right'))
    original_inputs = native.get_inputs
    native.get_inputs = lambda length, count, tok, path: original_inputs(
        args.prompt_len if length == 2048 else length, count, tok, path)

    # Accepted v8 H2O adapter, unchanged. Memory measurements do not redesign it.
    if args.method == 'h2o':
        original_cache = native.SelfAttention.load_cache

        def load_cache(self, cache_home, cache_read_buf, i):
            if i == 0 or self.policy.compress_cache or self.policy.cpu_cache_compute or self.policy.hh_all:
                return original_cache(self, cache_home, cache_read_buf, i)
            k_home, v_home, acc = cache_home.val
            length = min(self.hh_k * 2 - 1, self.task.prompt_len) + 1 + i
            self.prefetch_kv.data[0, :length].copy_(k_home.data[:length])
            self.prefetch_kv.data[1, :length].copy_(v_home.data[:length])
            shape = (length, k_home.shape[1], k_home.shape[2])
            k_copy = native.TorchTensor(shape, k_home.dtype, self.prefetch_kv.data[0, :length], k_home.device)
            v_copy = native.TorchTensor(shape, v_home.dtype, self.prefetch_kv.data[1, :length], v_home.device)
            indices = (slice(0, length), slice(0, k_home.shape[1]))
            cache_read_buf.store((k_copy.smart_copy(self.attention_compute, indices),
                                  v_copy.smart_copy(self.attention_compute, indices),
                                  acc.smart_copy(self.attention_compute, indices)))

        native.SelfAttention.load_cache = load_cache

    observations = {}
    original_generate = native.OptLM.generate
    original_compute = native.OptLM.compute_layer

    def compute_layer(self, i, j, k, *extra, **kw):
        if observations.get('measuring') and i == 1 and j == 0 and k == 0 and 'prefill' not in observations:
            observations['prefill'] = snapshot(torch)
            torch.cuda.reset_peak_memory_stats()
        return original_compute(self, i, j, k, *extra, **kw)

    def generate(self, inputs, max_new_tokens=32, *extra, **kw):
        if max_new_tokens == 1:
            return original_generate(self, inputs, max_new_tokens, *extra, **kw)
        observations['before'] = snapshot(torch)
        torch.cuda.reset_peak_memory_stats()
        observations['measuring'] = True
        output = original_generate(self, inputs, max_new_tokens, *extra, **kw)
        observations['decode'] = snapshot(torch)
        observations['measuring'] = False
        if output.shape != (args.batch, args.prompt_len + args.gen_len):
            raise RuntimeError('Unexpected generated token shape')
        observations['output_shape'] = list(output.shape)
        observations['output_token_ids'] = output[:, -args.gen_len:].tolist()
        return output

    native.OptLM.compute_layer = compute_layer
    native.OptLM.generate = generate
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scratch = args.output.parent / (args.output.stem + '_offload')
    argv = ['--model', 'facebook/opt-13b', '--path', str(WEIGHTS),
            '--offload-dir', str(scratch), '--prompt-len', str(args.prompt_len),
            '--gen-len', str(args.gen_len), '--gpu-batch-size', str(args.batch),
            '--num-gpu-batches', '1', '--percent', '100', '0', '0', '100', '100', '0',
            '--overlap', 'false', '--warmup-input-path', str(PROMPT),
            '--test-input-path', str(PROMPT), '--verbose', '0', '--no-log']
    parameters = dict(weights_gpu_percent=100, kv_cpu_percent=100, overlap=False,
                      warmup_generated_tokens=1, measured_runs=1)
    if args.method in ('infinigen', 'h2o'):
        # v8 applies this capacity to both methods. Record the effective value.
        effective_cap = args.prompt_len + args.gen_len + 1
        argv += ['--max-num-kv', str(effective_cap)]
        parameters['effective_max_num_kv'] = effective_cap
    if args.method == 'infinigen':
        argv += ['--alpha', '7', '--partial-weight-ratio', '0.3', '--prefetch-strategy', 'infinigen']
        parameters.update(alpha=7, partial_weight_ratio=.3,
                          historical_nominal_max_num_kv=int((args.prompt_len + args.gen_len) * .3) + 1)
    if args.method == 'h2o':
        argv += ['--hh-ratio', '.15', '--hh-all', 'false', '--hh-long-seq', 'false']
        parameters.update(heavy_ratio=.15, native_recent_policy=True, decode_length_adapter='v8')
    native_parser = argparse.ArgumentParser()
    native.add_parser_arguments(native_parser)
    native.run_flexgen(native_parser.parse_args(argv))
    from flexgen.timer import timers
    costs = [float(v) for v in timers('generate').costs]
    if len(costs) != args.gen_len or 'prefill' not in observations:
        raise RuntimeError('Incomplete generation/timing or memory observation')
    prefill_s, decode_s = costs[0], sum(costs[1:])
    result = dict(complete=True, method=args.method, gpu=args.gpu, model='OPT-13B',
                  precision='FP16', prompt_len=args.prompt_len, generation_len=args.gen_len,
                  batch_size=args.batch, runtime='native FlexGen, accepted v8 adapters where applicable',
                  prefill_s=prefill_s, decode_s=decode_s, total_s=prefill_s+decode_s,
                  decode_tokens_per_s=args.batch*(args.gen_len-1)/decode_s,
                  parameters=parameters,
                  memory=combined_memory(observations['before'], observations['prefill'], observations['decode']),
                  source_sha256={str(p): digest(p) for p in (Path(__file__), frontend_path, backend_path)},
                  prompt_sha256=digest(PROMPT), weights_root=str(WEIGHTS),
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  output_shape=observations['output_shape'],
                  generated_token_ids=observations['output_token_ids'])
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'generated_token_ids'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
