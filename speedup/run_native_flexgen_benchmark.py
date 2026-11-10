#!/usr/bin/env python3
"""Run one original FlexGen method without touching package symlinks.

Methods load their native, isolated FlexGen frontend/backend implementation.
Each invocation binds one process to one GPU. Workload summary is written as
JSON after the native runner reports its prefill/decode/total timers.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLEXGEN = ROOT / 'speedup'
FLEX_ROOT = FLEXGEN / 'flexgen'
MODEL_WEIGHTS = Path('/home/liushenyu/opt-13b-flexgen-np/opt-13b-np')
MODEL_CONFIG = Path('/home/liushenyu/opt-13b')
INFINIGEN_SOURCE = ROOT / 'accuracy/setup/opt-model/opt-13b'
PREDICTOR = Path('/home/liushenyu/test/predictor_topk_0.3_iwl_epoch_20.pth')
TOKENIZER = Path('/home/liushenyu/opt-13b')
PROMPT_FILE = FLEX_ROOT / 'pg19_firstbook.txt'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=('infinigen', 'neural', 'h2o'), required=True)
    p.add_argument('--gpu', type=int, choices=(2, 3), required=True)
    p.add_argument('--prompt-len', type=int, required=True)
    p.add_argument('--gen-len', type=int, default=128)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=10)
    args = p.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible != str(args.gpu):
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to this one GPU; model is never sharded across cards')
    if not 1 <= args.threads <= 10 or len(os.sched_getaffinity(0)) > 10:
        raise RuntimeError('Each lane is limited to at most ten CPU cores')
    if args.prompt_len < 16 or args.prompt_len + args.gen_len > 2048:
        raise ValueError('OPT-13B context limit is 2048')
    if args.output.exists():
        raise FileExistsError(args.output)
    if not MODEL_WEIGHTS.is_dir() or not MODEL_CONFIG.is_dir() or not PROMPT_FILE.is_file():
        raise FileNotFoundError('Missing local OPT-13B FlexGen weights, config, or prompt')
    if args.method == 'infinigen' and not INFINIGEN_SOURCE.is_dir():
        raise FileNotFoundError('Missing transformed InfiniGen checkpoint')
    if args.method == 'neural' and not PREDICTOR.is_file():
        raise FileNotFoundError(PREDICTOR)

    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261027)
    torch.cuda.manual_seed_all(20261027)

    sys.path.insert(0, str(FLEX_ROOT))
    sys.path.insert(0, str(FLEX_ROOT / 'infinigen'))
    if args.method == 'h2o':
        backend_path = FLEX_ROOT / 'h2o/pytorch_backend.py'
        frontend_path = FLEX_ROOT / 'h2o/flex_opt.py'
        frontend_name = 'flexgen.flex_opt_review_h2o'
    else:
        backend_path = FLEX_ROOT / 'infinigen/pytorch_backend_v7.py'
        frontend_path = FLEX_ROOT / 'infinigen/flex_opt_v7.py'
        frontend_name = 'flexgen.flex_opt_review_inf'
    backend = load_module('flexgen.pytorch_backend', backend_path)
    frontend = load_module(frontend_name, frontend_path)

    # The archived native frontends name OPT-30B on the Hub. Resolve that
    # tokenizer request to the local OPT-13B tokenizer without network access.
    original_from_pretrained = AutoTokenizer.from_pretrained
    AutoTokenizer.from_pretrained = lambda *a, **kw: original_from_pretrained(
        str(TOKENIZER), use_fast=False, local_files_only=True,
        padding_side=kw.get('padding_side', 'right'))
    original_get_inputs = frontend.get_inputs
    def get_inputs(length, num_prompts, tokenizer, path):
        actual_length = args.prompt_len if length == 2048 else length
        return original_get_inputs(actual_length, num_prompts, tokenizer, path)
    frontend.get_inputs = get_inputs

    argv = [
        '--model', 'facebook/opt-13b', '--path', str(MODEL_WEIGHTS),
        '--prompt-len', str(args.prompt_len), '--gen-len', str(args.gen_len),
        '--gpu-batch-size', str(args.batch), '--num-gpu-batches', '1',
        '--percent', '100', '0', '0', '100', '100', '0',
        '--overlap', 'false', '--warmup-input-path', str(PROMPT_FILE),
        '--test-input-path', str(PROMPT_FILE), '--verbose', '0', '--no-log',
    ]
    if args.method == 'h2o':
        argv += ['--hh-ratio', '0.15', '--hh-all', 'false', '--hh-long-seq', 'false']
    if args.method == 'infinigen':
        argv += ['--alpha', '7', '--partial-weight-ratio', '0.3',
                 '--max-num-kv', str(max(1, int((args.prompt_len + args.gen_len) * 0.3) + 1)),
                 '--prefetch-strategy', 'infinigen']
        # InfiniGen's native path uses the transformed checkpoint.
        argv[argv.index(str(MODEL_WEIGHTS))] = str(INFINIGEN_SOURCE)
    elif args.method == 'neural':
        argv += ['--prefetch-strategy', 'neural', '--predictor-path', str(PREDICTOR),
                 '--prefetch-budget-ratio', '0.3', '--max-num-kv', str(args.prompt_len)]
    else:
        argv += ['--max-num-kv', str(args.prompt_len)]

    # Parse the unmodified native command-line schema, then execute its own
    # run_flexgen implementation in-process so metrics remain method-native.
    parser = argparse.ArgumentParser()
    frontend.add_parser_arguments(parser)
    native_args = parser.parse_args(argv)
    if hasattr(frontend, 'get_inputs'):
        frontend.get_inputs = get_inputs
    started = time.perf_counter()
    frontend.run_flexgen(native_args)
    elapsed = time.perf_counter() - started
    try:
        from flexgen.utils import timers
        costs = timers('generate').costs
        prefill_s = float(costs[0])
        decode_s = float(sum(costs[1:]))
        total_s = prefill_s + decode_s
    except (ImportError, IndexError, TypeError):
        raise RuntimeError('Native FlexGen timer values were unavailable')
    output = dict(
        complete=True, method=args.method, runtime='original FlexGen method-specific runtime',
        gpu=args.gpu, model='OPT-13B', model_path=str(MODEL_CONFIG),
        precision='FP16', prompt_len=args.prompt_len, generation_len=args.gen_len,
        batch_size=args.batch, elapsed_wall_s=elapsed,
        prefill_s=prefill_s, decode_s=decode_s, total_s=total_s,
        decode_tokens_per_s=args.batch * max(args.gen_len - 1, 0) / max(decode_s, 1e-12),
        end_to_end_tokens_per_s=args.batch * args.gen_len / max(total_s, 1e-12),
        parameters={'infinigen': {'alpha': 7, 'partial_weight_ratio': .3,
                    'max_num_kv': max(1, int((args.prompt_len + args.gen_len) * .3) + 1),
                                  'source': str(INFINIGEN_SOURCE)},
                    'neural': {'checkpoint': str(PREDICTOR), 'budget': .3,
                               'recent_union': 20},
                    'h2o': {'heavy_ratio': .15, 'native_recent_policy': True,
                            'first_two_layers_full': True}}[args.method],
        note='Metrics use the native FlexGen generate timer after warmup.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
