#!/usr/bin/env python3
"""Run existing InfiniGen, H2O and KV fake-quantization without changing sources."""
import argparse
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'transformers/src'))
sys.path.insert(0, str(HERE))
from run_long_context_controls import score_record, aggregate
from summarize_long_context_controls import sha


def load_source(path, suffix):
    name = 'transformers.models.opt.modeling_opt_native_' + suffix
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--strategy', required=True, choices=['infinigen', 'h2o', 'quant4'])
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--smoke', action='store_true', help='One sample per task and length')
    p.add_argument('--threads', type=int, default=10)
    args = p.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible not in ('0', '1', '2', '3') or not torch.cuda.is_available():
        raise RuntimeError('Expose one authorized CUDA GPU (0–3)')
    if not 1 <= args.threads <= 10 or len(os.sched_getaffinity(0)) > 10:
        raise ValueError('Limit worker to 10 CPU cores and threads')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = HERE.parent / ('src/modeling_opt_inf.py' if args.strategy == 'infinigen' else 'src/modeling_opt_orig.py')
    model_path = HERE.parent / 'setup/opt-model/opt-13b' if args.strategy == 'infinigen' else Path('/home/liushenyu/opt-13b')
    partial_dir = HERE.parent / 'setup/weights/opt-13b_0.3'
    sources = [Path(__file__).resolve(), source, HERE / 'run_long_context_controls.py',
               HERE / 'summarize_long_context_controls.py']
    if args.strategy == 'h2o':
        sources.append(HERE / 'utils_lm_eval/modify_opt.py')
    source_hashes = {str(x): sha(x) for x in sources}
    inputs = {
        'qa': HERE / 'results/long_context_retrieval_combined_20261017.jsonl',
        'cloze': HERE / 'results/long_context_cloze_balanced_20261018.jsonl',
    }
    suffix = '-smoke' if args.smoke else ''
    jobs = []
    for protocol, path in inputs.items():
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        if args.smoke:
            seen, subset = set(), []
            for row in rows:
                cell = (row['task'], row['length'])
                if cell not in seen:
                    seen.add(cell)
                    subset.append(row)
            rows = subset
        out = args.output_dir / f'{protocol}-{args.strategy}{suffix}.jsonl'
        metrics = out.with_suffix('.metrics.json')
        if out.exists() or metrics.exists():
            if out.exists() and metrics.exists():
                saved = json.loads(metrics.read_text())
                if (saved.get('complete') and saved['input_sha256'] == sha(path)
                        and saved['source_sha256'] == source_hashes and saved['requests'] == len(rows)):
                    print(f'Complete; skip {protocol} {args.strategy}', flush=True)
                    continue
            raise FileExistsError(f'Refusing to overwrite {out}')
        jobs.append((protocol, path, rows, out, metrics))
    if not jobs:
        return
    module = load_source(source, args.strategy)
    cls = module.OPTForCausalLM
    config = cls.config_class.from_pretrained(str(model_path), local_files_only=True)
    model, loading = cls.from_pretrained(str(model_path), config=config, torch_dtype=torch.float16,
                                        device_map={'': 0}, local_files_only=True, output_loading_info=True)
    expected_inf_bias = all(k.endswith(('self_attn.q_proj.bias', 'self_attn.k_proj.bias'))
                            for k in loading.get('unexpected_keys', []))
    if (loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs')
            or (loading.get('unexpected_keys') and (args.strategy != 'infinigen' or not expected_inf_bias))):
        raise RuntimeError(f'Checkpoint mismatch: {loading}')
    model.eval()
    # Check Full behavior BEFORE enabling a baseline. Existing controls use
    # FP16 softmax, whereas these native sources upcast softmax to FP32;
    # InfiniGen also folds/scales/rotates QK in FP16. This is a numerical
    # cross-check, not a claim of bitwise equivalence between implementations.
    verification = []
    with torch.inference_mode():
        for protocol, path, rows, _, _ in jobs:
            reference_dir = 'long_context_cloze_controls_20261018' if protocol == 'cloze' else 'long_context_controls_balanced_20261017'
            refs = {r['id']: r for r in map(json.loads, (HERE / 'results' / reference_dir / 'needle-full-b0p3.jsonl').read_text().splitlines())}
            seen = set()
            for row in rows:
                cell = (row['task'], row['length'])
                if cell in seen:
                    continue
                seen.add(cell)
                for layer in model.model.decoder.layers:
                    if hasattr(layer.self_attn, 'previous_hidden_states'):
                        layer.self_attn.previous_hidden_states = None
                ids = torch.tensor([row['input_ids']], device='cuda')
                output = model(ids, use_cache=False)
                scored = score_record(row, output.logits)
                ref = refs[row['id']]
                delta = max(abs(a-b) for a,b in zip(scored['candidate_logprobs'], ref['candidate_logprobs']))
                verification.append(dict(protocol=protocol, id=row['id'], max_logprob_delta=delta,
                                         rank=scored['rank'], reference_rank=ref['rank']))
                if delta > (0.25 if args.strategy == 'infinigen' else 0.1) or scored['rank'] != ref['rank']:
                    raise RuntimeError(f'Full equivalence check failed: {verification[-1]}')
                del output, ids
    print(f'Full numerical cross-check passed: {len(verification)} samples', flush=True)
    counters = {'selected': 0, 'candidate': 0, 'final_selected': 0, 'final_candidate': 0}

    def count(mask):
        h, n, m = mask.shape
        if n != m:
            raise ValueError('Only uncached square causal evaluation is supported')
        counters['selected'] += int(mask.sum())
        counters['candidate'] += h*n*(n+1)//2
        counters['final_selected'] += int(mask[:, -1].sum())
        counters['final_candidate'] += h*n

    partial_hashes = None
    if args.strategy == 'infinigen':
        partial_hashes = {}
        for i, layer in enumerate(model.model.decoder.layers):
            a = layer.self_attn
            path = partial_dir / f'partial_weight_q_{i}.pt'
            partial_hashes[str(path)] = sha(path)
            a.partial_weight_q = torch.load(path, map_location='cpu').to(model.device, dtype=torch.float16)
            a.partial_weight_ratio, a.alpha, a.capacity, a.budget = 0.3, 7, 1.0, 0.3
            native = a.kv_cache_mask
            def observed(self, attn, native=native):
                mask, density = native(attn)
                count(mask == 0)
                return mask, density
            a.kv_cache_mask = types.MethodType(observed, a)
    elif args.strategy == 'h2o':
        from utils_lm_eval import modify_opt
        config.heavy_ratio = config.recent_ratio = 0.15
        model = modify_opt.convert_kvcache_opt_heavy_recent(model, config).eval()
        native = modify_opt.local_heavy_hitter_mask
        def observed(attn, heavy_budget, no_padding_seq_length=None):
            heavy = native(attn, heavy_budget, no_padding_seq_length)
            recent = torch.triu(torch.ones_like(heavy), diagonal=-int(0.15 * attn.shape[-1]))
            count(torch.tril(heavy | recent))
            return heavy
        modify_opt.local_heavy_hitter_mask = observed
        # Native implementation saves a diagnostic attention clone; release after use.
        for layer in model.model.decoder.layers[2:]:
            layer.self_attn.register_forward_hook(lambda a, inputs, outputs: setattr(a, 'attn', None))
    else:
        for layer in model.model.decoder.layers[2:]:
            layer.self_attn.enable_quant = True
            layer.self_attn.qbits = 4

    for protocol, path, rows, out, metrics in jobs:
        for key in counters:
            counters[key] = 0
        records, timings = [], []
        started = time.perf_counter()
        with out.open('x') as target, torch.inference_mode():
            for number, row in enumerate(rows, 1):
                if len(row['input_ids']) != row['length'] or row['length'] > config.max_position_embeddings:
                    raise ValueError(row['id'])
                for layer in model.model.decoder.layers:
                    if hasattr(layer.self_attn, 'previous_hidden_states'):
                        layer.self_attn.previous_hidden_states = None
                ids = torch.tensor([row['input_ids']], device='cuda')
                torch.cuda.synchronize()
                begin = time.perf_counter()
                output = model(ids, use_cache=False)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-begin
                record = score_record(row, output.logits)
                target.write(json.dumps(record, ensure_ascii=False)+'\n')
                target.flush()
                records.append(record)
                timings.append(dict(id=row['id'], tokens=row['length'], seconds=seconds))
                del output, ids
                if number == 1 or number % 10 == 0 or number == len(rows):
                    print(f'{protocol} {args.strategy} processed={number}/{len(rows)} seconds={time.perf_counter()-started:.1f}', flush=True)
        # Both native sparse algorithms leave the first two layers Full.
        all_candidate = sum(r['length']*(r['length']+1)//2 for r in rows)*1600
        all_final = sum(r['length'] for r in rows)*1600
        if args.strategy == 'quant4':
            selected, final_selected = all_candidate, all_final
        else:
            assert counters['candidate'] == all_candidate*38//40
            selected = counters['selected'] + all_candidate*2//40
            final_selected = counters['final_selected'] + all_final*2//40
        data = dict(complete=True, strategy=args.strategy, protocol=protocol, smoke=args.smoke,
                    requests=len(rows), model_path=str(model_path), underlying_model='/home/liushenyu/opt-13b',
                    input_path=str(path), input_sha256=sha(path), source_sha256=source_hashes,
                    partial_weight_sha256=partial_hashes, loading_info=loading,
                    expected_unused_projection_biases=(args.strategy == 'infinigen'),
                    full_equivalence_checks=verification,
                    nominal_budget_ratio=0.3 if args.strategy != 'quant4' else None,
                    native_configuration={'infinigen': dict(alpha=7, capacity=1., budget=.3, partial_weight_ratio=.3, first_full_layers=2, source=str(source)),
                                          'h2o': dict(heavy_ratio=.15, recent_ratio=.15, first_full_layers=2, recent_includes_current=True),
                                          'quant4': dict(qbits=4, group_size=64, first_full_layers=2, kv_fake_quantization=True,
                                                         ideal_payload_fraction_fp16=.05+.95*4/16)}[args.strategy],
                    recent20_added=False, selected_entries=selected, candidate_entries=all_candidate,
                    actual_budget_ratio=selected/all_candidate, final_selected_entries=final_selected,
                    final_candidate_entries=all_final, final_step_density=final_selected/all_final,
                    aggregate=aggregate(records), per_request=timings,
                    evaluation_mode='Uncached, unpadded all-position causal forward, batch 1; same inputs and scoring as existing controls; no gold answer appended.',
                    budget_note='Native whole-sequence capacity, warmup and Full layers retained; not strict per-prefix 30%. Quant retains all tokens and simulates KV rounding in FP16, with no measured memory saving.',
                    timing_scope='Diagnostic only, not deployed decoding or transfer latency',
                    cuda_visible_devices=visible, cpu_affinity=sorted(os.sched_getaffinity(0)),
                    torch_num_threads=torch.get_num_threads(), max_position_embeddings=config.max_position_embeddings,
                    total_layers=config.num_hidden_layers, num_attention_heads=config.num_attention_heads,
                    forward_seconds=sum(t['seconds'] for t in timings), wall_seconds=time.perf_counter()-started)
        with metrics.open('x') as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
        print(f'END {protocol} {args.strategy}: top1={data["aggregate"]["overall"]["top1"]:.4f} density={selected/all_candidate:.4f}', flush=True)


if __name__ == '__main__':
    main()
