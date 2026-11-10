#!/usr/bin/env python3
"""Evaluate long-context retrieval with matched OPT control methods."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "src/modeling_opt_strict_controls.py"
sys.path.insert(0, str(HERE.parent / "transformers/src"))
sys.path.insert(0, str(HERE))

STRATEGIES = ("full", "neural_offline", "oracle", "previous_step", "layer5_reuse", "quest_capped", "quest_native")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module():
    name = "transformers.models.opt.modeling_opt_long_context_controls"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def definition(strategy):
    return {
        "full": "Full causal attention reference.",
        "neural_offline": "Historical Neuro-Fetcher: 1 encoder, 4 heads, FFN 2048, shallow-5 reuse, top-k UNION recent-20. Bidirectional predictor sees the entire prompt (including final question); offline reference, not causal per-step decoding or latency evidence.",
        "oracle": "Current-query true QK top-k, independent per head, exact configured floor budget, no recency union.",
        "previous_step": "Previous query true QK ranking, reselected at current k within previous history, exact configured floor budget, no recency union.",
        "layer5_reuse": "Layers up to reuse_layer use current-query Oracle; deeper layers reuse that layer's indices, exact configured floor budget, no recency union.",
        "quest_capped": "Per-head Quest page-bound ordering with exact configured floor token cap and no recent window.",
        "quest_native": "Quest official accuracy-style page rule; reported separately because its exceptions exceed 30%.",
    }[strategy]


def score_record(request, logits):
    candidate_ids = torch.tensor(request["candidate_ids"], device=logits.device, dtype=torch.long)
    scores = logits[0, -1].float().log_softmax(dim=-1).index_select(0, candidate_ids)
    gold_index = request["candidate_words"].index(request["answer"])
    order = scores.argsort(descending=True, stable=True)
    rank = int((order == gold_index).nonzero(as_tuple=False)[0].item()) + 1
    if not bool(torch.isfinite(scores).all()):
        raise ValueError('Non-finite candidate log probabilities')
    return {
        "id": request["id"], "task": request.get('task', 'needle_retrieval'), "length": request["length"],
        "needle_depth": request["needle_depth"], "needle_start": request["needle_start"],
        "answer": request["answer"], "candidate_words": request["candidate_words"],
        "candidate_logprobs": [float(v) for v in scores.cpu()],
        "rank": rank, "correct": rank == 1,
        "greedy_correct": int(logits[0, -1].argmax()) == request['candidate_ids'][gold_index],
        "gold_logprob": float(scores[gold_index]),
        "greedy_token_id": int(logits[0, -1].argmax()),
    }


def aggregate(records):
    def one(items):
        return {"samples": len(items),
                "top1": sum(x["correct"] for x in items) / len(items) if items else None,
                "greedy_accuracy": sum(x["greedy_correct"] for x in items) / len(items) if items else None,
                "mean_gold_logprob": sum(x["gold_logprob"] for x in items) / len(items) if items else None,
                "mean_rank": sum(x["rank"] for x in items) / len(items) if items else None}
    by_length, by_depth, by_task, cells = {}, {}, {}, {}
    for item in records:
        by_length.setdefault(str(item["length"]), []).append(item)
        by_depth.setdefault(str(item["needle_depth"]), []).append(item)
        by_task.setdefault(item['task'], []).append(item)
        cells.setdefault(f"{item['task']}/L{item['length']}/D{item['needle_depth']}", []).append(item)
    return {"overall": one(records),
            "by_length": {k: one(v) for k, v in sorted(by_length.items())},
            "by_depth": {k: one(v) for k, v in sorted(by_depth.items())},
            "by_task": {k: one(v) for k, v in sorted(by_task.items())},
            "by_task_length_depth": {k: one(v) for k, v in sorted(cells.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=HERE / "results/long_context_retrieval_combined_20261017.jsonl")
    parser.add_argument("--output-dir", type=Path,
                        default=HERE / "results/long_context_controls_20261017")
    parser.add_argument("--model-path", type=Path, default=Path("/home/liushenyu/opt-13b"))
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=[s for s in STRATEGIES if s != 'neural_offline'])
    parser.add_argument("--budget-ratio", type=float, default=0.30)
    parser.add_argument("--reuse-layer", type=int, default=5)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument('--predictor-path', type=Path, default=Path('/home/liushenyu/test/predictor_topk_0.3_iwl_epoch_20.pth'))
    parser.add_argument('--predictor-embedding-path', type=Path, default=Path('/home/liushenyu/test/opt_13b_embed_weights.pth'))
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"0", "1", "2", "3"}:
        raise ValueError("Expose exactly one authorized GPU with CUDA_VISIBLE_DEVICES=0, 1, 2, or 3")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU loading")
    if not 1 <= args.threads <= 10 or not 0 < args.budget_ratio < 1 or args.limit < 0:
        raise ValueError("Invalid threads or budget ratio")
    if 'neural_offline' in args.strategies and len(args.strategies) != 1:
        raise ValueError('Run neural_offline in its own worker to preserve independent model implementations')
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    requests = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if args.limit:
        requests = requests[:args.limit]
    if not requests:
        raise ValueError("Empty input")
    if len({r['id'] for r in requests}) != len(requests):
        raise ValueError('Duplicate sample IDs')
    for request in requests:
        if (len(request['input_ids']) != request['length'] or request['length'] < 1
                or not 0 <= request['needle_start'] < request['length']
                or request['answer'] not in request['candidate_words']
                or len(set(request['candidate_ids'])) != len(request['candidate_words'])):
            raise ValueError(f"Invalid input metadata: {request['id']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)

    def paths(strategy):
        stem = f"needle-{strategy}-b{args.budget_ratio:g}".replace(".", "p")
        if strategy == "layer5_reuse":
            stem += f"-source{args.reuse_layer}"
        if strategy.startswith("quest_"):
            stem += f"-page{args.page_size}"
        if args.limit:
            stem += f"-limit{args.limit}"
        return args.output_dir / (stem + ".jsonl"), args.output_dir / (stem + ".metrics.json")

    source_paths = [SOURCE, SOURCE.with_name('modeling_opt_neu.py'),
                    SOURCE.with_name('modeling_opt_neural_controls.py'),
                    Path(__file__).resolve(), HERE / "neural_reviewer_controls.py",
                    HERE / 'predictor_model2A.py']
    source_hashes = {str(path): sha256(path) for path in source_paths}
    input_hash = sha256(args.input)
    pending = []
    for strategy in args.strategies:
        output, metrics = paths(strategy)
        if output.exists() and metrics.exists():
            saved = json.loads(metrics.read_text())
            if (saved.get('complete') and saved.get('source_sha256') == source_hashes
                    and saved.get('input_sha256') == input_hash
                    and saved.get('model_path') == str(args.model_path)
                    and saved.get('requests') == len(requests)
                    and saved.get('budget_ratio') == args.budget_ratio
                    and saved.get('reuse_layer_one_based') == (args.reuse_layer if strategy == 'layer5_reuse' else None)
                    and saved.get('quest_page_size') == (args.page_size if strategy.startswith('quest_') else None)):
                print(f'Complete; skip {strategy}', flush=True)
                continue
        if output.exists() or metrics.exists():
            raise FileExistsError(f"Refusing to overwrite or mix results: {output}")
        pending.append(strategy)
    if not pending:
        return

    neural = pending == ['neural_offline']
    if neural:
        from neural_reviewer_controls import load_model_module
        module = load_model_module()
    else:
        module = load_module()
    from transformers import AutoTokenizer
    cls = module.OPTForCausalLM
    config = cls.config_class.from_pretrained(str(args.model_path), local_files_only=True)
    config.prefetch_strategy = "strict_control"
    config.strict_strategy = args.strategies[0]
    config.strict_ratio = args.budget_ratio
    config.strict_reuse_layer = args.reuse_layer - 1
    config.quest_page_size = args.page_size
    if not 1 <= args.reuse_layer <= config.num_hidden_layers or args.page_size < 1:
        raise ValueError('Invalid reuse layer or page size')
    if neural:
        config.prefetch_strategy = 'neural'
        config.control_strategy = 'neural'
        config.prefetch_budget_ratio = args.budget_ratio
        config.num_predict_layers = 5
        config.prefetch_layers_list = list(range(5))
    if config.max_position_embeddings < max(r["length"] for r in requests):
        raise ValueError("Input exceeds checkpoint learned position embedding limit")
    print(f"Loading {args.model_path}; requests={len(requests)}; GPU={visible}; strategies={args.strategies}", flush=True)
    load_start = time.perf_counter()
    model, loading = cls.from_pretrained(
        str(args.model_path), config=config, torch_dtype=torch.float16,
        device_map={"": 0}, local_files_only=True, output_loading_info=True,
    )
    allowed = ('predictor.', 'predictor_embed_13b.') if neural else ()
    bad_missing = [k for k in loading.get('missing_keys', []) if not k.startswith(allowed)]
    if bad_missing or any(loading.get(k) for k in ("unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Checkpoint must load without mismatches: {loading}")
    predictor_hashes = None
    if neural:
        model.predictor.load_state_dict(torch.load(args.predictor_path, map_location='cpu'), strict=True)
        embedding = torch.load(args.predictor_embedding_path, map_location='cpu')
        with torch.no_grad():
            if embedding.shape != model.predictor_embed_13b.weight.shape:
                raise ValueError('Predictor embedding shape mismatch')
            model.predictor_embed_13b.weight.copy_(embedding.to(model.predictor_embed_13b.weight))
            if not torch.equal(model.predictor_embed_13b.weight, embedding.to(model.predictor_embed_13b.weight)):
                raise ValueError('Predictor embedding restore failed')
        del embedding
        model.predictor.to(model.device, dtype=torch.float32).eval()
        predictor_hashes = {str(p): sha256(p) for p in (args.predictor_path, args.predictor_embedding_path)}
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), use_fast=False, local_files_only=True)
    attention_modules = [layer.self_attn for layer in model.model.decoder.layers]
    shared_state = {}
    for attention in attention_modules:
        attention.control_state = shared_state
    load_seconds = time.perf_counter() - load_start

    for strategy in pending:
        for attention in attention_modules:
            attention.strict_strategy = strategy
            attention.control_stats = None
        shared_state.clear()
        output_path, metrics_path = paths(strategy)
        records, timings = [], []
        started = time.perf_counter()
        print(f"BEGIN {strategy}: {definition(strategy)}", flush=True)
        with output_path.open("x", encoding="utf-8") as target, torch.inference_mode():
            for number, request in enumerate(requests, 1):
                ids = torch.tensor(request["input_ids"], dtype=torch.long,
                                   device=model.device).unsqueeze(0)
                torch.cuda.synchronize()
                begin = time.perf_counter()
                output = model(ids, use_cache=False)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - begin
                record = score_record(request, output.logits)
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
                timings.append({"id": request["id"], "tokens": len(request["input_ids"]), "seconds": seconds})
                del output, ids
                if number == 1 or number % 10 == 0 or number == len(requests):
                    target.flush()
                    print(f"{strategy} processed={number}/{len(requests)} wall_seconds={time.perf_counter()-started:.1f}", flush=True)
        if strategy == "full":
            candidates = sum(r['length'] * (r['length'] + 1) // 2 for r in requests) * config.num_hidden_layers * config.num_attention_heads
            selected = expected = candidates
            mismatch = above = 0
            actual_budget = 1.0
        elif neural:
            counts = [a.control_stats.cpu().tolist() for a in attention_modules]
            selected, candidates = [sum(row[i] for row in counts) for i in range(2)]
            expected = mismatch = above = None
            actual_budget = selected / candidates
        else:
            counts = [a.control_stats.cpu().tolist() for a in attention_modules]
            selected, candidates, expected, mismatch, above = [sum(row[i] for row in counts) for i in range(5)]
            if strategy != "quest_native" and mismatch:
                raise RuntimeError(f"Exact budget violation: {mismatch} rows")
            actual_budget = selected / candidates
        metrics = {
            "complete": True, "suite": "needle_in_a_haystack", "strategy": strategy,
            "definition": definition(strategy), "model_path": str(args.model_path),
            "requests": len(requests), "budget_ratio": args.budget_ratio,
            "recent_window": 20 if neural else 0, "index_scope": "shared" if neural else "per_head",
            "predictor_sha256": predictor_hashes,
            "evaluation_mode": "Uncached, unpadded all-position sparse causal attention; predictor, when present, is historical bidirectional whole-prompt offline reference. No gold answer token is appended.",
            "timing_scope": "Diagnostic forward wall time only; not KV offload or deployed decoding latency",
            "reuse_layer_one_based": args.reuse_layer if strategy == "layer5_reuse" else None,
            "quest_page_size": args.page_size if strategy.startswith("quest_") else None,
            "quest_native_exceptions": strategy == "quest_native",
            "selected_entries": selected, "candidate_entries": candidates,
            "expected_floor_entries": expected, "actual_budget_ratio": actual_budget,
            "rows_with_budget_mismatch": mismatch, "rows_above_budget": above,
            "aggregate": aggregate(records), "per_request": timings,
            "loading_info": loading, "load_seconds": load_seconds,
            "forward_seconds": sum(t["seconds"] for t in timings),
            "wall_seconds": time.perf_counter() - started,
            "cuda_visible_devices": visible, "torch_num_threads": torch.get_num_threads(),
            "dtype": "float16", "source_sha256": source_hashes, "input_sha256": input_hash,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "max_position_embeddings": config.max_position_embeddings,
            "total_layers": config.num_hidden_layers, "num_attention_heads": config.num_attention_heads,
            "budget_rounding": 'max(1, floor(ratio*prefix)) for strict controls; historical neural ceil top-k union recent-20 with ties; Quest native exceptions separately labeled',
        }
        with metrics_path.open("x", encoding="utf-8") as target:
            json.dump(metrics, target, indent=2)
        print(f"END {strategy} top1={metrics['aggregate']['overall']['top1']:.4f} actual_budget={actual_budget:.6f}", flush=True)


if __name__ == "__main__":
    main()
