#!/usr/bin/env python3
"""Run strictly budgeted 13B controls and two explicitly named Quest variants."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "src/modeling_opt_strict_controls.py"
sys.path.insert(0, str(HERE.parent / "transformers/src"))
sys.path.insert(0, str(HERE))
from neural_reviewer_controls import result_record

STRATEGIES = ("oracle", "previous_step", "layer5_reuse", "quest_native", "quest_capped")


def load_module():
    name = "transformers.models.opt.modeling_opt_strict_controls"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def definitions(strategy):
    return {
        "oracle": "Each head selects its current query's true QK top-k independently; exact cardinality, no forced recent/current tokens.",
        "previous_step": "Each head selects using the preceding query's true QK ranking within that preceding history at CURRENT k; first query bootstraps itself. Idealized, non-recursive, no recency union.",
        "layer5_reuse": "Layers 1-5 use per-head current-query Oracle; layers 6-40 reuse layer 5's current-query selected indices for the corresponding head. One-based layer numbering.",
        "quest_native": "Quest official ACCURACY selection adapted to OPT: per-head signed-key page bounds, min(visible pages, max(3, floor(token budget/page size))) pages; first 2 layers Full. No added recent window. Causal decode selection emulated at every token position.",
        "quest_capped": "OPT budget-capped Quest adaptation: same per-head page bounds, exact token cap, partial last selected page, deterministic earliest-position ties; no first-2-layer Full exception, no recent window. Causal decode selection emulated at every token position.",
    }[strategy]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["winogrande", "openbookqa", "copa", "rte"])
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--model-path", type=Path, default=Path("/home/liushenyu/opt-13b"))
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--budget-ratio", type=float, default=.3)
    parser.add_argument("--reuse-layer", type=int, default=5, help="One-based shallow source layer")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/strict_controls_20261017")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.budget_ratio <= 1 or args.limit < 0 or args.page_size < 1:
        raise ValueError("Invalid budget, limit, or page size")
    if args.budget_ratio == 1 and "previous_step" in args.strategies:
        raise ValueError("Previous-step cannot fill a 100% current-history budget using only previous history")
    if not 1 <= args.threads <= 10:
        raise ValueError("Each GPU worker is limited to at most 10 CPU threads")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"0", "1", "2", "3"}:
        raise ValueError("Expose exactly one authorized GPU with CUDA_VISIBLE_DEVICES=0, 1, 2, or 3")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU loading")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.budget_ratio:g}".replace(".", "p")
    input_path = args.input_path or HERE / f"results/{args.task}-{args.shots}.jsonl"
    input_sha = file_sha(input_path)
    source_paths = [SOURCE, SOURCE.with_name("modeling_opt_neu.py"),
                    Path(__file__).resolve(), HERE / "neural_reviewer_controls.py"]
    source_hashes = {str(p): file_sha(p) for p in source_paths}

    def paths(strategy):
        stem = f"{args.task}-{args.shots}-{args.model_path.name}-{strategy}-per_head-b{tag}-r0-floor"
        if strategy == "layer5_reuse":
            stem += f"-source{args.reuse_layer}"
        if strategy.startswith("quest_"):
            stem += f"-page{args.page_size}"
        if args.limit:
            stem += f"-limit{args.limit}"
        return tuple(args.output_dir / (stem + ending) for ending in (".jsonl", ".metrics.json"))

    pending = []
    for strategy in args.strategies:
        output, metrics = paths(strategy)
        if metrics.exists() and output.exists():
            saved = json.loads(metrics.read_text())
            compatible = (saved.get("complete") and saved.get("strategy") == strategy
                          and saved.get("source_sha256") == source_hashes
                          and saved.get("input_sha256") == input_sha
                          and saved.get("model_path") == str(args.model_path)
                          and saved.get("budget_ratio") == args.budget_ratio)
            if compatible:
                print(f"Complete; skip {output.name}", flush=True)
                continue
        if output.exists() or metrics.exists():
            raise FileExistsError(f"Partial or incompatible result; use another output directory: {output}")
        pending.append(strategy)
    if not pending:
        return
    requests = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    if args.limit:
        requests = requests[:args.limit]
    if not requests:
        raise ValueError("Empty task input")
    module = load_module()
    from transformers import AutoTokenizer
    cls = module.OPTForCausalLM
    config = cls.config_class.from_pretrained(str(args.model_path), local_files_only=True)
    config.prefetch_strategy = "strict_control"
    config.strict_strategy = pending[0]
    config.strict_ratio = args.budget_ratio
    config.strict_reuse_layer = args.reuse_layer - 1
    config.quest_page_size = args.page_size
    if not 1 <= args.reuse_layer <= config.num_hidden_layers:
        raise ValueError("Reuse layer outside model")
    print(f"Loading {args.model_path}; task={args.task}; requests={len(requests)}; GPU={visible}", flush=True)
    load_start = time.perf_counter()
    model, loading = cls.from_pretrained(
        str(args.model_path), config=config, torch_dtype=torch.float16,
        device_map={"": 0}, local_files_only=True, output_loading_info=True,
    )
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Checkpoint must load without mismatches: {loading}")
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
        print(f"BEGIN {strategy}: {definitions(strategy)}", flush=True)
        started = time.perf_counter()
        timings = []
        with output_path.open("x", encoding="utf-8") as target, torch.inference_mode():
            for number, request in enumerate(requests, 1):
                ids = tokenizer(request["prompt"], add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
                if ids.shape[1] > config.max_position_embeddings:
                    raise ValueError("Input too long; no implicit truncation")
                torch.cuda.synchronize()
                begin = time.perf_counter()
                output = model(ids, use_cache=False)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - begin
                target.write(json.dumps(result_record(request, ids, output.logits, tokenizer), ensure_ascii=False) + "\n")
                timings.append({"tokens": int(ids.shape[1]), "seconds": seconds})
                del output
                if number == 1 or number % 100 == 0 or number == len(requests):
                    target.flush()
                    print(f"{strategy} processed={number}/{len(requests)} wall_seconds={time.perf_counter()-started:.1f}", flush=True)
        counts = [a.control_stats.cpu().tolist() for a in attention_modules]
        selected, candidates, expected, mismatch, above = [sum(row[i] for row in counts) for i in range(5)]
        if strategy != "quest_native" and mismatch:
            raise RuntimeError(f"Exact budget violation: {mismatch} head/query rows")
        metrics = {
            "complete": True, "task": args.task, "shots": args.shots, "strategy": strategy,
            "definition": definitions(strategy), "model_path": str(args.model_path),
            "index_scope": "per_head", "budget_ratio": args.budget_ratio,
            "budget_rounding": "max(1, floor(ratio * visible history length)); rational arithmetic",
            "recent_window": 0, "force_current_token": False,
            "reuse_layer_one_based": args.reuse_layer if strategy == "layer5_reuse" else None,
            "quest_page_size": args.page_size if strategy.startswith("quest_") else None,
            "quest_reference_commit": "01c1623bf9395009520874e989e29f683203b357" if strategy.startswith("quest_") else None,
            "quest_native_exceptions": strategy == "quest_native",
            "evaluation_mode": "uncached, unpadded, causal decode selection at every input position; all prompt and answer positions use the selected method",
            "requests": len(requests), "tokens": sum(t["tokens"] for t in timings),
            "selected_entries": selected, "candidate_entries": candidates,
            "expected_floor_entries": expected, "actual_budget_ratio": selected / candidates,
            "target_budget_ratio_after_rounding": expected / candidates,
            "rows_with_budget_mismatch": mismatch, "rows_above_budget": above,
            "total_layers": len(attention_modules), "loading_info": loading,
            "cuda_visible_devices": visible, "torch_num_threads": torch.get_num_threads(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "dtype": "float16",
            "load_seconds": load_seconds, "forward_seconds": sum(t["seconds"] for t in timings),
            "wall_seconds": time.perf_counter() - started, "per_request": timings,
            "source_sha256": source_hashes, "input_sha256": input_sha,
        }
        # Write a completion marker only after inference and all budget checks.
        with metrics_path.open("x", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"END {strategy} actual_budget={selected/candidates:.6f} rows_above_budget={above} metrics={metrics_path}", flush=True)


if __name__ == "__main__":
    main()
