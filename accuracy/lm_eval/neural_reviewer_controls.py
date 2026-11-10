#!/usr/bin/env python3
"""Run neural-matched controls without modifying model files or symlinks."""

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
MODEL_SOURCE = HERE.parent / "src/modeling_opt_neural_controls.py"
sys.path.insert(0, str(HERE.parent / "transformers/src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def load_model_module():
    name = "transformers.models.opt.modeling_opt_neural_controls"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, MODEL_SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="winogrande")
    p.add_argument("--shots", type=int, default=5)
    p.add_argument("--strategy", choices=["full", "neural", "oracle", "previous_step"], default="oracle")
    p.add_argument("--index-scope", choices=["shared", "per_head"], default="shared")
    p.add_argument("--budget-ratio", type=float, default=.30)
    p.add_argument("--recent-window", type=int, default=20)
    p.add_argument("--model-path", type=Path, default=Path("/home/liushenyu/opt-13b"))
    p.add_argument("--predictor-path", type=Path,
                   default=Path("/home/liushenyu/test/predictor_topk_0.3_iwl_epoch_20.pth"))
    p.add_argument("--predictor-embedding-mode", choices=["original", "restore"], default="restore",
                   help="restore reloads predictor embeddings AFTER from_pretrained; original is diagnostic only")
    p.add_argument("--predictor-embedding-path", type=Path,
                   default=Path("/home/liushenyu/test/opt_13b_embed_weights.pth"))
    p.add_argument("--input-path", type=Path)
    p.add_argument("--output-dir", type=Path, default=HERE / "results/neural_controls_v2")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def result_record(request, input_ids, logits, tokenizer):
    # Same arithmetic and JSON schema as run_lm_eval_harness.py. Batch the
    # device-to-host transfers to avoid one CUDA synchronization per token.
    logprobs = logits.log_softmax(dim=-1)
    values, indices = logprobs.squeeze(0).topk(dim=-1, k=1)
    gold = torch.gather(logprobs[:, :-1], -1, input_ids[:, 1:].unsqueeze(-1))
    token_ids = input_ids.squeeze(0).cpu().tolist()
    best_ids = indices[:-1, 0].cpu().tolist()
    best_values = values[:-1, 0].cpu().tolist()
    return {"request": request, "result": {
        "choices": [{"text": request["prompt"], "logprobs": {
            "tokens": tokenizer.convert_ids_to_tokens(token_ids),
            "token_logprobs": [None] + gold.reshape(-1).cpu().tolist(),
            "top_logprobs": [None] + [{tokenizer.convert_ids_to_tokens(i): v}
                                       for i, v in zip(best_ids, best_values)],
            "text_offset": [],
        }, "finish_reason": "length"}],
        "request_time": {"batch_time": 0, "batch_size": 1},
    }}


def main():
    args = parse_args()
    if not 0 < args.budget_ratio <= 1 or args.limit < 0 or args.recent_window < 0:
        raise ValueError("Invalid budget, limit, or recent window")
    if args.strategy == "neural" and (args.recent_window != 20 or args.index_scope != "shared"):
        raise ValueError("The unmodified neural reference uses shared indices and recent_window=20")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to load OPT-13B onto CPU")
    torch.manual_seed(0)
    module = load_model_module()
    from transformers import AutoTokenizer
    cls = module.OPTForCausalLM
    config = cls.config_class.from_pretrained(str(args.model_path), local_files_only=True)
    config.control_strategy = args.strategy
    config.control_index_scope = args.index_scope
    config.control_recent_window = args.recent_window
    config.prefetch_budget_ratio = args.budget_ratio
    if args.strategy == "neural":
        config.prefetch_strategy = "neural"
        config.num_predict_layers = 5
        config.prefetch_layers_list = [0, 1, 2, 3, 4]
        if not args.predictor_path.is_file():
            raise FileNotFoundError(args.predictor_path)
    else:
        config.prefetch_strategy = "control"

    tag = f"{args.budget_ratio:g}".replace(".", "p")
    stem = f"{args.task}-{args.shots}-{args.model_path.name}-{args.strategy}-{args.index_scope}-b{tag}-r{args.recent_window}"
    if args.limit:
        stem += f"-limit{args.limit}"
    if args.strategy == "neural" and args.predictor_embedding_mode == "restore":
        stem += "-embedding-restored"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / (stem + ".jsonl")
    metrics_path = args.output_dir / (stem + ".metrics.json")
    if output_path.exists() or metrics_path.exists():
        raise FileExistsError(f"Refusing to overwrite results: {stem}")
    input_path = args.input_path or HERE / f"results/{args.task}-{args.shots}.jsonl"
    requests = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    if args.limit:
        requests = requests[:args.limit]
    if not requests:
        raise ValueError("No input requests")
    print(f"model={args.model_path} strategy={args.strategy} scope={args.index_scope} "
          f"budget={args.budget_ratio} recency_union={args.recent_window} requests={len(requests)}", flush=True)
    started = time.perf_counter()
    # One visible 48 GB GPU comfortably holds standard OPT-13B. This matches
    # the original neural runner's device placement; never ignore bad shapes.
    model, loading = cls.from_pretrained(
        str(args.model_path), config=config, torch_dtype=torch.float16,
        device_map={"": 0}, local_files_only=True, output_loading_info=True,
    )
    allowed_missing = ("predictor.", "predictor_embed_13b.") if args.strategy == "neural" else ()
    bad_missing = [k for k in loading.get("missing_keys", []) if not k.startswith(allowed_missing)]
    if bad_missing or loading.get("mismatched_keys") or loading.get("unexpected_keys") or loading.get("error_msgs"):
        raise RuntimeError(f"Checkpoint is not a strict match: {loading}")
    embedding_diagnostic = None
    if args.strategy == "neural":
        state = torch.load(args.predictor_path, map_location="cpu")
        model.predictor.load_state_dict(state, strict=True)
        del state
        embedding = torch.load(args.predictor_embedding_path, map_location="cpu")
        observed = model.predictor_embed_13b.weight
        if observed.shape != embedding.shape:
            raise ValueError("Predictor embedding file shape does not match the model")
        # Check all entries. Keep the original behavior as a separately named
        # condition rather than silently changing the neural reference.
        expected = embedding.to(device=observed.device, dtype=observed.dtype)
        matched_before = torch.equal(observed, expected)
        if args.predictor_embedding_mode == "restore":
            with torch.no_grad():
                observed.copy_(expected)
        matched_after = torch.equal(observed, expected)
        embedding_diagnostic = {
            "mode": args.predictor_embedding_mode,
            "path": str(args.predictor_embedding_path),
            "matches_before_reload": matched_before,
            "matches_after_reload": matched_after,
        }
        print(f"predictor_embedding={json.dumps(embedding_diagnostic)}", flush=True)
        del expected, embedding
        model.predictor.to(model.device, dtype=torch.float32)
        model.predictor.eval()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), use_fast=False, local_files_only=True)
    attentions = [layer.self_attn for layer in model.model.decoder.layers]
    for attn in attentions:
        attn.reset_control_stats()
    load_seconds = time.perf_counter() - started
    per_request = []
    run_start = time.perf_counter()
    # Stream each finished request so file growth is useful for monitoring.
    with output_path.open("x", encoding="utf-8") as target, torch.inference_mode():
        for number, request in enumerate(requests, start=1):
            ids = tokenizer(request["prompt"], add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
            if ids.shape[1] > config.max_position_embeddings:
                raise ValueError("Input exceeds the model's position limit; no implicit truncation")
            torch.cuda.synchronize()
            begin = time.perf_counter()
            out = model(ids, use_cache=False)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            target.write(json.dumps(result_record(request, ids, out.logits, tokenizer), ensure_ascii=False) + "\n")
            per_request.append({"tokens": int(ids.shape[1]), "seconds": seconds})
            del out
            if number == 1 or number % 100 == 0 or number == len(requests):
                target.flush()
                print(f"processed={number}/{len(requests)} wall_seconds={time.perf_counter()-run_start:.1f}", flush=True)
    counts = [a.control_stats.cpu().tolist() for a in attentions if a.control_stats is not None]
    selected = sum(x[0] for x in counts)
    candidates = sum(x[1] for x in counts)
    sources = [MODEL_SOURCE, module.BASE_PATH, Path(__file__).resolve()]
    metrics = {
        "task": args.task, "shots": args.shots, "strategy": args.strategy,
        "model_path": str(args.model_path), "predictor_path": str(args.predictor_path) if args.strategy == "neural" else None,
        "index_scope": args.index_scope, "budget_ratio": args.budget_ratio,
        "recent_window": args.recent_window, "budget_definition": "ceil(history_length * ratio) UNION recent_window; threshold ties retained",
        "previous_step_definition": "previous query TRUE ranking, reselected at current k within previous history; not recursive reuse" if args.strategy == "previous_step" else None,
        "requests": len(requests), "tokens": sum(r["tokens"] for r in per_request),
        "forward_seconds": sum(r["seconds"] for r in per_request),
        "wall_seconds": time.perf_counter() - run_start, "load_seconds": load_seconds,
        "selected_entries": selected, "candidate_entries": candidates,
        "actual_budget_ratio": selected / candidates if candidates else 1.0,
        "layers_with_sparse_masks": len(counts), "total_layers": len(attentions),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_num_threads": torch.get_num_threads(), "loading_info": loading,
        "predictor_embedding": embedding_diagnostic,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "per_request": per_request,
    }
    with metrics_path.open("x", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k:v for k,v in metrics.items() if k not in {"per_request", "loading_info", "source_sha256"}}, indent=2), flush=True)
    print(f"results={output_path}\nmetrics={metrics_path}", flush=True)


if __name__ == "__main__":
    main()
