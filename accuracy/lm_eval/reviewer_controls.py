#!/usr/bin/env python3
"""Run matched-budget reviewer controls without changing the existing runner.

The controls are evaluated on the same prompt JSONL files used by the local
LM-eval harness.  ``previous_step`` reuses the preceding query's selected KV
indices; ``oracle`` selects the current query's true attention top-k and is an
upper-bound control, not an implementable prefetcher.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ACCURACY = HERE.parent
TRANSFORMERS_SRC = ACCURACY / "transformers" / "src"
REVIEWER_MODEL = ACCURACY / "src" / "modeling_opt_reviewer.py"
if str(TRANSFORMERS_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSFORMERS_SRC))


def load_reviewer_model_class():
    module_name = "transformers.models.opt.modeling_opt_reviewer"
    spec = importlib.util.spec_from_file_location(module_name, REVIEWER_MODEL)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reviewer model from {REVIEWER_MODEL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.OPTForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="openbookqa")
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--model-path", type=Path,
                        default=Path("/home/liushenyu/opt-13b"))
    parser.add_argument("--tokenizer-path", type=Path,
                        default=Path("/home/liushenyu/opt-13b"))
    parser.add_argument("--strategy", choices=("full", "previous_step", "oracle"),
                        default="previous_step")
    parser.add_argument("--budget-ratio", type=float, default=0.20)
    parser.add_argument("--recent-window", type=int, default=1,
                        help="mandatory recent positions included within the matched total budget")
    parser.add_argument("--limit", type=int, default=0,
                        help="evaluate only the first N requests; 0 means all")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, or a CUDA device such as cuda:0")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_paths(args):
    args.input_path = args.input_path or (HERE / "results" / f"{args.task}-{args.shots}.jsonl")
    budget_tag = f"b{args.budget_ratio:g}".replace(".", "p")
    suffix = "" if args.strategy == "full" else f"-{budget_tag}"
    base = f"{args.task}-{args.shots}-opt-13b-{args.strategy}{suffix}"
    args.output_path = args.output_path or (HERE / "results" / f"{base}.jsonl")
    args.metrics_path = args.metrics_path or (HERE / "results" / f"{base}.metrics.json")
    return args


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_model(args, device):
    model_cls = load_reviewer_model_class()
    from transformers import AutoTokenizer

    dtype = torch.float16 if args.dtype == "float16" and device.type == "cuda" else torch.float32
    config = model_cls.config_class.from_pretrained(str(args.model_path))
    config.reviewer_strategy = None if args.strategy == "full" else args.strategy
    config.reviewer_budget_ratio = args.budget_ratio
    config.reviewer_recent_window = max(1, args.recent_window)

    load_kwargs = {
        "config": config,
        "torch_dtype": dtype,
        "local_files_only": True,
    }
    if device.type == "cuda":
        # The shell wrapper exposes GPUs 0-3.  Accelerate shards the model
        # across those visible devices when available.
        load_kwargs["device_map"] = "auto"
        load_kwargs["low_cpu_mem_usage"] = True
    model = model_cls.from_pretrained(str(args.model_path), **load_kwargs)
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), use_fast=False,
                                              local_files_only=True)
    tokenizer.model_max_length = args.max_length
    return model, tokenizer, dtype


def reviewer_attentions(model):
    layers = getattr(getattr(model, "model", None), "decoder", None)
    return [] if layers is None else [layer.self_attn for layer in layers.layers]


def reset_reviewer_state(model):
    for attention in reviewer_attentions(model):
        attention.reviewer_previous_indices = None
        attention.reviewer_selected_indices = None
        attention.reviewer_iok = []
        attention.reviewer_selected_count = 0
        attention.reviewer_candidate_count = 0


def collect_metrics(model):
    attentions = reviewer_attentions(model)
    selected = sum(x.reviewer_selected_count for x in attentions)
    candidates = sum(x.reviewer_candidate_count for x in attentions)
    iok = []
    for attention in attentions:
        for value in attention.reviewer_iok:
            if isinstance(value, list):
                iok.extend(value)
            else:
                iok.append(value)
    return {
        "layers": len(attentions),
        "selected_entries": selected,
        "candidate_entries": candidates,
        "actual_budget_ratio": selected / candidates if candidates else None,
        "mean_previous_step_iok": sum(iok) / len(iok) if iok else None,
    }


def make_result(request, prompt, tokenizer, logits, input_ids):
    logprobs = logits.log_softmax(dim=-1)
    values, indices = logprobs.squeeze(0).topk(dim=-1, k=1)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    gold_indices = input_ids[:, 1:]
    token_logprobs = [None] + torch.gather(
        logprobs[:, :-1, :], -1, gold_indices.unsqueeze(-1)
    ).squeeze(-1).squeeze(0).cpu().tolist()
    top_logprobs = [None] + [
        {tokenizer.convert_ids_to_tokens(i.item()): v.item()}
        for v, i in zip(values.squeeze(-1)[:-1], indices.squeeze(-1)[:-1])
    ]
    return {
        "request": request,
        "result": {"choices": [{
            "text": prompt,
            "logprobs": {
                "tokens": tokens,
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
                "text_offset": [],
            },
            "finish_reason": "length",
        }], "request_time": {"batch_time": 0, "batch_size": 1}},
    }


def main():
    args = resolve_paths(parse_args())
    torch.manual_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = choose_device(args.device)
    if not args.input_path.exists():
        raise FileNotFoundError(f"input JSONL does not exist: {args.input_path}")
    if args.strategy != "full" and not (0 < args.budget_ratio <= 1):
        raise ValueError("--budget-ratio must be in (0, 1]")
    if args.recent_window < 1:
        raise ValueError("--recent-window must be at least 1 to retain the current token")

    requests = []
    with args.input_path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                requests.append(json.loads(line))
    if args.limit > 0:
        requests = requests[:args.limit]
    if not requests:
        raise ValueError(f"input JSONL is empty: {args.input_path}")

    print(f"strategy={args.strategy} budget_ratio={args.budget_ratio} device={device} model={args.model_path}")
    model, tokenizer, dtype = configure_model(args, device)
    results = []
    total_tokens = 0
    total_elapsed = 0.0
    per_request = []
    for number, request in enumerate(requests, start=1):
        prompt = request["prompt"]
        encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        # Match EvalHarnessAdaptor.process_request, which retains the suffix
        # when a prompt exceeds the fixed sequence length.
        input_ids = encoded.input_ids[:, -args.max_length:]
        if device.type == "cuda" and hasattr(model, "device"):
            input_ids = input_ids.to(model.device)
        else:
            input_ids = input_ids.to(device)
        reset_reviewer_state(model)
        start = time.perf_counter()
        with torch.inference_mode():
            output = model(input_ids, use_cache=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        results.append(make_result(request, prompt, tokenizer, output.logits, input_ids))
        total_tokens += int(input_ids.shape[1])
        total_elapsed += elapsed
        per_request.append({"tokens": int(input_ids.shape[1]), "seconds": elapsed})
        if number == 1 or number % 25 == 0 or number == len(requests):
            print(f"processed={number}/{len(requests)} tokens={input_ids.shape[1]} seconds={elapsed:.3f}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as target:
        for result in results:
            target.write(json.dumps(result, ensure_ascii=False) + "\n")

    metrics = {
        "strategy": args.strategy,
        "model_path": str(args.model_path),
        "task": args.task,
        "shots": args.shots,
        "budget_ratio": args.budget_ratio,
        "recent_window": args.recent_window,
        "dtype": str(dtype),
        "device": str(device),
        "requests": len(requests),
        "tokens": total_tokens,
        "seconds": total_elapsed,
        "tokens_per_second": total_tokens / total_elapsed if total_elapsed else None,
        "per_request": per_request,
        "control": collect_metrics(model),
    }
    with args.metrics_path.open("w", encoding="utf-8") as target:
        json.dump(metrics, target, indent=2)
    print(json.dumps(metrics["control"], indent=2))
    print(f"results={args.output_path}")
    print(f"metrics={args.metrics_path}")


if __name__ == "__main__":
    main()
