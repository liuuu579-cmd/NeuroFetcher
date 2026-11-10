#!/usr/bin/env python3
"""OPT-13B throughput runner for Quest's native page-selection rule.

Prefill uses full causal attention. Cached decode selects whole pages using
Quest's sign-aware min/max bounds, page size 16 and minimum 3 pages. The first
two OPT layers remain full. This is a Transformers runtime adaptation; KV is
resident on the GPU, so it is not the official Quest paging system.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSFORMERS_SRC = ROOT / "accuracy/transformers/src"
ACCURACY_SRC = ROOT / "accuracy/src"
CONTROLS_SOURCE = ROOT / "accuracy/src/modeling_opt_strict_controls.py"
MODEL_PATH = Path("/home/liushenyu/opt-13b")
PROMPT_PATH = ROOT / "speedup/flexgen/pg19_firstbook.txt"


def load_controls():
    sys.path.insert(0, str(TRANSFORMERS_SRC))
    sys.path.insert(0, str(ACCURACY_SRC))
    name = "transformers.models.opt.modeling_opt_quest_cached_v8"
    spec = importlib.util.spec_from_file_location(name, CONTROLS_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_attention(controls, torch, page_size, minimum_pages, variant):
    base = controls.base
    original_attention = controls.OriginalOPTAttention

    class QuestCachedAttention(original_attention):
        selected_tokens = 0
        candidate_tokens = 0

        def forward(self, hidden_states, key_value_states=None, past_key_value=None,
                    attention_mask=None, layer_head_mask=None, output_attentions=False,
                    predictor_logits=None):
            if key_value_states is not None or layer_head_mask is not None or predictor_logits is not None:
                raise ValueError("Quest cached runner supports decoder self-attention only")
            batch, target_len, _ = hidden_states.shape
            if target_len != 1 or past_key_value is None:
                return super().forward(hidden_states, key_value_states, past_key_value,
                                       attention_mask, layer_head_mask, output_attentions,
                                       predictor_logits)

            query = self._shape(self.q_proj(hidden_states) * self.scaling, target_len, batch)
            key_new = self._shape(self.k_proj(hidden_states), target_len, batch)
            value_new = self._shape(self.v_proj(hidden_states), target_len, batch)
            key = torch.cat((past_key_value[0], key_new), dim=2)
            value = torch.cat((past_key_value[1], value_new), dim=2)
            src_len = key.shape[2]
            weights = torch.matmul(query, key.transpose(-1, -2))

            if self.layer_idx < 2 and variant == "native":
                selected = torch.ones((batch, self.num_heads, src_len),
                                      dtype=torch.bool, device=weights.device)
            else:
                page_count = (src_len + page_size - 1) // page_size
                pad_len = page_count * page_size - src_len
                padded = torch.nn.functional.pad(key, (0, 0, 0, pad_len))
                pages = padded.view(batch, self.num_heads, page_count, page_size, self.head_dim)
                page_max = pages.amax(dim=3).float()
                page_min = pages.amin(dim=3).float()
                q = query[:, :, 0].float()
                scores = torch.einsum("bhd,bhpd->bhp", q.clamp(min=0), page_max)
                scores += torch.einsum("bhd,bhpd->bhp", q.clamp(max=0), page_min)
                if variant == "native":
                    page_budget = min(page_count, max(minimum_pages,
                                                      int((src_len * 0.30) // page_size)))
                    selected_pages = torch.zeros_like(scores, dtype=torch.bool)
                    chosen = scores.topk(page_budget, dim=-1).indices
                    selected_pages.scatter_(-1, chosen, True)
                    selected = selected_pages.repeat_interleave(page_size, dim=-1)[..., :src_len]
                else:
                    token_scores = scores.repeat_interleave(page_size, dim=-1)[..., :src_len]
                    token_budget = min(src_len, max(1, math.ceil(src_len * 0.30)))
                    chosen = token_scores.argsort(dim=-1, descending=True, stable=True)[..., :token_budget]
                    selected = torch.zeros_like(token_scores, dtype=torch.bool)
                    selected.scatter_(-1, chosen, True)

            self.selected_tokens += int(selected.sum().item())
            self.candidate_tokens += batch * self.num_heads * src_len
            weights = weights.masked_fill(~selected[:, :, None, :], torch.finfo(weights.dtype).min)
            if attention_mask is not None:
                if attention_mask.shape[-2] != target_len:
                    attention_mask = attention_mask[..., -target_len:, :]
                weights = weights + attention_mask
            probs = torch.softmax(weights.float(), dim=-1).to(value.dtype)
            context = torch.matmul(probs, value)
            context = context.transpose(1, 2).reshape(batch, target_len, self.embed_dim)
            return self.out_proj(context), probs if output_attentions else None, (key, value)

    base.OPTAttention = QuestCachedAttention
    return QuestCachedAttention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--prompt-len", type=int, required=True)
    parser.add_argument("--gen-len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--minimum-pages", type=int, default=3)
    parser.add_argument("--variant", choices=("native", "capped"), default="native")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise RuntimeError("Expose exactly the selected GPU with CUDA_VISIBLE_DEVICES")
    if not 1 <= args.threads <= 10 or len(os.sched_getaffinity(0)) > 10:
        raise RuntimeError("Each experiment is limited to ten CPU cores")
    if args.prompt_len + args.gen_len > 2048 or args.prompt_len < 16:
        raise ValueError("OPT-13B context length must be within 16..2048")
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261027)
    torch.cuda.manual_seed_all(20261027)
    controls = load_controls()
    attention_cls = build_attention(controls, torch, args.page_size, args.minimum_pages,
                                    args.variant)
    sys.path.insert(0, str(ROOT / "speedup/flexgen"))
    from flexgen.opt_config import disable_hf_opt_init, disable_torch_init
    disable_hf_opt_init()
    disable_torch_init()
    config = controls.OPTForCausalLM.config_class.from_pretrained(
        str(MODEL_PATH), local_files_only=True)
    config.strict_strategy = "quest_cached_v8"
    model = controls.OPTForCausalLM.from_pretrained(
        str(MODEL_PATH), config=config, torch_dtype=torch.float16,
        local_files_only=True).to("cuda:0").eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH), use_fast=False, local_files_only=True)
    prompt = PROMPT_PATH.read_text()
    encoded = tokenizer(prompt, truncation=True, max_length=args.prompt_len,
                        padding="max_length", return_tensors="pt")
    input_ids = encoded.input_ids.to("cuda:0").expand(args.batch, -1).contiguous()
    attention_mask = encoded.attention_mask.to("cuda:0").expand(args.batch, -1).contiguous()

    def generate_once():
        for layer in model.model.decoder.layers:
            layer.self_attn.selected_tokens = 0
            layer.self_attn.candidate_tokens = 0
        with torch.inference_mode():
            first = model(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=True, return_dict=True)
            past = first.past_key_values
            next_ids = first.logits[:, -1].argmax(dim=-1, keepdim=True)
            full_mask = attention_mask
            for _ in range(args.gen_len - 1):
                full_mask = torch.cat((full_mask, full_mask.new_ones((args.batch, 1))), dim=1)
                out = model(input_ids=next_ids, attention_mask=full_mask,
                            past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                next_ids = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        return sum(layer.self_attn.selected_tokens for layer in model.model.decoder.layers), \
            sum(layer.self_attn.candidate_tokens for layer in model.model.decoder.layers)

    generate_once()
    torch.cuda.synchronize()
    for layer in model.model.decoder.layers:
        layer.self_attn.selected_tokens = 0
        layer.self_attn.candidate_tokens = 0
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        prefill_start = time.perf_counter()
        first = model(input_ids=input_ids, attention_mask=attention_mask,
                      use_cache=True, return_dict=True)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - prefill_start
        past = first.past_key_values
        next_ids = first.logits[:, -1].argmax(dim=-1, keepdim=True)
        full_mask = attention_mask
        decode_start = time.perf_counter()
        for _ in range(args.gen_len - 1):
            full_mask = torch.cat((full_mask, full_mask.new_ones((args.batch, 1))), dim=1)
            out = model(input_ids=next_ids, attention_mask=full_mask,
                        past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            next_ids = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - decode_start
    total_s = time.perf_counter() - start
    selected = sum(layer.self_attn.selected_tokens for layer in model.model.decoder.layers)
    candidates = sum(layer.self_attn.candidate_tokens for layer in model.model.decoder.layers)
    result = {
        "complete": True,
        "method": "quest_native" if args.variant == "native" else "quest_capped",
        "runtime": "Transformers OPT cached-decode adaptation; full GPU KV, not official Quest paging runtime",
        "gpu": args.gpu,
        "model": "OPT-13B",
        "precision": "FP16",
        "prompt_len": args.prompt_len,
        "generation_len": args.gen_len,
        "batch_size": args.batch,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "total_s": total_s,
        "decode_tokens_per_s": args.batch * max(args.gen_len - 1, 0) / max(decode_s, 1e-12),
        "end_to_end_tokens_per_s": args.batch * args.gen_len / max(total_s, 1e-12),
        "selection_density": selected / max(candidates, 1),
        "parameters": {"page_size": args.page_size, "minimum_pages": args.minimum_pages,
                       "variant": args.variant,
                       "token_budget_ratio": 0.30 if args.variant == "capped" else None,
                       "first_two_layers_full": args.variant == "native", "recent_union": False},
        "note": ("Prefill is full causal. Decode applies Quest page bounds independently per head; "
                 "capped uses a strict ceil(0.30 * visible_length) token budget."
                 if args.variant == "capped" else
                 "Prefill is full causal. Decode applies Quest page bounds independently per head."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
