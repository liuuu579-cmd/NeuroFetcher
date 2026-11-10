"""Additive OPT quality controls with explicit token/page budget definitions.

Quest selection reference: mit-han-lab/Quest, commit
01c1623bf9395009520874e989e29f683203b357, evaluation/quest_attention.py.
This is a PyTorch OPT accuracy adaptation, not the Quest CUDA runtime.
"""

import importlib.util
from fractions import Fraction
from pathlib import Path
import sys

import torch
from torch import nn


BASE_PATH = Path(__file__).with_name("modeling_opt_neu.py")
BASE_NAME = "transformers.models.opt._strict_controls_base"
if BASE_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(BASE_NAME, BASE_PATH)
    base = importlib.util.module_from_spec(spec)
    sys.modules[BASE_NAME] = base
    spec.loader.exec_module(base)
else:
    base = sys.modules[BASE_NAME]
OriginalOPTAttention = base.OPTAttention


def budgets(length, ratio, device):
    """Exact rational floor, with one compulsory entry for nonempty attention."""
    fraction = Fraction(str(ratio))
    lengths = torch.arange(1, length + 1, device=device)
    return (lengths * fraction.numerator // fraction.denominator).clamp(min=1)


def exact_topk(scores, k, valid):
    """Exact cardinality, deterministic lower-position tie break; no recency."""
    order = scores.masked_fill(~valid, float("-inf")).argsort(
        dim=-1, descending=True, stable=True
    )
    take = torch.arange(scores.shape[-1], device=scores.device) < k[..., None]
    selected = torch.zeros_like(scores, dtype=torch.bool)
    return selected.scatter(-1, order, take.expand_as(order)) & valid


def quest_page_scores(query, key, page_size):
    """Per-head sign-aware page bounds with no future-key metadata leakage.

    Complete older pages use their min/max keys. The current partial page
    uses prefix extrema, as it would in single-token cached decoding.
    Returns [batch, heads, queries, pages] float32 bounds.
    """
    batch, heads, length, dim = key.shape
    if query.shape != key.shape:
        raise ValueError("Expected uncached full-sequence Q/K with matching shapes")
    pad = (-length) % page_size
    pages = (length + pad) // page_size
    extended = nn.functional.pad(key, (0, 0, 0, pad)).view(
        batch, heads, pages, page_size, dim
    )
    # The last page is replaced with its prefix bound for each visible query.
    # Therefore the padded values cannot influence a visible page's score.
    maximum = extended.amax(dim=-2).float()
    minimum = extended.amin(dim=-2).float()
    q = query.float()
    scores = q.clamp(min=0) @ maximum.transpose(-1, -2)
    scores += q.clamp(max=0) @ minimum.transpose(-1, -2)
    prefix_max = extended.cummax(dim=-2).values.reshape(batch, heads, -1, dim)[:, :, :length]
    prefix_min = extended.cummin(dim=-2).values.reshape(batch, heads, -1, dim)[:, :, :length]
    current_bound = (q.clamp(min=0) * prefix_max.float()
                     + q.clamp(max=0) * prefix_min.float()).sum(-1)
    current_page = torch.arange(length, device=q.device) // page_size
    scores.scatter_(-1, current_page.view(1, 1, length, 1).expand(batch, heads, -1, 1),
                    current_bound.unsqueeze(-1))
    return scores


def quest_selection(query, key, k, page_size, native):
    scores = quest_page_scores(query, key, page_size)
    length, pages = scores.shape[-2:]
    positions = torch.arange(length, device=query.device)
    current_page = positions // page_size
    valid_pages = torch.arange(pages, device=query.device)[None, :] <= current_page[:, None]
    causal = positions[None, :] <= positions[:, None]
    if native:
        # Faithful to official accuracy selector: >=3 pages, floor token/page.
        page_k = torch.maximum(k // page_size, k.new_tensor(3))
        page_k = torch.minimum(page_k, current_page + 1)
        selected_pages = exact_topk(scores, page_k, valid_pages)
        return selected_pages.repeat_interleave(page_size, dim=-1)[..., :length] & causal
    # Explicit budget-capped adaptation: page-bound order, partially consume
    # the final selected page if necessary. No first-two-layer Full exception.
    token_scores = scores.repeat_interleave(page_size, dim=-1)[..., :length]
    return exact_topk(token_scores, k, causal)


class OPTAttention(OriginalOPTAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strict_strategy = getattr(self.config, "strict_strategy", "full")
        self.strict_ratio = getattr(self.config, "strict_ratio", .3)
        self.reuse_layer = getattr(self.config, "strict_reuse_layer", 4)
        self.page_size = getattr(self.config, "quest_page_size", 16)
        self.control_state = None
        self.control_stats = None

    def _record(self, selected, k):
        counts = selected.sum(-1, dtype=torch.int64)
        target = k.expand_as(counts)
        length = selected.shape[-1]
        candidates = selected.shape[0] * self.num_heads * length * (length + 1) // 2
        values = torch.stack((counts.sum(), counts.new_tensor(candidates), target.sum(),
                              (counts != target).sum(), (counts > target).sum()))
        self.control_stats = values if self.control_stats is None else self.control_stats + values

    def selection(self, weights, query, key):
        length = weights.shape[-1]
        positions = torch.arange(length, device=weights.device)
        valid = positions[None, :] <= positions[:, None]
        k = budgets(length, self.strict_ratio, weights.device)
        if self.strict_strategy == "layer5_reuse" and self.layer_idx > self.reuse_layer:
            selected = self.control_state.get("shallow_mask")
            if selected is None or selected.shape != weights.shape:
                raise RuntimeError("Missing matching layer-5 selection in this request")
        elif self.strict_strategy == "previous_step":
            previous = torch.cat((weights[..., :1, :], weights[..., :-1, :]), dim=-2)
            previous_length = positions.clamp(min=1)
            previous_valid = positions[None, :] < previous_length[:, None]
            selected = exact_topk(previous.float(), torch.minimum(k, previous_length), previous_valid)
        elif self.strict_strategy.startswith("quest_"):
            if self.strict_strategy == "quest_native" and self.layer_idx < 2:
                selected = valid.view(1, 1, length, length).expand_as(weights)
            else:
                selected = quest_selection(query, key, k, self.page_size,
                                           native=self.strict_strategy == "quest_native")
        else:
            selected = exact_topk(weights.float(), k, valid)
        if self.strict_strategy == "layer5_reuse" and self.layer_idx == self.reuse_layer:
            self.control_state["shallow_mask"] = selected
        self._record(selected, k)
        return selected

    def forward(self, hidden_states, key_value_states=None, past_key_value=None,
                attention_mask=None, layer_head_mask=None, output_attentions=False,
                predictor_logits=None):
        if self.strict_strategy == "full":
            return super().forward(hidden_states, key_value_states, past_key_value,
                                   attention_mask, layer_head_mask, output_attentions, predictor_logits)
        if key_value_states is not None or past_key_value is not None:
            raise ValueError("Only uncached self-attention quality evaluation is supported")
        if predictor_logits is not None or layer_head_mask is not None:
            raise ValueError("Controls do not use predictor logits or head masking")
        if attention_mask is not None and not bool(attention_mask.bool().all()):
            raise ValueError("Only unpadded requests are supported")
        if self.control_state is None:
            raise RuntimeError("Attach a shared control_state before running the model")
        if self.layer_idx == 0:
            self.control_state.clear()
        batch, length, _ = hidden_states.shape
        query = self._shape(self.q_proj(hidden_states) * self.scaling, length, batch)
        key = self._shape(self.k_proj(hidden_states), length, batch)
        value = self._shape(self.v_proj(hidden_states), length, batch)
        weights = query @ key.transpose(-1, -2)
        selected = self.selection(weights, query, key)
        weights = weights.masked_fill(~selected, torch.finfo(weights.dtype).min)
        # Original neural reference uses fp16 softmax; Quest reference upcasts.
        probs = (weights.float().softmax(-1).to(weights.dtype)
                 if self.strict_strategy.startswith("quest_") else weights.softmax(-1))
        returned = probs if output_attentions else None
        output = nn.functional.dropout(probs, p=self.dropout, training=self.training) @ value
        output = output.transpose(1, 2).reshape(batch, length, self.embed_dim)
        return self.out_proj(output), returned, (key, value) if self.is_decoder else None


base.OPTAttention = OPTAttention
OPTForCausalLM = base.OPTForCausalLM
