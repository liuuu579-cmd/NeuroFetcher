"""Reviewer controls built on the existing neural OPT implementation.

The base module is loaded under a private name: no model file or symlink is
changed. Full and neural execute the original attention forward unchanged.
Controls change only the selection source, retaining top-k UNION recent-20.
"""

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn


BASE_PATH = Path(__file__).with_name("modeling_opt_neu.py")
BASE_NAME = "transformers.models.opt._neural_controls_base"
if BASE_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(BASE_NAME, BASE_PATH)
    base = importlib.util.module_from_spec(spec)
    sys.modules[BASE_NAME] = base
    spec.loader.exec_module(base)
else:
    base = sys.modules[BASE_NAME]

OriginalOPTAttention = base.OPTAttention


class OPTAttention(OriginalOPTAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_strategy = getattr(self.config, "control_strategy", "full")
        self.control_index_scope = getattr(self.config, "control_index_scope", "shared")
        self.control_recent_window = getattr(self.config, "control_recent_window", 20)
        self.control_stats = None

    def reset_control_stats(self):
        self.control_stats = None

    def _record_mask(self, selected):
        """Accumulate actual unmasked entries on device, including all requests."""
        bsz, heads, queries, keys = selected.shape
        if queries != keys:
            raise ValueError("These controls evaluate uncached, full-sequence requests only")
        multiplier = self.num_heads // heads
        counts = torch.stack([
            selected.sum(dtype=torch.int64) * multiplier,
            selected.new_tensor(bsz * self.num_heads * queries * (queries + 1) // 2,
                                dtype=torch.int64),
        ])
        self.control_stats = counts if self.control_stats is None else self.control_stats + counts

    def _create_causal_prefetch_mask(self, *args, **kwargs):
        # Preserve the neural implementation, including threshold ties and
        # the 20-token recency union, and observe the exact mask it used.
        mask = super()._create_causal_prefetch_mask(*args, **kwargs)
        self._record_mask(mask == 0)
        return mask

    def _selection(self, scores, valid):
        """Select using current or previous TRUE scores; never reuse recursively."""
        queries, keys = scores.shape[-2:]
        lengths = torch.arange(1, queries + 1, device=scores.device)
        k = (lengths * self.prefetch_budget_ratio).ceil().long().clamp(min=1)
        if self.control_strategy == "previous_step":
            # Query i uses the preceding query's ranking. Reconcile its old
            # history with today's requested k; query zero bootstraps itself.
            scores = torch.cat([scores[..., :1, :], scores[..., :-1, :]], dim=-2)
            old_lengths = (lengths - 1).clamp(min=1)
            k = torch.minimum(k, old_lengths)
            old_valid = torch.arange(keys, device=scores.device)[None, :] < old_lengths[:, None]
            scores = scores.masked_fill(~old_valid, float("-inf"))
        sorted_scores = scores.sort(dim=-1, descending=True).values
        thresholds = sorted_scores.gather(
            -1, (k - 1).view(1, 1, queries, 1).expand(*scores.shape[:-1], 1)
        )
        selected = (scores >= thresholds) & valid
        positions = torch.arange(keys, device=scores.device)[None, :]
        recency = (positions <= lengths[:, None] - 1) & (
            positions >= lengths[:, None] - self.control_recent_window
        )
        return selected | (recency & valid)

    def control_mask(self, attn_weights, attention_mask=None):
        bsz, heads, queries, keys = attn_weights.shape
        if queries != keys:
            raise ValueError("Cached decoding is not supported by this evaluation control")
        causal = torch.ones((queries, keys), dtype=torch.bool,
                            device=attn_weights.device).tril()
        valid = causal.view(1, 1, queries, keys).expand(bsz, 1, -1, -1)
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape != (bsz, keys):
                raise ValueError("Expected the neural decoder's two-dimensional padding mask")
            if not bool(attention_mask.bool().all()):
                raise ValueError("Use unpadded requests to keep the neural history budget definition")
        scores = attn_weights.float().masked_fill(~valid, float("-inf"))
        if self.control_index_scope == "shared":
            # Neural selects one key set per layer, shared by all heads.
            # Rank true attention mass averaged across heads, not raw logits.
            scores = scores.softmax(dim=-1).mean(dim=1, keepdim=True)
            scores = scores.masked_fill(~valid, float("-inf"))
        selected = self._selection(scores, valid)
        self._record_mask(selected)
        return torch.zeros_like(selected, dtype=attn_weights.dtype).masked_fill(
            ~selected, torch.finfo(attn_weights.dtype).min
        )

    def forward(self, hidden_states, key_value_states=None, past_key_value=None,
                attention_mask=None, layer_head_mask=None, output_attentions=False,
                predictor_logits=None):
        if self.control_strategy in {"full", "neural"}:
            return super().forward(
                hidden_states, key_value_states, past_key_value, attention_mask,
                layer_head_mask, output_attentions, predictor_logits
            )
        if key_value_states is not None or past_key_value is not None:
            raise ValueError("Controls support uncached self-attention evaluation only")
        if layer_head_mask is not None:
            raise ValueError("The neural reference does not apply layer_head_mask")

        # Same projections, scaling, softmax, dropout and output projection as
        # modeling_opt_neu.py: replace only its mask selection source.
        bsz, queries, _ = hidden_states.size()
        query = self.q_proj(hidden_states) * self.scaling
        key = self._shape(self.k_proj(hidden_states), -1, bsz)
        value = self._shape(self.v_proj(hidden_states), -1, bsz)
        cache = (key, value) if self.is_decoder else None
        shape = (bsz * self.num_heads, -1, self.head_dim)
        query = self._shape(query, queries, bsz).view(*shape)
        key = key.view(*shape)
        value = value.view(*shape)
        weights = torch.bmm(query, key.transpose(1, 2)).view(
            bsz, self.num_heads, queries, queries
        )
        weights = weights + self.control_mask(weights, attention_mask)
        probs = nn.functional.softmax(weights.view(bsz * self.num_heads, queries, queries), dim=-1)
        returned = probs.view(bsz, self.num_heads, queries, queries) if output_attentions else None
        probs = nn.functional.dropout(probs, p=self.dropout, training=self.training)
        output = torch.bmm(probs, value).view(bsz, self.num_heads, queries, self.head_dim)
        output = output.transpose(1, 2).reshape(bsz, queries, self.embed_dim)
        return self.out_proj(output), returned, cache


# OPTDecoderLayer resolves OPTAttention in its own module at construction.
# This private module patch leaves the public transformers module untouched.
base.OPTAttention = OPTAttention
OPTForCausalLM = base.OPTForCausalLM
