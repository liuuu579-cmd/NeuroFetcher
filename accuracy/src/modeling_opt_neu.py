# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch OPT model."""
import math
import torch.nn.functional as F
from predictor_model2A import PredictorTransformer

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_opt import OPTConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/opt-350m"
_CONFIG_FOR_DOC = "OPTConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ArthurZ/opt-350m-dummy-sc"
_SEQ_CLASS_EXPECTED_LOSS = 1.71
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_0'"

OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # See all OPT models at https://huggingface.co/models?filter=opt
]


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        # <<< MODIFIED: 接收完整的 config 对象 >>>
        config: OPTConfig = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # <<< MODIFIED: 从 config 中获取 alpha 等参数 >>>
        # 将 config 保存起来，以便 forward 方法中使用
        self.config = config 
        # 从 config 中获取 alpha，如果未定义则使用一个默认值 (例如 5.0)
        self.alpha = getattr(config, "alpha", 5.0)
        self.prefetch_budget_ratio = getattr(config, "prefetch_budget_ratio", 0.3)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.layer_idx = layer_idx

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    def _create_causal_sparsified_mask(
        self,
        prefetch_probs: torch.Tensor,
        bsz: int,
        tgt_len: int,
        src_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Creates a sparse causal attention mask guided by neural network predictions.
        """
        # --- 设定预算 ---
        # 这是一个关键超参数，可以放入config中以便调整
        HISTORICAL_BUDGET = getattr(self.config, "historical_budget", 32)
        
        # 1. 初始化一个全为-inf的掩码
        final_attention_mask = torch.full(
            (bsz, self.num_heads, tgt_len, src_len),
            torch.finfo(dtype).min,
            device=device
        )
        
        # 2. 遍历每个Query Token，为其打开关注权限
        for i in range(tgt_len):
            # a. 确定当前Query `i` 的历史范围 (0 到 i)
            # prefetch_probs shape: [bsz, src_len]
            history_range_probs = prefetch_probs[:, :i + 1]
            
            # b. 在历史范围内，根据神经网络的预测概率，选出最重要的k个位置
            num_to_keep = min(i + 1, HISTORICAL_BUDGET)
            
            if num_to_keep > 0:
                # topk返回最高的概率值和它们对应的索引
                # top_history_indices shape: [bsz, k]
                _, top_history_indices = torch.topk(history_range_probs, k=num_to_keep, dim=-1)
                
                # c. 在最终掩码中，为Query `i` 打开对这些选定历史位置的关注权限
                # 我们需要将索引广播以匹配掩码的形状
                indices_to_unmask = top_history_indices.unsqueeze(1).unsqueeze(1)
                
                final_attention_mask[:, :, i:i + 1, :].scatter_(
                    dim=-1,
                    index=indices_to_unmask.expand(-1, self.num_heads, -1, -1),
                    value=0.0
                )

        return final_attention_mask
    """
    def _create_causal_prefetch_mask(
        self,
        predictor_logits: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        bsz: int,
        tgt_len: int,
        src_len: int
    ):
        
        min_val = torch.finfo(dtype).min
    
        # 1. 创建因果掩码 (布尔型)
        # Shape: [tgt_len, src_len]
        causal_mask_bool = torch.tril(torch.ones((tgt_len, src_len), device=device, dtype=torch.bool))
    
        # 2. 将 predictor_logits 广播到目标形状
        # predictor_logits: [bsz, src_len] -> [bsz, 1, src_len]
        logits_expanded = predictor_logits.unsqueeze(1)
    
        # 3. 应用因果掩码：将未来的 logits 设为极小值
        # 结果 shape: [bsz, tgt_len, src_len]
        causally_masked_logits = torch.where(
            causal_mask_bool,
            logits_expanded,
            torch.tensor(min_val, dtype=dtype, device=device)
        )
    
        # 4. <<< NEW LOGIC: 基于 Top-K 比例选择 >>>
        # a. 计算每个 Query 位置 (行) 的 k 值
        # historical_lengths: [1, 2, 3, ..., tgt_len]
        historical_lengths = torch.arange(1, tgt_len + 1, device=device)
        # k_values: 根据比例计算每个位置的 k，至少为 1
        # self.prefetch_budget_ratio 需要在 __init__ 中从 config 获取
        k_values = (historical_lengths * self.prefetch_budget_ratio).ceil().long().clamp(min=1)
    
        # b. 找到每行的 top-k logits 的阈值
        # k_values 需要扩展维度以便 topk 函数使用
        # causally_masked_logits shape: [bsz, tgt_len, src_len]
        # k_values shape: [tgt_len] -> [1, tgt_len, 1] for broadcasting
        k_values_expanded = k_values.view(1, tgt_len, 1).expand(bsz, -1, -1)
        
        # torch.topk returns (values, indices). We only need the k-th value as a threshold.
        # 使用 gather 来处理每个行需要不同 k 值的情况
        # 首先对整个 causally_masked_logits 排序
        sorted_logits, _ = torch.sort(causally_masked_logits, dim=-1, descending=True)
        
        # k_indices 是 k_values-1，因为是0-indexed
        k_indices = (k_values_expanded - 1).clamp(min=0)
        # 使用 gather 获取每行的第 k 大的值
        thresholds = torch.gather(sorted_logits, dim=-1, index=k_indices)
    
        # 5. 生成最终的布尔掩码
        # 一个logit必须大于等于它所在行的第k大的logit才会被选中
        prefetch_mask_bool = (causally_masked_logits >= thresholds)
    
        # 6. 将布尔掩码转换为 additive mask for attention
        final_mask = torch.zeros((bsz, 1, tgt_len, src_len), device=device, dtype=dtype)
        final_mask.masked_fill_(~prefetch_mask_bool.unsqueeze(1), min_val)
    
        return final_mask
    """
    def _create_causal_prefetch_mask(
        self,
        predictor_logits: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        bsz: int,
        tgt_len: int,
        src_len: int
    ):
        
        min_val = torch.finfo(dtype).min
    
        # =================================================================
        # 步骤 1: 计算基础的 Top-K 预取掩码 (这一部分本身就是因果的)
        # =================================================================
        
        # 1a. 创建一个基础的因果掩码，后面会多次用到它
        causal_mask_bool = torch.tril(torch.ones((tgt_len, src_len), device=device, dtype=torch.bool))
    
        # 1b. 广播 logits 并应用因果掩码
        logits_expanded = predictor_logits.unsqueeze(1)
        causally_masked_logits = torch.where(
            causal_mask_bool,
            logits_expanded,
            torch.tensor(min_val, dtype=dtype, device=device)
        )
    
        # 1c. 基于 Top-K 比例计算阈值
        historical_lengths = torch.arange(1, tgt_len + 1, device=device)
        k_values = (historical_lengths * self.prefetch_budget_ratio).ceil().long().clamp(min=1)
        k_values_expanded = k_values.view(1, tgt_len, 1).expand(bsz, -1, -1)
        sorted_logits, _ = torch.sort(causally_masked_logits, dim=-1, descending=True)
        k_indices = (k_values_expanded - 1).clamp(min=0)
        thresholds = torch.gather(sorted_logits, dim=-1, index=k_indices)
    
        # 1d. 生成基础的预取掩码 (布尔型)
        prefetch_mask_bool_base = (causally_masked_logits >= thresholds)
    
        # =================================================================
        # 步骤 2: 创建并合并强制保留掩码，同时严格遵守因果关系
        # =================================================================
    
        NUM_SINK_TOKENS = 20
    
        # 2a. 创建强制保留 "前 NUM_SINK_TOKENS" 的掩码
        sink_mask = torch.zeros((1, src_len), dtype=torch.bool, device=device)
        num_to_force_keep_start = min(src_len, NUM_SINK_TOKENS)
        sink_mask[:, :num_to_force_keep_start] = True
    
        # 2b. 创建强制保留 "最近 NUM_SINK_TOKENS" 的掩码 (Causal Recency Mask)
        #    这个实现本身就是因果的，因为它结合了 tril 和 triu
        recency_mask = torch.ones((tgt_len, src_len), device=device, dtype=torch.bool).tril(0).triu(-(NUM_SINK_TOKENS - 1))
    
        # 2c. 合并两个强制保留掩码
        #    sink_mask 会被广播到 (tgt_len, src_len) 的形状与 recency_mask 进行 "或" 操作
        forced_keep_mask_bool = recency_mask #sink_mask | recency_mask
        
        
        causal_forced_keep_mask = forced_keep_mask_bool & causal_mask_bool  
    
        # 2d. 使用逻辑 "或" 将基础预取掩码和因果强制保留掩码合并
        final_mask_bool =  prefetch_mask_bool_base | causal_forced_keep_mask
    
        # =================================================================
        # 步骤 3: 转换为 Additive Attention Mask
        # =================================================================
        
        final_mask = torch.zeros((bsz, 1, tgt_len, src_len), device=device, dtype=dtype)
        final_mask.masked_fill_(~final_mask_bool.unsqueeze(1), min_val)
    
        return final_mask
    # 在 OPTAttention 类中添加这个新方法
    def _create_topk_from_attn_mask(self, attn_weights: torch.Tensor):
        """
        Creates a prefetching mask based on the attention scores themselves and a top-k ratio,
        while preserving causality.
        """
        bsz, num_heads, tgt_len, src_len = attn_weights.shape
        dtype = attn_weights.dtype
        device = attn_weights.device
        min_val = torch.finfo(dtype).min
    
        # 1. 创建因果掩码 (布尔型)
        causal_mask_bool = torch.tril(torch.ones((tgt_len, src_len), device=device, dtype=torch.bool))
        # 扩展到可以和 attn_weights 广播: [1, 1, tgt_len, src_len]
        causal_mask_bool = causal_mask_bool.view(1, 1, tgt_len, src_len)
    
        # 2. 应用因果掩码：将未来的 attention scores 设为极小值
        # 这确保我们只在历史信息中选择 top-k
        causally_masked_attn = torch.where(
            causal_mask_bool,
            attn_weights,
            torch.tensor(min_val, dtype=dtype, device=device)
        )
    
        # 3. 计算每个 Query 位置 (行) 的 k 值
        historical_lengths = torch.arange(1, tgt_len + 1, device=device)
        k_values = (historical_lengths * self.prefetch_budget_ratio).ceil().long().clamp(min=1)
        
        # 4. 找到每行的 top-k attention scores 的阈值
        # k_values shape: [tgt_len] -> [1, 1, tgt_len, 1] for broadcasting
        k_values_expanded = k_values.view(1, 1, tgt_len, 1).expand(bsz, num_heads, -1, -1)
        
        # 对 causally_masked_attn 沿最后一个维度排序
        sorted_attn, _ = torch.sort(causally_masked_attn, dim=-1, descending=True)
        
        # k_indices 是 k_values-1，因为是0-indexed
        k_indices = (k_values_expanded - 1).clamp(min=0)
        # 使用 gather 获取每行的第 k 大的值作为阈值
        thresholds = torch.gather(sorted_attn, dim=-1, index=k_indices)
    
        # 5. 生成最终的布尔掩码
        # 一个 score 必须大于等于它所在行的第k大的 score 才会被选中
        prefetch_mask_bool = (causally_masked_attn >= thresholds)
    
        # 6. 将布尔掩码转换为 additive mask for attention
        final_mask = torch.zeros_like(attn_weights, device=device, dtype=dtype)
        final_mask.masked_fill_(~prefetch_mask_bool, min_val)
    
        return final_mask
    
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        predictor_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        
        attn_weights_reshaped = None
    
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
    
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    
        if self.is_decoder:
            past_key_value = (key_states, value_states)
    
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
    
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        final_attention_mask = None
        #final_attention_mask = self._create_topk_from_attn_mask(attn_weights)
        
        
        # --- Start of Masking Logic ---
        if predictor_logits is not None:
            # 使用新的、正确的因果预取掩码生成逻辑
            final_attention_mask = self._create_causal_prefetch_mask(
                predictor_logits=predictor_logits,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
                bsz=bsz,
                tgt_len=tgt_len,
                src_len=src_len,
            )
        else:
            # 默认情况：标准的因果掩码
            final_attention_mask = _make_causal_mask(
                (bsz, tgt_len), hidden_states.dtype, device=hidden_states.device, past_key_values_length=src_len - tgt_len
            )
        # (重要) 总是应用 padding mask
        if attention_mask is not None:
            expanded_padding_mask = _expand_mask(attention_mask, hidden_states.dtype, tgt_len)
            final_attention_mask = final_attention_mask + expanded_padding_mask
        # --- End of Masking Logic ---
    
        if final_attention_mask.size() not in [(bsz, 1, tgt_len, src_len), (bsz, self.num_heads, tgt_len, src_len)]:
             raise ValueError(f"Attention mask has an incorrect size: {final_attention_mask.size()}")
        
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + final_attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
    
        if output_attentions:
            attn_weights_reshaped = attn_probs.view(bsz, self.num_heads, tgt_len, src_len)
        
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    
        attn_output = torch.bmm(attn_probs, value_states)
    
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
    
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
    
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
    
        return attn_output, attn_weights_reshaped, past_key_value


class OPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig, layer_idx: int):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=config.enable_bias,
            # <<< MODIFIED: 将 config 传递给 OPTAttention >>>
            config=config,
            layer_idx=layer_idx,
        )
        # ... (rest of __init__ is unchanged) ...
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        # <<< MODIFIED: Added prefetch_indices parameter >>>
        predictor_logits: Optional[torch.Tensor] = None,
        predictor_inputs: Optional[torch.Tensor] = None, # <<< 新增这行
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            # <<< MODIFIED: Pass prefetch_indices to OPTAttention >>>
            predictor_logits=predictor_logits,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # ... (rest of forward method is unchanged) ...
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = (residual + hidden_states).view(hidden_states_shape)
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


OPT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OPTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTPreTrainedModel(PreTrainedModel):
    config_class = OPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OPTDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (OPTDecoder)):
            module.gradient_checkpointing = value


OPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class OPTDecoder(OPTPreTrainedModel):
    # ... (__init__ and other methods are unchanged) ...
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine)
        else:
            self.final_layer_norm = None
        self.layers = nn.ModuleList([OPTDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self.post_init()

    # ... (other methods are unchanged) ...
    def get_input_embeddings(self):
        return self.embed_tokens
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # ... (此函数保持不变) ...
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(input_shape, inputs_embeds.dtype, device=inputs_embeds.device, past_key_values_length=past_key_values_length)
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(inputs_embeds.device)
            combined_attention_mask = (expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask)
        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # <<< MODIFIED: Added predictor parameter >>>
        predictor: Optional[nn.Module] = None,
        predictor_inputs: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
                # ... (initial part of forward method is mostly unchanged) ...
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # <<< FIX STARTS HERE >>>
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        
        # --- Create a default attention_mask if not provided ---
        # This is CRITICAL for positional embeddings, which always need an attention_mask.
        # We will call this `pos_embed_attention_mask` to distinguish it from the causal mask.
        if attention_mask is None:
            #attention_mask = torch.ones(input_shape, device=inputs_embeds.device if inputs_embeds is not None else input_ids.device)
            attention_mask = torch.ones(input_shape, device=inputs_embeds.device)
        
        # <<< FIX ENDS HERE >>> 1
        # <<< MODIFIED: 根据配置列表选择性地启用神经网络预取 >>>
        predictor_logits_per_layer = None
        if (
            hasattr(self.config, "prefetch_strategy")
            and self.config.prefetch_strategy == "neural"
            and predictor is not None
            and not self.training
        ):
            logger.info("Running Neural Prefetcher, passing logits to attention layer...")
            with torch.no_grad():
                # 防御性检查
                if predictor_inputs is None:
                    raise ValueError("predictor_inputs is None! Please check OPTForCausalLM.forward")
                
                # <<< 使用 13B 的输入 predictor_inputs，而不是 6.7B 的 inputs_embeds >>>
                predicted_logits = predictor(predictor_inputs.to(torch.float32))

                predictor_logits_per_layer = []
                num_predict_layers = predicted_logits.shape[1] # 通常是 5

                # 从 config 中获取要进行预取的层列表，如果未定义则默认为 None
                prefetch_layers_list = getattr(self.config, "prefetch_layers_list", None)
                
                if prefetch_layers_list:
                    logger.info(f"Applying neural prefetch ONLY to layers specified in prefetch_layers_list: {prefetch_layers_list}")
                else:
                    # 如果列表不存在或为空，这是一个重要的调试信息
                    logger.warning("prefetch_layers_list not found in config or is empty. All first 5 layers will use full cache.")


                # 2. 根据新逻辑为每一层准备 logits 或 None
                for layer_idx in range(self.config.num_hidden_layers):
                    # --- 逻辑核心：处理前 num_predict_layers 层 (e.g., 0-4) ---
                    if layer_idx < num_predict_layers:
                        # 检查当前层是否在指定的预取列表中
                        if prefetch_layers_list and layer_idx in prefetch_layers_list:
                            # 如果在列表中，使用该层的预测 logits
                            logits_for_layer = predicted_logits[:, layer_idx, :]
                            predictor_logits_per_layer.append(logits_for_layer)
                        else:
                            # 如果不在列表中（或列表未提供），则不进行预取（使用全缓存）
                            # 传递 None 会让 OPTAttention 模块回退到标准因果掩码
                            predictor_logits_per_layer.append(None)
                    
                    # --- 逻辑核心：处理后续层 (e.g., 5 及以后) ---
                    else:
                        # 逻辑不变：这些层遵循第 num_predict_layers 层的预测结果
                        prediction_source_idx = num_predict_layers - 1 # 使用最后一个预测结果
                        logits_for_layer = predicted_logits[:, prediction_source_idx, :]
                        predictor_logits_per_layer.append(logits_for_layer)

        # <<< END OF MODIFICATION >>>

        # ... (rest of the method continues, passing the indices down) ...
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        
        # <<< MODIFIED: Determine which attention mask to use >>>
        # If we are prefetching, we will generate the mask inside OPTAttention and don't need the default causal mask here.
        

        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds
        
        # ... (gradient checkpointing and other setup unchanged) ...
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            # ... (layerdrop logic unchanged) ...
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            # <<< MODIFIED: Get prefetch indices for the current layer >>>
            current_layer_predictor_logits = predictor_logits_per_layer[idx] if predictor_logits_per_layer is not None else None

            # ... (gradient checkpointing logic unchanged, but need to pass the new arg) ...
            if self.gradient_checkpointing and self.training:
                # ...
                pass
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    # Pass the original mask if not prefetching, otherwise pass None
                    attention_mask=attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    # <<< MODIFIED: Pass indices to OPTDecoderLayer >>>
                    predictor_logits=current_layer_predictor_logits,
                )
            
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        # ... (final part of the method is unchanged) ...
        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns,)


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = OPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 新增参数
        predictor: Optional[nn.Module] = None,
        predictor_inputs: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # 传递 predictor
            predictor=predictor,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


class OPTForCausalLM(OPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        
        # --- NEURAL PREFETCHER INITIALIZATION ---
        self.predictor = None
        self.predictor_embed_13b = None # <<< 新增
        if hasattr(config, "prefetch_strategy") and config.prefetch_strategy == 'neural':
            NUM_PREDICT_LAYERS = 5
            
            PREDICTOR_NHEAD = 4
            PREDICTOR_ENCODER_LAYERS = 1
            PREDICTOR_D_HID = 2048
            '''
            PREDICTOR_NHEAD = 16
            PREDICTOR_ENCODER_LAYERS = 8
            PREDICTOR_D_HID = 4096
            '''
            self.predictor = PredictorTransformer(
                input_dim=5120,
                num_opt_layers=NUM_PREDICT_LAYERS,
                nhead=PREDICTOR_NHEAD,
                num_encoder_layers=PREDICTOR_ENCODER_LAYERS,
                d_hid=PREDICTOR_D_HID
            )
            import os
            self.predictor_embed_13b = nn.Embedding(config.vocab_size, 5120, padding_idx=config.pad_token_id)
            embed_path = "/home/liushenyu/test/opt_13b_embed_weights.pth"
            if os.path.exists(embed_path):
                print(f"===================================")
                print(f"成功挂载 13B Embedding 权重: {embed_path}")
                print(f"===================================")
                self.predictor_embed_13b.weight.data.copy_(torch.load(embed_path, map_location="cpu"))
            else:
                print(f"警告: 找不到 13B Embedding 文件 {embed_path}!")
            
            self.predictor_embed_13b.requires_grad_(False)
            # 权重将由外部的 run_lm_eval_harness.py 脚本加载
        # --- END OF INITIALIZATION ---

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        predictor_inputs = None
        if self.predictor is not None and self.predictor_embed_13b is not None and input_ids is not None:
            # 用独立的表查出 5120 维度的向量，并转到与当前模型所在的 GPU 设备
            predictor_inputs = self.predictor_embed_13b(input_ids).to(self.device)

        # 传递给 decoder
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            predictor=self.predictor,
            predictor_inputs=predictor_inputs, # <<< 把算好的 13B embedding 传进去
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    # ... (剩余的 prepare_inputs_for_generation 和 _reorder_cache 方法保持不变) ...
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The OPT Model transformer with a sequence classification head on top (linear layer).

    [`OPTForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    OPT_START_DOCSTRING,
)
class OPTForSequenceClassification(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = OPTModel(config)
        self.score = nn.Linear(config.word_embed_proj_dim, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value


@add_start_docstrings(
    """
    The OPT Model transformer with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    OPT_START_DOCSTRING,
)
class OPTForQuestionAnswering(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.model = OPTModel(config)
        self.qa_outputs = nn.Linear(config.word_embed_proj_dim, 2)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForQuestionAnswering
        >>> import torch

        >>> torch.manual_seed(4)  # doctest: +IGNORE_RESULT
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> # note: we are loading a OPTForQuestionAnswering from the hub here,
        >>> # so the head will be randomly initialized, hence the predictions will be random
        >>> model = OPTForQuestionAnswering.from_pretrained("facebook/opt-350m")

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

        >>> inputs = tokenizer(question, text, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> answer_start_index = outputs.start_logits.argmax()
        >>> answer_end_index = outputs.end_logits.argmax()

        >>> answer_offset = len(tokenizer(question)[0])

        >>> predict_answer_tokens = inputs.input_ids[
        ...     0, answer_offset + answer_start_index : answer_offset + answer_end_index + 1
        ... ]
        >>> predicted = tokenizer.decode(predict_answer_tokens)
        >>> predicted
        ' a nice puppet'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value
