"""
Usage:
python3 -m flexgen.flex_opt --model facebook/opt-1.3b --gpu-batch-size 32 --percent 100 0 100 0 100 0
"""

import argparse
import dataclasses
import os
import pickle
import time
from typing import Union, List, Optional

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from flexgen.compression import CompressionConfig
from flexgen.opt_config import OptConfig, get_opt_config, download_opt_weights
from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import, TorchTensor)
from flexgen.timer import timers
from flexgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log)
from predictor_model2A import PredictorTransformer
from infinigen.skewing_controller import weight_bias_concat
from infinigen.kv_selection_controller import select_kv
from infinigen.partial_weight_generation_controller import set_partial_cache, set_partial_weight

fix_recursive_import()

DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes


@dataclasses.dataclass(frozen=True)
class Policy:
    gpu_batch_size: int
    num_gpu_batches: int

    # percent = a means a%
    w_gpu_percent: float
    w_cpu_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    act_gpu_percent: float
    act_cpu_percent: float

    # Whether to overlap the I/O and compute
    overlap: bool

    # Whether to separate attention and mlp as two layers
    sep_layer: bool

    # Whether to use pinned memory for weights on CPU
    pin_weight: bool

    # Whether to compute attention on CPU
    cpu_cache_compute: bool

    # Sparsity of attention weights
    attn_sparsity: float

    # Compress weights with group-wise quantization
    compress_weight: bool
    comp_weight_config: CompressionConfig

    # Compress KV cache with group-wise quantization
    compress_cache: bool
    comp_cache_config: CompressionConfig
    prefetch_strategy: Optional[str]
    prefetch_budget_ratio: float
    @property
    def w_disk_percent(self):
        return 100 - self.w_gpu_percent - self.w_cpu_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 100 - self.act_gpu_percent - self.act_cpu_percent


def get_choice(cur_percent, percents, choices):
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 100) < 1e-5

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]


def init_weight_list(weight_specs, policy, env):
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
    dev_choices = [env.disk, env.cpu, env.gpu]

    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)
    ret = []
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        shape, dtype, filename = weight_specs[i]

        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight

        if not compress:
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                weight.load_from_np(np.ones(shape, dtype))
                #weight.load_from_np(np.random.rand(*shape).astype(dtype))
        else:
            weight = home.compressed_device.allocate(
                shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                for i in range(2):
                    x = weight.data[i]
                    x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))

        ret.append(weight)
    return ret


class InputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, s, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.max_seq_len, self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_token
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
            # w_pos
            ((s + 2, h), dtype, path + "decoder.embed_positions.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_token, w_pos = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_token.smart_copy(dst), w_pos.smart_copy(dst)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i, **kwargs): # Added **kwargs for compatibility
        pass  # do nothing

    def prefetch_cache(self, cache_home, cache_read_buf, i, **kwargs):
        pass # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len), np.int64

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, **kwargs):
        # Compute input embedding
        donate = [False] * 4
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_token, donate[2]), (w_pos, donate[3]) = weight_read_buf.pop()
        else:
            (w_token, _), (w_pos, _) = weight_read_buf.val

        h = self.compute.opt_input_embed(h, mask,
            w_token, w_pos, self.config.pad_token_id, donate)
        hidden.val = h


class OutputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_ln
            ((h,), dtype, path + "decoder.layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "decoder.layer_norm.bias"),
            # w_token
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, b_ln, w_token = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((w_ln.smart_copy(dst2), b_ln.smart_copy(dst2),
                w_token.smart_copy(dst1)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i, **kwargs):
        pass  # do nothing
    
    def prefetch_cache(self, cache_home, cache_read_buf, i, **kwargs):
        pass # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, **kwargs):
        donate = [False] * 4
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_ln, donate[1]), (b_ln, donate[2]), (w_token, donate[3]) = weight_read_buf.pop()
        else:
            (w_ln, _), (b_ln, _), (w_token, _) = weight_read_buf.val

        h = self.compute.opt_output_embed(h, w_ln, b_ln, w_token, donate,
            self.task.do_sample, self.task.temperature)
        hidden.val = h


class SelfAttention:
    def __init__(self, config, env, policy, layer_id, enable_prefetching, partial_weight_ratio=0.2, alpha=4, max_num_kv=400):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)

        self.task = None
        self.prefetch_budget_ratio = policy.prefetch_budget_ratio
        self.enable_prefetching = enable_prefetching
        self.prefetch_idx = None
        self.partial_index = None
        self.alpha = alpha
        self.max_num_kv = max_num_kv
        if self.layer_id > 1:
            self.partial_weight_ratio = partial_weight_ratio
        else:
            self.partial_weight_ratio = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}.self_attn"))
        weight_specs = [
            # w_q
            ((h, h), dtype, path + ".q_proj.weight"),
            # b_q
            ((h,), dtype, path + ".q_proj.bias"),
            # w_k
            ((h, h), dtype, path + ".k_proj.weight"),
            # b_k
            ((h,), dtype, path + ".k_proj.bias"),
            # w_v
            ((h, h), dtype, path + ".v_proj.weight"),
            # b_v
            ((h,), dtype, path + ".v_proj.bias"),
            # w_out
            ((h, h), dtype, path + ".out_proj.weight"),
            # b_out
            ((h,), dtype, path + ".out_proj.bias"),
            # w_ln
            ((h,), dtype, path + "_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        # WQ, WK are fused with bias for Infinigen, but not used by neural prefetch
        # Keep this logic for compatibility
        head_dim = h // self.config.n_head
        weights[0].data = weight_bias_concat(weights[0].data, weights[1].data, True, head_dim)
        weights[0].shape = (h, h+1)
        weights[2].data = weight_bias_concat(weights[2].data, weights[3].data)
        weights[2].shape = (h, h+1)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_q.smart_copy(dst1), b_q.smart_copy(dst2),
                w_k.smart_copy(dst1), b_k.smart_copy(dst2),
                w_v.smart_copy(dst1), b_v.smart_copy(dst2),
                w_out.smart_copy(dst1), b_out.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))

    def init_cache_one_gpu_batch(self, cache_home):
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed

        if self.policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device

        cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy)
        cache_home.store(cache)
        # This prefetch_kv is for infinigen, can be ignored for neural
        if self.layer_id > 1:
            self.prefetch_kv = device.allocate((2, self.max_num_kv, cache_home.val[0].shape[1], cache_home.val[0].shape[2]), np.float16, pin_memory=True)

    def load_cache(self, cache_home, cache_read_buf, i, **kwargs):
        if i == 0:  # prefill, no cache
            return

        k_home, v_home = cache_home.val

        if self.policy.compress_cache:
            dst = self.attention_compute.compressed_device
        else:
            dst = self.attention_compute
        
        # In pipelined mode, this function is only called for prefill (i=0) or 
        # when not using pipelining. It loads the full cache.
        # The new `prefetch_cache` will handle sparse loading.
        indices = (slice(0, self.task.prompt_len + i), slice(0, k_home.shape[1]))
        cache_read_buf.store((
            k_home.smart_copy(dst, indices),
            v_home.smart_copy(dst, indices),
        ))

    def prefetch_cache(self, cache_home, cache_read_buf, i, logits):
        """
        Asynchronously loads a sparse subset of the KV cache based on predictor logits.
        This function replaces the need for `load_cache` during decoding in pipelined mode.
        """
        if i == 0:
            return # Prefill doesn't need prefetching
        
        k_home, v_home = cache_home.val
        
        if logits is None: # Fallback to dense loading if no logits are provided
             self.load_cache(cache_home, cache_read_buf, i)
             return

        # 1. Calculate sparse indices from logits
        seq_len = self.task.prompt_len + i
        budget = int(seq_len * self.policy.prefetch_budget_ratio)
        
        # Ensure budget is at least 1 and at most the sequence length
        budget = max(1, min(budget, seq_len -1)) 
        
        # We only need predictions for the current token, which is the last one.
        # Logits shape: [bs, seq_len]
        last_token_logits = logits[:, -1] # Shape: [bs] -> incorrect, predictor gives all layers.
                                          # In OptLM, we already select the last layer, giving [bs, seq_len]
                                          # And for single token gen, it's [bs, 1, seq_len]. Squeeze it.
        if last_token_logits.dim() == 3:
            last_token_logits = last_token_logits.squeeze(1)

        # We care about the first batch element's prediction to guide the load.
        # A more advanced implementation could load different indices per batch element.
        scores = last_token_logits[0, :seq_len]
        top_k_indices = torch.topk(scores, k=budget, dim=-1).indices

        # 2. Force-include the most recent KV pair
        latest_token_idx = torch.tensor([seq_len - 1], device=top_k_indices.device, dtype=top_k_indices.dtype)
        indices = torch.cat([top_k_indices, latest_token_idx])
        indices = torch.unique(indices) # Remove duplicates and sort

        # 3. Asynchronously load k and v caches using these sparse indices
        dst = self.attention_compute.compressed_device if self.policy.compress_cache else self.attention_compute
        
        # `general_copy` supports tensor indices, triggering a sparse gather copy.
        k_sparse = k_home.copy(dst, src_indices=indices)
        v_sparse = v_home.copy(dst, src_indices=indices)
        
        cache_read_buf.store((k_sparse, v_sparse))


    def store_cache(self, cache_home, cache_write_buf, i):
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()

        if i == self.task.gen_len - 1:
            return

        if i == 0:  # prefill
            indices = (slice(0, k_new.shape[0]), slice(0, k_new.shape[1]))
        else:  # decoding
            pos = self.task.prompt_len + i
            indices = (slice(pos - k_new.shape[0], pos), slice(0, k_new.shape[1]))

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
            cache_write_buf, i, k, warmup=None, partial_weight_read_buf=None, partial_cache_read_buf=None, 
            speculation_stream=None, prev_partial_cache_read_buf=None, prev_partial_weight_read_buf=None, weight_home=None,
            # <<< New arguments for prefetching >>>
            predictor_logits: Optional[torch.Tensor] = None):
        
        n_head = self.config.n_head

        donate = [False] * 14
        h, donate[0] = hidden.val, True
        head_dim = h.shape[-1] // n_head

        if k == self.policy.num_gpu_batches - 1:
            ((w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]),
             (w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]), (b_out, donate[9]),
             (w_ln, donate[10]), (b_ln, donate[11])) = weight_read_buf.pop()
        else:
            ((w_q, _), (b_q, _), (w_k, _), (b_k, _),
             (w_v, _), (b_v, _), (w_out, _), (b_out, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val
        
        p_w_q = partial_weight_read_buf.val if self.enable_prefetching and i > 0 and partial_weight_read_buf else None

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            h, new_k_cache, new_v_cache, w_q, w_k, self.partial_index = self.compute.mha(h, mask, w_q, b_q,
                w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                self.policy.compress_cache, self.policy.comp_cache_config, warmup, self.partial_weight_ratio,
                # Pass prefetcher args, they will be used to create the sparse mask
                predictor_logits=predictor_logits,
                prefetch_budget_ratio=self.prefetch_budget_ratio)
            cache_write_buf.store((new_k_cache, new_v_cache))
            
            # Infinigen-specific logic
            if (prev_partial_cache_read_buf is not None) and (not warmup):
                prev_partial_cache_read_buf.store(set_partial_cache(new_k_cache.data, self.partial_index, n_head, head_dim))
                prev_partial_weight_read_buf.store(set_partial_weight(w_q.data, self.partial_index, n_head, head_dim))
            if warmup:
                weight_home.val[0] = w_q.smart_copy(weight_home.val[0].device)[0]
                weight_home.val[2] = w_k.smart_copy(weight_home.val[2].device)[0]

        else:  # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[12]), (v_cache, donate[13]) = cache_read_buf.pop()
            
            partial_k_cache = partial_cache_read_buf.val if self.enable_prefetching and partial_cache_read_buf else None

            # mha_gen now handles predictor_logits to create a sparse attention mask
            h, new_k_cache, new_v_cache, self.prefetch_idx = self.compute.mha_gen(h, mask, w_q,
                b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head,
                k_cache, v_cache, donate, self.policy.attn_sparsity,
                self.policy.compress_cache, self.policy.comp_cache_config, p_w_q, partial_k_cache, speculation_stream, self.alpha, self.max_num_kv,
                predictor_logits=predictor_logits, 
                prefetch_budget_ratio=self.prefetch_budget_ratio)

            cache_write_buf.store((new_k_cache, new_v_cache))
            
            # Infinigen-specific logic
            if (prev_partial_cache_read_buf is not None) and (self.layer_id > 1):
                prev_partial_cache_read_buf.val = torch.cat((prev_partial_cache_read_buf.val, set_partial_cache(new_k_cache.data, self.partial_index, n_head, head_dim)))

        hidden.val = h


class MLP:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}."))
        weight_specs = [
            ((4 * h, h), dtype, path + "fc1.weight"),
            ((4 * h,), dtype, path + "fc1.bias"),
            ((h, 4 * h), dtype, path + "fc2.weight"),
            ((h,), dtype, path + "fc2.bias"),
            ((h,), dtype, path + "final_layer_norm.weight"),
            ((h,), dtype, path + "final_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        wi, bi, wo, bo, w_ln, b_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                wi.smart_copy(dst1), bi.smart_copy(dst2),
                wo.smart_copy(dst1), bo.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass

    def load_cache(self, cache_home, cache_read_buf, i, **kwargs):
        pass

    def prefetch_cache(self, cache_home, cache_read_buf, i, **kwargs):
        pass

    def store_cache(self, cache_home, cache_write_buf, i):
        pass

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, **kwargs):
        donate = [False] * 7
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            ((wi, donate[1]), (bi, donate[2]), (wo, donate[3]), (bo, donate[4]),
             (w_ln, donate[5]), (b_ln, donate[6])) = weight_read_buf.pop()
        else:
            ((wi, _), (bi, _), (wo, _), (bo, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val

        h = self.compute.mlp(h, wi, bi, wo, bo, w_ln, b_ln, donate)
        hidden.val = h


class TransformerLayer: # This class is a wrapper and needs to pass args through
    def __init__(self, config, env, policy, i):
        self.attention = SelfAttention(config, env, policy, i)
        self.mlp = MLP(config, env, policy, i)
        self.policy = policy
        self.compute = self.attention.compute

    def set_task(self, task):
        self.attention.set_task(task)
        self.mlp.set_task(task)

    def init_weight(self, weight_home, path):
        home1, home2 = ValueHolder(), ValueHolder()
        self.attention.init_weight(home1, path)
        self.mlp.init_weight(home2, path)
        weight_home.store((home1, home2))

    def load_weight(self, weight_home, weight_read_buf, k):
        read_buf1, read_buf2 = ValueHolder(), ValueHolder()
        home1, home2 = weight_home.val
        self.attention.load_weight(home1, read_buf1, k)
        self.mlp.load_weight(home2, read_buf2, k)
        if k == 0:
            weight_read_buf.store((read_buf1, read_buf2))

    def init_cache_one_gpu_batch(self, cache_home):
        self.attention.init_cache_one_gpu_batch(cache_home)

    def load_cache(self, cache_home, cache_read_buf, i, **kwargs):
        self.attention.load_cache(cache_home, cache_read_buf, i, **kwargs)

    def prefetch_cache(self, cache_home, cache_read_buf, i, **kwargs):
        self.attention.prefetch_cache(cache_home, cache_read_buf, i, **kwargs)

    def store_cache(self, cache_home, cache_write_buf, i):
        self.attention.store_cache(cache_home, cache_write_buf, i)

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
            cache_write_buf, i, k, **kwargs): # Use kwargs to pass through optional args
        if k == self.policy.num_gpu_batches - 1:
            read_buf1, read_buf2 = weight_read_buf.pop()
        else:
            read_buf1, read_buf2 = weight_read_buf.val
        
        # Pass predictor_logits through to the attention layer
        self.attention.forward(hidden, cache_read_buf, read_buf1, attention_mask,
                               cache_write_buf, i, k, **kwargs)
        self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k)


class OptLM:
    def __init__(self,
                 config: Union[str, OptConfig],
                 env: ExecutionEnv,
                 path: str,
                 policy: Policy,
                 partial_weight_ratio,
                 alpha,
                 max_num_kv,
                 predictor_path: Optional[str] = None):
        if isinstance(config, str):
            config = get_opt_config(config)
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches
        self.predictor = None
        
        if self.policy.prefetch_strategy == 'neural':
            if not predictor_path:
                raise ValueError("--predictor-path must be provided for 'neural' strategy.")
            print("Initializing Neural Prefetch Predictor...")
            self.predictor = PredictorTransformer(
                input_dim=self.config.hidden_size,
                num_opt_layers=5,
                nhead=4,
                num_encoder_layers=1,
                d_hid=2048
            )
            print(f"Loading predictor weights from: {predictor_path}")
            predictor_state_dict = torch.load(predictor_path, map_location="cpu")
            self.predictor.load_state_dict(predictor_state_dict)
            self.predictor.to(self.env.gpu.dev).eval()
            print("Neural predictor loaded and moved to GPU.")
        
        layers = []
        self.attn_layer = []
        layers.append(InputEmbed(self.config, self.env, self.policy))
        for i in range(self.config.num_hidden_layers):
            is_prefetch_layer = (i > 0) and (i != (self.config.num_hidden_layers - 1))
            if policy.sep_layer:
                layers.append(SelfAttention(self.config, self.env, self.policy, i, is_prefetch_layer, partial_weight_ratio, alpha, max_num_kv))
                self.attn_layer.append(len(layers) - 1)
                layers.append(MLP(self.config, self.env, self.policy, i))
            else:
                layers.append(TransformerLayer(self.config, self.env, self.policy, i))
                self.attn_layer.append(len(layers) - 1)
        layers.append(OutputEmbed(self.config, self.env, self.policy))
        self.layers = layers
        self.num_layers = len(layers)

        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        # CUDA streams for normal operation
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()
        
        # CUDA streams for Infinigen speculation
        self.speculation_stream = torch.cuda.Stream()
        self.prefetch_cache_stream = torch.cuda.Stream() # Keep for infinigen
        self.prefetch_evt = torch.cuda.Event() # Keep for infinigen

        # === START: PIPELINE PREFETCHING PRIMITIVES (Requirement 3.1 & 3.2) ===
        self.prefetch_stream = torch.cuda.Stream()
        self.prefetch_event = torch.cuda.Event()
        self.next_prefetch_logits_buf = ValueHolder()
        # === END: PIPELINE PREFETCHING PRIMITIVES ===

        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.partial_cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        self.partial_weight_read_buf = array_1d(num_layers, ValueHolder)
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)

        self.task = None
        self.init_all_weights()

    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)

    def init_weight(self, j):
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.config.name}-np")))
        check_path = os.path.join(expanded_path, "decoder.embed_positions.weight")
        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            download_opt_weights(self.config.name, self.path)

        self.layers[j].init_weight(self.weight_home[j], expanded_path)

    def load_weight(self, i, j, k, overlap=True):
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len: return
        
        stream = self.load_weight_stream if overlap else None
        with torch.cuda.stream(stream):
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)

    def delete_weight(self, j, k):
        if k == 0:
            for x in self.weight_home[j].pop():
                if isinstance(x, ValueHolder):
                    for y in x.pop(): y.delete()
                else: x.delete()

    def init_cache(self, j, k):
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])

    def load_cache(self, i, j, k, overlap=True):
        if i == 0: return

        stream = self.load_cache_stream if overlap else None
        with torch.cuda.stream(stream):
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)

    # === START: New Pipelined Helper Functions (Requirement 3.7 & 3.8) ===
    def _async_predict_for_next_step(self, input_embeds_vh):
        """
        Runs the neural predictor on the prefetch_stream.
        This is triggered by the main stream as soon as input embeddings are available.
        """
        with torch.cuda.stream(self.prefetch_stream):
            with torch.no_grad():
                # The predictor expects float32 input
                input_data = input_embeds_vh.val.data.to(torch.float32)
                predicted_logits = self.predictor(input_data)
                
                # Use the last layer's prediction for all subsequent attention layers
                logits_for_all_layers = predicted_logits[:, -1, :]
                self.next_prefetch_logits_buf.store(logits_for_all_layers)
            
            self.prefetch_event.record()
    
    def _async_load_sparse_cache(self, i, k, prefetch_logits):
        """
        Launches asynchronous sparse KV cache loads for all attention layers on the prefetch_stream.
        """
        with torch.cuda.stream(self.prefetch_stream):
            for j in self.attn_layer:
                self.layers[j].prefetch_cache(
                    self.cache_home[j][k],
                    self.cache_read_buf[j][k],
                    i,
                    logits=prefetch_logits
                )
    # === END: New Pipelined Helper Functions ===

    def store_cache(self, i, j, k, overlap=True):
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1: return
        if i == self.task.gen_len - 1:
            self.cache_write_buf[j][k].pop()
            return

        stream = self.store_cache_stream if overlap else None
        with torch.cuda.stream(stream):
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v: x.delete()

    def load_hidden(self, i, j, k):
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len: return

        dst = self.layers[j].compute
        if j == 0:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            if i == 0:
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:
                pos = self.task.prompt_len + i
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[left:right, pos-1:pos])
        else:
            val = self.hidden[i][j-1][k].pop().move(dst)
        self.hidden[i][j][k].store(val)

    def store_hidden(self, i, j, k):
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1: return

        if j == self.num_layers - 1:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
            pos = self.task.prompt_len + i
            if self.task.stop:
                stopped = self.stopped[left:right]
                self.output_ids[left:right, pos:pos+1] = np.where(stopped, self.config.pad_token_id, ids)
                stopped[:] = np.logical_or(stopped, ids == self.task.stop)
            else:
                self.output_ids[left:right, pos:pos+1] = ids
        else:
            x = self.hidden[i][j][k]
            if x.val: x.val = x.val.move(self.act_home)

    def compute_layer(self, i, j, k, **kwargs):
        # Pass kwargs (like predictor_logits) down to the layer's forward method
        self.layers[j].forward(
            self.hidden[i][j][k],
            self.cache_read_buf[j][k],
            self.weight_read_buf[j],
            self.attention_mask[k],
            self.cache_write_buf[j][k],
            i, k, **kwargs
        )

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def init_all_weights(self):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        for j in range(self.num_layers):
            self.init_weight(j)

    def delete_all_weights(self):
        for j in range(self.num_layers):
            self.delete_weight(j, 0)

    def update_attention_mask(self, i, k):
        if i > 0:
            mask = self.attention_mask[k]
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        gpu_batch_size = self.policy.gpu_batch_size
        left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]
        
        compute_device = self.env.cpu if self.policy.cpu_cache_compute else self.env.gpu
        val = compute_device.allocate((gpu_batch_size, self.task.prompt_len), bool)
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask[k].store(val)

    def generate(self,
                 inputs: Union[np.array, List[List[int]]],
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 stop: Optional[int] = None,
                 debug_mode: Optional[str] = None,
                 cut_gen_len: Optional[int] = None,
                 verbose: int = 0,
                 warmup: bool = False):
        task = Task(inputs=inputs, prompt_len=len(inputs[0]), gen_len=max_new_tokens, cut_gen_len=cut_gen_len,
                    do_sample=do_sample, temperature=temperature, stop=stop)
        
        num_gpu_batches = self.num_gpu_batches
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len
        self.warmup = warmup

        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len), self.config.pad_token_id, dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        
        for j in range(self.num_layers):
            for k in range(num_gpu_batches):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
                self.partial_cache_read_buf[j][k].clear()
        for j in range(self.num_layers):
            self.weight_read_buf[j].clear()
            self.partial_weight_read_buf[j].clear()
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()
        self.hidden = array_3d(gen_len, self.num_layers, num_gpu_batches, ValueHolder)

        self.set_task(task)
        for j in range(self.num_layers):
            for k in range(num_gpu_batches): self.init_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)
        
        # Determine generation loop
        if debug_mode is None:
            if not self.policy.overlap:
                self.generation_loop_normal()
            else:
                # === USE PIPELINED LOOP FOR NEURAL PREFETCHING ===
                if self.policy.prefetch_strategy == 'neural':
                    print("Using Pipelined Neural Prefetching Generation Loop")
                    self.generation_loop_pipelined()
                elif num_gpu_batches == 1:
                    self.generation_loop_overlap_single_batch()
                else:
                    self.generation_loop_overlap_multi_batch()
        elif debug_mode == "fewer_batch":
            if num_gpu_batches == 1: self.generation_loop_debug_single_batch()
            else: self.generation_loop_debug_multi_batch()
        elif debug_mode == "breakdown":
            self.generation_loop_debug_normal()
        else:
            raise ValueError(f"Invalid debug mode: {debug_mode}")

        for j in range(self.num_layers):
            for k in range(num_gpu_batches): self.delete_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()

        return self.output_ids

    def generation_loop_pipelined(self):
        """
        Pipelined generation loop for neural prefetching.
        Overlaps predictor computation and data loading with main model execution.
        """
        # Prefill phase (i=0), no pipeline yet
        i = 0
        timers("generate").start()
        for k in range(self.num_gpu_batches):
            self.update_attention_mask(i, k)

        for j in range(self.num_layers):
            self.load_weight(i, j, 0) # Load all weights for the first (and only) batch group
        self.sync() # Ensure weights are loaded

        for j in range(self.num_layers):
            for k in range(self.num_gpu_batches):
                self.load_hidden(i, j, k)
                # For prefill, there are no logits, so it runs dense attention
                self.compute_layer(i, j, k, predictor_logits=None)
                self.store_hidden(i, j, k)
                self.store_cache(i, j, k)
        
        self.sync()
        timers("generate").stop()
        
        if self.execute_gen_len == 1:
            return

        # Kickstart the pipeline: run prediction for the first decoding step (i=1)
        for k in range(self.num_gpu_batches):
            input_embeds_vh = self.hidden[0][0][k]
            self._async_predict_for_next_step(input_embeds_vh)
        
        # Decoding phase (i > 0) with pipeline
        for i in range(1, self.execute_gen_len):
            timers("generate").start()
            
            # --- PIPELINE STEP 1: Sync and get prediction results for step `i` ---
            # Wait for the prediction launched in step `i-1` to complete
            self.prefetch_event.wait()
            current_logits = self.next_prefetch_logits_buf.pop()
            
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
                # --- PIPELINE STEP 2: Launch async sparse KV load for step `i` ---
                self._async_load_sparse_cache(i, k, current_logits)
            
            # Load weights for all layers for the current step
            for j in range(self.num_layers):
                self.load_weight(i, j, 0)
            self.sync() # Wait for weights

            # --- PIPELINE STEP 3 & 4: Compute InputEmbed and trigger next prediction ---
            j = 0 # InputEmbed layer
            for k in range(self.num_gpu_batches):
                self.load_hidden(i, j, k)
                self.compute_layer(i, j, k) # No logits needed
                self.store_hidden(i, j, k)
                # self.store_cache(i, j, k) # No cache for InputEmbed

            if i < self.execute_gen_len - 1:
                # Trigger prediction for step `i+1` in the background
                for k in range(self.num_gpu_batches):
                    input_embeds_vh = self.hidden[i][0][k]
                    self._async_predict_for_next_step(input_embeds_vh)

            # --- PIPELINE STEP 5, 6, 7: Sync, compute core layers, generate token ---
            for j in range(1, self.num_layers):
                is_first_attn = j == self.attn_layer[0]
                if is_first_attn:
                    # Wait for sparse KV cache data for step `i` to be ready
                    torch.cuda.current_stream().wait_stream(self.prefetch_stream)
                
                for k in range(self.num_gpu_batches):
                    self.load_hidden(i, j, k)
                    # Pass the logits to attention layers for sparse computation
                    kwargs = {'predictor_logits': current_logits} if j in self.attn_layer else {}
                    self.compute_layer(i, j, k, **kwargs)
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k)
            
            self.sync()
            timers("generate").stop()
            if self.task.stop and np.all(self.stopped):
                break

    def generation_loop_normal(self):
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k, overlap=False)

                for k in range(self.num_gpu_batches):
                    self.load_cache(i, j, k, overlap=False)
                    self.load_hidden(i, j, k)
                    
                    if (j in self.attn_layer[1:-1]) and (i > 0): self.sync()
                        
                    self.compute_layer(i, j, k)
                    self.sync()
                    
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k, overlap=False)
                    
                    if self.policy.prefetch_strategy == 'infinigen':
                        if j in self.attn_layer[1:-1] and (i > 0):
                            next_attn = self.attn_layer[self.attn_layer.index(j) + 1]
                            prefetch_idx = self.layers[j].prefetch_idx
                            with torch.cuda.stream(self.prefetch_cache_stream):
                                self.layers[next_attn].prefetch_cache(
                                    self.cache_home[next_attn][k], self.cache_read_buf[next_attn][k],
                                    i, prefetch_idx, self.prefetch_cache_stream)
                            self.prefetch_evt.record()
            
            timers("generate").stop()

    def generation_loop_debug_normal(self):
        # ... (code unchanged) ...
        pass
    
    # ... All other generation loops (debug, overlap_single/multi_batch) remain unchanged ...
    def generation_loop_debug_normal(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill_total").reset()
        timers("decoding_gpu_batch").reset()

        timers("load_weight").reset()
        timers("load_cache_prefill").reset()
        timers("load_cache_decoding").reset()
        timers("store_cache_prefill").reset()
        timers("store_cache_decoding").reset()
        timers("compute_layer_prefill").reset()
        timers("compute_layer_decoding").reset()
        load_weight_timer = timers("load_weight")

        for i in range(self.execute_gen_len):
            if i == 0:
                timers("prefill_total").start()
                load_cache_timer = timers("load_cache_prefill")
                store_cache_timer = timers("store_cache_prefill")
                compute_layer_timer = timers("compute_layer_prefill")
            else:
                load_cache_timer = timers("load_cache_decoding")
                store_cache_timer = timers("store_cache_decoding")
                compute_layer_timer = timers("compute_layer_decoding")

            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)

            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()

                load_weight_timer.start(self.sync)
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k, overlap=False)
                load_weight_timer.stop(self.sync)

                for k in range(self.num_gpu_batches):
                    load_cache_timer.start(self.sync)
                    self.load_cache(i, j, k, overlap=False)
                    load_cache_timer.stop(self.sync)
                    self.load_hidden(i, j, k)
                    compute_layer_timer.start(self.sync)
                    self.compute_layer(i, j, k)
                    compute_layer_timer.stop(self.sync)
                    self.store_hidden(i, j, k)
                    store_cache_timer.start(self.sync)
                    self.store_cache(i, j, k, overlap=False)
                    store_cache_timer.stop(self.sync)

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill_total").stop(self.sync)

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill_total").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

        # Debug the costs of individual functions
        print(f"#layers: {self.num_layers}")

        print(f"#batches prefill:  "
              f"{self.num_layers * self.num_gpu_batches}")
        print(f"#batches decoding: "
              f"{(self.task.gen_len - 1) * self.num_layers * self.num_gpu_batches}")
        print(f"load_weight            (per-layer)"
              f": {np.mean(timers('load_weight').costs):.6f} s")
        for stage in ["prefill", "decoding"]:
            for func in ["load_cache", "store_cache", "compute_layer"]:
                name = func + "_" + stage
                costs = timers(name).costs
                print(f"{name:22s} (per-batch): {np.mean(costs):.6f} s")

    def generation_loop_overlap_single_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k, overlap=True)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                self.load_weight(i, j+1, 0, overlap=True)
                self.load_cache(i, j+1, 0, overlap=True)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0, overlap=True)
                self.store_hidden(i, j, 0)
                self.sync()
            timers("generate").stop()

            if self.task.stop and np.all(self.stopped):
                break

    def generation_loop_overlap_multi_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k, overlap=True)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k, overlap=True)
                    self.load_cache(i, j, k+1, overlap=True)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1, overlap=True)
                    self.sync()
            timers("generate").stop()

        # Epilogue
        self.store_hidden(
            self.execute_gen_len-1, self.num_layers-1, self.num_gpu_batches-1)

    def generation_loop_debug_single_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k, overlap=True)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                self.load_weight(i, j+1, 0, overlap=True)
                self.load_cache(i, j+1, 0, overlap=True)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0, overlap=True)
                self.store_hidden(i, j, 0)
                self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def generation_loop_debug_multi_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k, overlap=True)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k, overlap=True)
                    self.load_cache(i, j, k+1, overlap=True)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1, overlap=True)
                    self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def __del__(self):
        self.delete_all_weights()
# ... (main function and argument parser remain unchanged) ...

def get_filename(args):
    model_size = args.model.split('-')[-1]
    percent = ""
    for i in range(len(args.percent)):
        percent += str(args.percent[i]) + "-"
    filename = f"fo-{model_size}-gbs{args.gpu_batch_size}-" \
               f"ngbs{args.num_gpu_batches}-" \
               f"prompt{args.prompt_len}-" \
               f"gen{args.gen_len}-percent-{percent}"
    if args.cpu_cache_compute:
        filename += "cpu-cache"
    else:
        filename += "gpu-cache"
    if args.compress_weight:
        filename += "-compw"
    if args.compress_cache:
        filename += "-compc"
    return filename


def get_inputs(prompt_len, num_prompts, tokenizer, path):
    prompts = []
    with open(path, 'r') as file:
        prompts.append(file.read())
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len).input_ids
    input_ids[0] = input_ids[0][:prompt_len]
    return (input_ids[0],) * num_prompts

def run_flexgen(args):
    if args.model == "facebook/galactica-30b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b", padding_side="left")
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    # Task and policy
    warmup_inputs = get_inputs(2048, num_prompts, tokenizer, args.warmup_input_path)
    inputs = get_inputs(prompt_len, num_prompts, tokenizer, args.test_input_path)

    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False),
                    prefetch_strategy=args.prefetch_strategy,
                    prefetch_budget_ratio=args.prefetch_budget_ratio)
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    opt_config = get_opt_config(args.model)
    model = OptLM(opt_config, env, args.path, policy, args.partial_weight_ratio, args.alpha, args.max_num_kv,
                  predictor_path=args.predictor_path)

    try:
        output_ids = model.generate(
            warmup_inputs, max_new_tokens=1, verbose=args.verbose, warmup=True)

        timers("generate").reset()
        output_ids = model.generate(
            inputs, max_new_tokens=args.gen_len,
            debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
        costs = timers("generate").costs
    finally:
        env.close_copy_threads()

    # Log output
    prefill_latency = costs[0]
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
    else:
        decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    print("+++++++++++++++++++++++++++++++++++++++++++++++++")
    print("FlexGen")
    print("input: " + str(prompt_len) + " output: " + str(gen_len) + " bsz: " + str(num_prompts))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Total: " + str(total_latency) + " Prefill: " + str(prefill_latency) + " Decode: " + str(decode_latency))
    print("=================================================")

def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b",
        help="The model name.")
    parser.add_argument("--path", type=str, default="~/opt_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--debug-mode", type=str,
        choices=["fewer_batch", "breakdown"])
    parser.add_argument("--gpu-batch-size", type=int, default=4)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")


    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument("--overlap", type=str2bool, nargs='?',
        const=True, default=True)

    parser.add_argument("--alpha", type=int, default=4)
    parser.add_argument("--partial-weight-ratio", type=float, default=0.2)
    parser.add_argument("--max-num-kv", type=int, default=400)
    
    parser.add_argument("--warmup-input-path", type=str)
    parser.add_argument("--test-input-path", type=str)
    parser.add_argument("--prefetch-strategy", type=str, default=None, choices=['infinigen', 'neural'],
                        help="Prefetching strategy to use.")
    parser.add_argument('--predictor-path', type=str, default=None,
                        help="[Neural] Path to the pretrained weights of the neural prefetch predictor.")
    parser.add_argument('--prefetch-budget-ratio', type=float, default=0.3,
                        help="[Neural] The ratio of sequence length to prefetch, constrained by budget.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    assert len(args.percent) == 6

    run_flexgen(args)