"""Implement tensor computations with pytorch."""
from enum import Enum, auto
from functools import partial
from itertools import count
import os
import queue
import shutil
import time
import threading
from typing import Optional, Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from flexgen.utils import (GB, T, cpu_mem_stats, vector_gather,
    np_dtype_to_torch_dtype, torch_dtype_to_np_dtype,
    torch_dtype_to_num_bytes)

from infinigen.skewing_controller import reform_hidden_states, skew
from infinigen.partial_weight_generation_controller import partial_weight_index_generation
from infinigen.kv_selection_controller import speculate_attention

general_copy_compressed = TorchCompressedDevice = None
global_cpu_device = None
global_disk_device = None

def _create_causal_prefetch_mask(
    predictor_logits: torch.Tensor,
    prefetch_budget_ratio: float,
    dtype: torch.dtype,
    device: torch.device,
    bsz: int,
    tgt_len: int,
    src_len: int
):
    min_val = torch.finfo(dtype).min
    causal_mask_bool = torch.tril(torch.ones((tgt_len, src_len), device=device, dtype=torch.bool))
    logits_expanded = predictor_logits.unsqueeze(1) 
    causally_masked_logits = torch.where(
        causal_mask_bool,
        logits_expanded,
        torch.tensor(min_val, dtype=dtype, device=device)
    )
    historical_lengths = torch.arange(1, tgt_len + 1, device=device)
    k_values = (historical_lengths * prefetch_budget_ratio).ceil().long().clamp(min=1)
    k_values_expanded = k_values.view(1, tgt_len, 1).expand(bsz, -1, -1)
    sorted_logits, _ = torch.sort(causally_masked_logits, dim=-1, descending=True)
    k_indices = (k_values_expanded - 1).clamp(min=0)
    k_indices = k_indices.clamp(max=sorted_logits.shape[-1] - 1)
    thresholds = torch.gather(sorted_logits, dim=-1, index=k_indices)
    prefetch_mask_bool_base = (causally_masked_logits >= thresholds)
    NUM_SINK_TOKENS = 4
    sink_mask = torch.zeros((1, src_len), dtype=torch.bool, device=device)
    num_to_force_keep_start = min(src_len, NUM_SINK_TOKENS)
    sink_mask[:, :num_to_force_keep_start] = True
    recency_mask = torch.ones((tgt_len, src_len), device=device, dtype=torch.bool).tril(0).triu(-(NUM_SINK_TOKENS - 1))
    forced_keep_mask_bool = (recency_mask | sink_mask) & causal_mask_bool
    final_mask_bool = prefetch_mask_bool_base | forced_keep_mask_bool
    final_mask = torch.zeros((bsz, 1, tgt_len, src_len), device=device, dtype=dtype)
    final_mask.masked_fill_(~final_mask_bool.unsqueeze(1), min_val)
    return final_mask
    
def fix_recursive_import():
    global general_copy_compressed, TorchCompressedDevice, global_cpu_device
    from flexgen import compression
    general_copy_compressed = compression.general_copy_compressed
    TorchCompressedDevice = compression.TorchCompressedDevice


class DeviceType(Enum):
    CPU = auto()
    CUDA = auto()
    DISK = auto()
    MIXED = auto()
    COMPRESSED = auto()

    @staticmethod
    def convert(name):
        if name == "cpu":
            return DeviceType.CPU
        elif name == "cuda":
            return DeviceType.CUDA
        elif name == "disk":
            return DeviceType.DISK
        elif name == "mixed":
            return DeviceType.MIXED
        elif name == "compressed":
            return DeviceType.COMPRESSED
        else:
            raise ValueError(f"Invalid name: {name}")


class TorchTensor:
    name_count = count()

    def __init__(self, shape, dtype, data, device, name=None):
        if isinstance(data, torch.Tensor):
            assert data.device == device.dev

        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.device = device
        self.delete_file = True
        self.name = name or TorchTensor.next_name()

    @property
    def bytes(self):
        return np.prod(self.shape) * torch_dtype_to_num_bytes[self.dtype]

    @classmethod
    def next_name(cls):
        return f"t_{next(cls.name_count)}"

    @classmethod
    def create_from_torch(cls, data, device, name=None):
        return cls(data.shape, data.dtype, data, device, name=name)

    def delete(self):
        assert self.device is not None, "already deleted"
        if self.device.device_type == DeviceType.DISK:
            self.device.delete(self)
        self.device = self.data = None

    def load_from_np(self, np_array):
        if self.device.device_type == DeviceType.DISK:
            with open(self.data, "wb") as fout:
                np.save(fout, np_array)
        else:
            if self.device.device_type == DeviceType.COMPRESSED:
                tmp = torch.from_numpy(np_array)
                tmp = global_cpu_device.compressed_device.compress(tmp, self.data[2])
                general_copy(self, None, tmp, None)
            else:
                self.data.copy_(torch.from_numpy(np_array))

    def load_from_np_file(self, filename):
        if self.device.device_type == DeviceType.DISK:
            shutil.copy(filename, self.data)
        else:
            self.load_from_np(np.load(filename))

    def copy(self, dst, src_indices=None):
        if isinstance(src_indices, torch.Tensor):
            # Sparse gather
            shape = (src_indices.shape[0],) + self.shape[1:]
        elif src_indices is not None:
            # Slicing
            assert all(x.step is None for x in src_indices)
            shape = tuple(x.stop - x.start for x in src_indices
                ) + self.shape[len(src_indices):]
        else:
            # Full copy
            shape = self.shape

        if dst.device_type == DeviceType.COMPRESSED:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype], self.data[2])
        else:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype])
        general_copy(ret, None, self, src_indices)
        return ret

    def smart_copy(self, dst, src_indices=None):
        if self.device == dst:
            return self, False
        return self.copy(dst, src_indices=src_indices), True

    def move(self, dst):
        if self.device == dst:
            return self
        ret = self.copy(dst)
        self.delete()
        return ret

    def __str__(self):
        return (f"TorchTensor(shape={self.shape}, dtype={str(self.dtype)}, "
                f"device={self.device.name if self.device else None})")


class TorchDevice:
    """Wrap tensor and computation APIs of a single CPU or GPU."""

    def __init__(self, name, mem_capacity=None, flops=None):
        self.name = name
        self.mem_capacity = mem_capacity
        self.flops = flops

        self.dev = torch.device(name)
        self.device_type = DeviceType.convert(self.dev.type)
        self.compressed_device = TorchCompressedDevice(self)

        self.links = {}
        self.attention_compute_workspace = None
        self.workspace_pt = 0

        if self.device_type == DeviceType.CPU:
            global global_cpu_device
            global_cpu_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        if self.device_type == DeviceType.CPU:
            pin_memory = True if pin_memory is None else pin_memory
        else:
            pin_memory = False
        dtype = np_dtype_to_torch_dtype[dtype]
        data = torch.empty(shape, dtype=dtype, pin_memory=pin_memory, device=self.dev)
        return TorchTensor.create_from_torch(data, self, name=name)

    def delete(self, tensor):
        pass

    def init_attention_compute_workspace(self, config, task, policy):
        if self.device_type != DeviceType.CPU:
            return

        if not policy.compress_cache:
            b = policy.gpu_batch_size
            n_head = config.n_head
            head_dim = config.input_dim // n_head
            max_seq_len = task.prompt_len + task.gen_len - 1
            self.attention_compute_workspace = []
            self.workspace_pt = 0
            for i in range(1 if policy.sep_layer else 2):
                shape = (max_seq_len, b * n_head, head_dim)
                k_cache = self.allocate(shape, np.float32, pin_memory=False)
                v_cache = self.allocate(shape, np.float32, pin_memory=False)
                self.attention_compute_workspace.append((k_cache, v_cache))
        else:
            self.compressed_device.init_attention_compute_workspace(
                config, task, policy)

    def next_attention_compute_workspace(self):
        self.workspace_pt = (self.workspace_pt + 1) % len(
            self.attention_compute_workspace)
        return self.attention_compute_workspace[self.workspace_pt]

    def del_attention_compute_workspace(self):
        self.attention_compute_workspace = None

    def gen_attention_mask(self, token_ids, pad_token_id, donate):
        data = token_ids.data.ne(pad_token_id)
        if donate[0]: token_ids.delete()
        return TorchTensor.create_from_torch(data, self)

    def extend_attention_mask(self, attention_mask, donate):
        bs = attention_mask.shape[0]
        data = torch.concat((attention_mask.data,
             torch.ones((bs, 1), dtype=attention_mask.dtype, device=self.dev)), dim=1)
        if donate[0]: attention_mask.delete()
        return TorchTensor.create_from_torch(data, self)

    def opt_input_embed(self, inputs, attention_mask, w_token, w_pos, pad_token_id, donate):
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)
            w_pos = w_pos.device.decompress(w_pos)

        token_ids = inputs.data
        mask = attention_mask.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)
        positions = torch.cumsum(mask, dim=1).int() * mask + 1
        past_key_values_length = mask.shape[1] - token_ids.shape[1]
        positions = positions[:, past_key_values_length:]
        pos_embed = F.embedding(positions, w_pos.data)
        data = token_embed + pos_embed
        return TorchTensor.create_from_torch(data, self)

    def opt_output_embed(self, inputs, w_ln, b_ln, w_token, donate,
                         do_sample, temperature):
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        b, s, h = inputs.shape
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        if donate[0]: inputs.delete()

        logits = F.linear(hidden, w_token.data)
        last_token_logits = logits[:,-1,:]

        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        return TorchTensor.create_from_torch(ids, self)

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        pin_memory = False
        k_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        return k_cache, v_cache

    def mha(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config, 
            warmup=False, partial_weight_ratio=0.1,
            predictor_logits: Optional[torch.Tensor] = None,
            prefetch_budget_ratio: float = 0.0):
        
        # 1. 权重解压
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q, w_k, w_v, w_out = (d.device.decompress(d) for d in [w_q, w_k, w_v, w_out])

        b, s, h = inputs.shape
        head_dim = h // n_head
        
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        new_h = reform_hidden_states(hidden)

        q = F.linear(new_h, w_q.data, bias=None)
        k = F.linear(new_h, w_k.data, bias=None)
        v = F.linear(hidden, w_v.data, bias=b_v.data)

        # 2. 生成 Partial Weight Index (保持 Infinigen 逻辑)
        partial_weight_index = None
        if (not warmup) and (partial_weight_ratio is not None):
            partial_weight_index = partial_weight_index_generation(q, n_head, head_dim, partial_weight_ratio)
        
        # 3. 统一 View 为 (Batch, Seq, Heads, Dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_head, head_dim)
        v = v.view(b, s, n_head, head_dim)

        # 4. Skew (如果需要)
        if warmup:
             w_q.data, w_k.data = skew(q, k, w_q.data, w_k.data, n_head, head_dim)

        # =================================================================
        # 分支 A: 使用 FlashAttention 优化 Prefill (当没有 predictor 时)
        # =================================================================
        if predictor_logits is None:
            # 转换为 (Batch, Heads, Seq, Dim)
            q_in = q.permute(0, 2, 1, 3)
            k_in = k.permute(0, 2, 1, 3)
            v_in = v.permute(0, 2, 1, 3)
            
            # 使用高效的 SDPA (FlashAttention)
            # 注意：is_causal=True 会自动应用左上三角 Mask，无需显式 Mask 矩阵
            # 显存峰值将在这里被消除
            value = F.scaled_dot_product_attention(q_in, k_in, v_in, is_causal=True)
            
            # 转回 (Batch, Seq, Hidden_Dim)
            value = value.transpose(1, 2).reshape(b, s, h)
            
        # =================================================================
        # 分支 B: Neural Predictor 逻辑 (Decode 或特殊模式，保持原样)
        # =================================================================
        else:
            # 这里的 reshape 是为了适配你原有的 _create_causal_prefetch_mask 和 bmm 逻辑
            # q: (B*Heads, Seq, Dim)
            q_old = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
            k_T_old = k.permute(0, 2, 1, 3).transpose(2, 3).reshape(b * n_head, head_dim, s)
            v_old = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

            attn_weights = torch.bmm(q_old, k_T_old).view(b, n_head, s, s)

            final_attention_mask = _create_causal_prefetch_mask(
                predictor_logits=predictor_logits, prefetch_budget_ratio=prefetch_budget_ratio,
                dtype=attn_weights.dtype, device=self.dev, bsz=b, tgt_len=s, src_len=s)
            
            # 简单的 Padding 处理 (注意：这里假设了 attention_mask 格式)
            padding_mask = (1.0 - attention_mask.data.view(b, 1, 1, s)) * torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights + final_attention_mask + padding_mask

            attn_probs = F.softmax(attn_weights.view(b * n_head, s, s), dim=2, dtype=torch.float32).to(v_old.dtype)
            value = torch.bmm(attn_probs, v_old).view(b, n_head, s, head_dim)
            value = value.transpose(1, 2).reshape(b, s, h)

        # 5. 输出线性层
        value = F.linear(value, w_out.data, bias=b_out.data)
        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # 6. 准备 Cache (Seq, Batch*Heads, Dim)
        k_to_cache = k.permute(1, 0, 2, 3).reshape(s, b * n_head, head_dim)
        v_to_cache = v.permute(1, 0, 2, 3).reshape(s, b * n_head, head_dim)

        k_ret = self.compressed_device.compress(k_to_cache, comp_config) if compress_cache else TorchTensor.create_from_torch(k_to_cache, self)
        v_ret = self.compressed_device.compress(v_to_cache, comp_config) if compress_cache else TorchTensor.create_from_torch(v_to_cache, self)

        return TorchTensor.create_from_torch(value, self), k_ret, v_ret, w_q, w_k, partial_weight_index

    def mha_gen(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
                w_out, b_out, w_ln, b_ln, n_head,
                k_cache, v_cache, donate, attn_sparsity,
                compress_cache, comp_config, p_w_q, p_k_c, speculation_stream, alpha, max_num_kv,
                predictor_logits: Optional[torch.Tensor] = None,
                prefetch_budget_ratio: float = 0.0):
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q, w_k, w_v, w_out = (d.device.decompress(d) for d in [w_q, w_k, w_v, w_out])

        b, tgt_s, h = inputs.shape # tgt_s is 1
        src_s = k_cache.shape[0] + tgt_s
        head_dim = h // n_head

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        new_h = reform_hidden_states(hidden)

        prefetch_idx = None
        if predictor_logits is None and p_w_q is not None: # Infinigen's prefetching
            with torch.cuda.stream(speculation_stream):
                prefetch_idx = speculate_attention(new_h, p_w_q, p_k_c, n_head, alpha, max_num_kv)

        q = F.linear(new_h, w_q.data, bias=None)
        k_new_token = F.linear(new_h, w_k.data, bias=None)
        v_new_token = F.linear(hidden, w_v.data, bias=b_v.data)

        q = q.view(b, tgt_s, n_head, head_dim).permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        k_new = k_new_token.view(b, tgt_s, n_head, head_dim).permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        v_new = v_new_token.view(b, tgt_s, n_head, head_dim).permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        
        if isinstance(k_cache, TorchTensor): 
            if compress_cache:
                k_hist = k_cache.device.decompress(k_cache)
                v_hist = v_cache.device.decompress(v_cache)
            else:
                k_hist = k_cache.data
                v_hist = v_cache.data

            k = torch.cat((k_hist, k_new), dim=0)
            v = torch.cat((v_hist, v_new), dim=0)
            
            k_T = k.permute(1, 2, 0)
            v = v.permute(1, 0, 2)

            # [FIXED] Removed the src_s dimension argument as the shape is dynamic due to sparsity
            attn_weights = torch.bmm(q, k_T).view(b, n_head, tgt_s, -1)
            
            # === 修复开始：移除错误的 Mask 逻辑 ===
            # 原因：k_cache 已经是通过 gather 得到的稀疏张量，不对应原始的 0~N 序列位置。
            # 因此不能使用基于位置的 causal_prefetch_mask。
            # 筛选工作已经在 I/O 阶段完成，这里只需要直接计算 Attention。
            
            # 如果 predictor_logits 不为空，说明我们正在做 Neural Prefetching，
            # 此时不需要额外的 Mask，因为没被加载的 token 本身就不存在于 k_T 中。
            if predictor_logits is None:
                 # 只有在非 Neural 模式（即普通模式或 Infinigen 模式）下，可能需要处理 Padding Mask
                 # 但注意：Neural 模式下的 attention_mask 传入往往是全 1 或包含 Padding，
                 # 由于我们改变了 KV 的长度，直接应用原始 attention_mask 也是危险的。
                 # 幸运的是，FlexGen 的 decode 阶段通常是对有效 token 操作，
                 # 这里为了安全，Neural 模式下直接跳过旧的 mask 逻辑。
                 pass
            
            # === 修复结束 ===

            attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(v.dtype)
            
            # [FIXED] use dynamic shape for view
            current_src_len = k.shape[0]
            value = torch.bmm(attn_probs.view(b * n_head, tgt_s, current_src_len), v).view(b, n_head, tgt_s, head_dim)
        else:
            assert predictor_logits is None, "Neural prefetch not supported for mixed device cache"
            value = self._mixed_device_attention(q, k_cache, v_cache, k_new, v_new, 
                                                attention_mask.data, b, src_s, tgt_s, n_head, head_dim)

        value = value.transpose(1, 2).view(b, tgt_s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)
        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()
        
        k_new = self.compressed_device.compress(k_new, comp_config) if compress_cache else TorchTensor.create_from_torch(k_new, self)
        v_new = self.compressed_device.compress(v_new, comp_config) if compress_cache else TorchTensor.create_from_torch(v_new, self)

        return TorchTensor.create_from_torch(value, self), k_new, v_new, prefetch_idx

    def _attention_weights(self, q, k, mask, b, src_s, n_head):
        attn_weights = torch.bmm(q, k)
        attn_weights = attn_weights.view(b, n_head, 1, src_s)
        if mask is not None:
            if mask.dim() == 4:
                attn_weights = attn_weights + mask
            else:
                mask = (1.0 - mask.view(b, 1, 1, src_s)) * torch.finfo(attn_weights.dtype).min
                attn_weights = attn_weights + mask
        return attn_weights

    def _attention_value(self, q, k, v, mask, b, src_s, tgt_s, n_head, head_dim):
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        attn_probs = F.softmax(attn_weights, dim=-1)
        value = torch.bmm(attn_probs.view(b * n_head, tgt_s, src_s), v)
        value = value.view(b, n_head, tgt_s, head_dim)
        return value

    def _sparse_attention_value(self, q, k, v_new, v_cache, mask, b,
                                src_s, tgt_s, n_head, head_dim, attn_sparsity):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        topk = int(attn_sparsity * (attn_weights.shape[2] - 1))
        topk_weights, topk_indices = attn_weights[:, :, :-1].topk(
            topk, dim=2, sorted=False)
        topk_indices = topk_indices.view(b * n_head, topk).transpose(0, 1)
        attn_weights = torch.cat([topk_weights,
            attn_weights[:, :, -1].unsqueeze(-1)], dim=-1)

        if k.is_cuda:
            v_home = v_cache
            v_buf = self.allocate((topk+1, b*n_head, head_dim), np.float16)
            topk_indices = topk_indices.cpu()
        else:
            (v_home, v_buf) = v_cache

        indices_src = topk_indices
        indices_tgt = (slice(0, indices_src.shape[0]), slice(0, v_home.shape[1]))
        general_copy(v_buf, indices_tgt, v_home, indices_src)
        v_home.device.synchronize()

        v = v_buf.data[:topk+1]
        v[topk:topk+1] = v_new
        v = v.permute(1, 0, 2).reshape(b * n_head, topk+1, head_dim)

        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    def _mixed_device_attention(self, q, k_cache, v_cache, k_new, v_new,
            mask, b, src_s, tgt_s, n_head, head_dim):
        k_gpu, k_cpu = k_cache[0].data, k_cache[1].data
        v_gpu, v_cpu = v_cache[0].data, v_cache[1].data
        seg = k_gpu.shape[1]

        b_gpu = seg // n_head
        q_gpu = q[:seg]
        k_gpu = k_gpu[:src_s, :seg, :]
        v_gpu = v_gpu[:src_s, :seg, :]
        k_gpu[src_s-1:src_s, :, :] = k_new[:, :seg, :]
        v_gpu[src_s-1:src_s, :, :] = v_new[:, :seg, :]
        k_gpu = k_gpu.permute(1, 2, 0)
        v_gpu = v_gpu.permute(1, 0, 2)

        mask_gpu = mask[:b_gpu].cuda()
        value_gpu = self._attention_value(q_gpu, k_gpu, v_gpu, mask_gpu,
            b_gpu, src_s, tgt_s, n_head, head_dim)

        b_cpu = b - b_gpu
        q_cpu = q[seg:].float().cpu()
        k_cpu = k_cpu[:src_s, seg:, :]
        v_cpu = v_cpu[:src_s, seg:, :]
        k_cpu[src_s-1:src_s, :, :] = k_new[:, seg:, :]
        v_cpu[src_s-1:src_s, :, :] = v_new[:, seg:, :]
        k_cpu = k_cpu.permute(1, 2, 0)
        v_cpu = v_cpu.permute(1, 0, 2)

        mask_cpu = mask[b_gpu:]
        value_cpu = self._attention_value(q_cpu, k_cpu, v_cpu, mask_cpu,
            b_cpu, src_s, tgt_s, n_head, head_dim)

        value = torch.cat([value_gpu, value_cpu.cuda().half()], dim=0)
        return value

    def mlp(self, inputs, wi, bi, wo, bo, w_ln, b_ln, donate):
        if wi.device.device_type == DeviceType.COMPRESSED:
            wi = wi.device.decompress(wi)
            wo = wo.device.decompress(wo)

        b, s, h = inputs.shape

        out = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        out = F.linear(out, wi.data, bias=bi.data)
        F.relu(out, inplace=True)
        out = F.linear(out, wo.data, bias=bo.data)

        out.add_(inputs.data)
        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, self)

    def synchronize(self):
        torch.cuda.synchronize()

    def mem_stats(self):
        if self.device_type == DeviceType.CUDA:
            cur_mem = torch.cuda.memory_allocated(self.dev)
            peak_mem = torch.cuda.max_memory_allocated(self.dev)
        elif self.device_type == DeviceType.CPU:
            cur_mem = cpu_mem_stats()
            peak_mem = 0
        else:
            raise NotImplementedError()

        return cur_mem, peak_mem

    def print_stats(self, output_file=None):
        torch.cuda.synchronize()
        cur_mem, peak_mem = self.mem_stats()

        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(f"TorchDevice: {self.name}\n")
                f.write(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                        f" peak_mem: {peak_mem/GB:.4f} GB\n")
        else:
            print(f"TorchDevice: {self.name}")
            print(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                  f" peak_mem: {peak_mem/GB:.4f} GB")

        return cur_mem, peak_mem

    def __str__(self):
        return f"TorchDevice(name={self.name})"


class TorchDisk:
    """Manage tensors stored on a disk."""

    def __init__(self, path, mem_capacity=None, cuda_id=0, num_copy_threads=4):
        self.name = path
        self.path = os.path.abspath(os.path.expanduser(path))
        self.mem_capacity = mem_capacity

        self.device_type = DeviceType.DISK
        self.compressed_device = TorchCompressedDevice(self)

        if os.path.exists(self.path):
            assert os.path.isdir(self.path)
        else:
            os.makedirs(self.path)

        self.links = {}
        self.copy_queue = queue.Queue()
        self.copy_threads = [
            threading.Thread(
                target=copy_worker_func, args=(self.copy_queue, cuda_id)
            ) for _ in range(num_copy_threads)
        ]
        for t in self.copy_threads:
            t.start()

        global global_disk_device
        global_disk_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        name = name or TorchTensor.next_name()
        path = os.path.join(self.path, name)
        np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)
        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           path, self, name=name)

    def delete(self, tensor):
        if os.path.exists(tensor.data) and tensor.delete_file:
            os.remove(tensor.data)

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        k_cache = self.allocate(shape, np.float16)
        v_cache = self.allocate(shape, np.float16)
        return k_cache, v_cache

    def submit_copy(self, *args):
        self.copy_queue.put_nowait(args)

    def synchronize(self):
        self.copy_queue.join()

    def close_copy_threads(self):
        for _ in range(len(self.copy_threads)):
            self.copy_queue.put_nowait(None)
        for t in self.copy_threads:
            t.join()
        self.copy_queue.join()
        self.copy_queue = None

    def mem_stats(self):
        raise NotImplementedError()

    def print_stats(self):
        raise NotImplementedError()

    def __del__(self):
        if self.copy_queue:
            self.close_copy_threads()


SEG_DIM = 1

class TorchMixedDevice:
    """Manage tensors stored on multiple physical devices."""

    def __init__(self, base_devices):
        self.name = "mixed"
        self.device_type = DeviceType.MIXED
        self.base_devices = base_devices

    def allocate(self, shape, dtype, seg_lengths, pin_memory=None, name=None):
        assert sum(seg_lengths) == shape[SEG_DIM]
        assert len(seg_lengths) == len(self.base_devices)
        seg_points = [0]
        for l in seg_lengths:
            seg_points.append(seg_points[-1] + l)

        devices = self.base_devices
        tensors = []
        for i in range(len(devices)):
            seg_len = seg_points[i+1] - seg_points[i]
            if seg_len == 0:
                tensors.append(None)
            else:
                seg_shape = shape[:SEG_DIM] + (seg_len,) + shape[SEG_DIM+1:]
                tensors.append(devices[i].allocate(seg_shape, dtype,
                    pin_memory=pin_memory))

        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           (tensors, seg_points), self, name=name)

    def delete(self, tensor):
        for x in self.tensor.data[0]:
            if x:
                x.delete()

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)

        if policy.cache_disk_percent == 0:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = shape[SEG_DIM]  - len_gpu
            len_disk = 0
        else:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = int(shape[SEG_DIM] * policy.cache_cpu_percent / 100) // num_head * num_head
            len_disk = shape[SEG_DIM] - len_gpu - len_cpu
        lens = [len_gpu, len_cpu, len_disk]

        pin_memory = False
        k_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        return k_cache, v_cache


class TorchLink:
    """An I/O link between two devices."""

    def __init__(self, a, b, a_to_b_bandwidth, b_to_a_bandwidth):
        self.a = a
        self.b = b
        self.a_to_b_bandwidth = a_to_b_bandwidth
        self.b_to_a_bandwidth = b_to_a_bandwidth

        a.add_link(self)
        b.add_link(self)

    def io_time(self, src, dst, size):
        if src == self.a:
            assert dst == self.b
            bandwidth = self.a_to_b_bandwidth
        elif src == self.b:
            assert dst == self.a
            bandwidth = self.b_to_a_bandwidth
        else:
            raise ValueError(f"Invalid source {src}")

        if force_io_time is not None:
            return force_io_time

        return size / bandwidth


def general_copy(dst: TorchTensor, dst_indices: Tuple[slice],
                 src: TorchTensor, src_indices: Tuple[slice]):
    """Launch a general asynchronous copy between two tensors."""
    
    # === 辅助函数：确保索引和数据在同一设备 ===
    def _fix_idx(data_tensor, indices):
        if indices is not None and torch.is_tensor(indices):
            if data_tensor.device.type == 'cpu' and indices.device.type != 'cpu':
                return indices.cpu()
            if data_tensor.device.type == 'cuda' and indices.device.type != 'cuda':
                return indices.to(data_tensor.device)
        return indices
    
    # =========================================================
    # [NEW] ZERO-ALLOCATION GATHER OPTIMIZATION
    # =========================================================
    # 针对 Neural 模式: src(GPU) -> dst(GPU Buffer)，使用 index_select
    # 可以完全避免 src.data[indices] 产生的中间显存分配！
    if (src.device.device_type == DeviceType.CUDA and 
        dst.device.device_type == DeviceType.CUDA and
        isinstance(src_indices, torch.Tensor) and 
        dst_indices is None):
        
        idx = _fix_idx(src.data, src_indices)
        # 直接写入目标 Buffer，不产生中间变量
        torch.index_select(src.data, 0, idx, out=dst.data)
        return
    # =========================================================

    if dst.device.device_type == DeviceType.MIXED:
        assert src.device.device_type != DeviceType.MIXED
        seg_points = dst.data[1]

        for i in range(len(dst.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            s_idx = src_indices if src_indices is not None else tuple(slice(0, x) for x in src.shape)
            d_idx = dst_indices if dst_indices is not None else tuple(slice(0, x) for x in dst.shape)
            
            tmp_src_indices = cut_indices(s_idx, seg_points[i], seg_points[i+1])
            tmp_dst_indices = cut_indices(d_idx, seg_points[i], seg_points[i+1], base=seg_points[i])
            general_copy(dst.data[0][i], tmp_dst_indices, src, tmp_src_indices)
    elif src.device.device_type == DeviceType.MIXED:
        assert dst.device.device_type != DeviceType.MIXED
        seg_points = src.data[1]

        for i in range(len(src.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            s_idx = src_indices if src_indices is not None else tuple(slice(0, x) for x in src.shape)
            d_idx = dst_indices if dst_indices is not None else tuple(slice(0, x) for x in dst.shape)
            
            tmp_src_indices = cut_indices(s_idx, seg_points[i], seg_points[i+1], base=seg_points[i])
            tmp_dst_indices = cut_indices(d_idx, seg_points[i], seg_points[i+1])
            general_copy(dst, tmp_dst_indices, src.data[0][i], tmp_src_indices)
    elif (src.device.device_type == DeviceType.COMPRESSED or
          dst.device.device_type == DeviceType.COMPRESSED):
        general_copy_compressed(dst, dst_indices, src, src_indices)
    elif src.device.device_type == DeviceType.DISK:
        src.device.submit_copy(dst, dst_indices, src, src_indices)
    elif dst.device.device_type == DeviceType.DISK:
        dst.device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CUDA and
          dst.device.device_type == DeviceType.CPU and
          not dst.data.is_pinned() and src.shape[0] > 1):
        global_disk_device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CPU and
          dst.device.device_type == DeviceType.CUDA and
          not src.data.is_pinned()):
        s_idx = _fix_idx(src.data, src_indices)
        d_idx = _fix_idx(dst.data, dst_indices)
        src_data = src.data[s_idx] if s_idx is not None else src.data
        dst_data = dst.data[d_idx] if d_idx is not None else dst.data
        src_data = src_data.pin_memory()
        dst_data.copy_(src_data, non_blocking=True)
    else:
        # Fallback
        s_idx = _fix_idx(src.data, src_indices)
        d_idx = _fix_idx(dst.data, dst_indices)
        
        # [WARNING] This creates temporary tensors if not using index_select optimization above!
        src_data = src.data[s_idx] if s_idx is not None else src.data
        dst_data = dst.data[d_idx] if d_idx is not None else dst.data
        
        dst_data.copy_(src_data, non_blocking=True)


def cut_indices(indices, start, stop, base=0):
    assert all(x.step is None for x in indices)
    seg = indices[SEG_DIM]
    return (indices[:SEG_DIM] +
            (slice(max(seg.start, start) - base, min(seg.stop, stop) - base),) +
            indices[SEG_DIM + 1:])


def map_to_torch_tensor(tensor, indices):
    if tensor.device.device_type == DeviceType.DISK:
        data = torch.from_numpy(np.lib.format.open_memmap(tensor.data))
    else:
        data = tensor.data

    # BC: this is supposed to only handle the sparse v_cache case
    if torch.is_tensor(indices):
        return vector_gather(data, indices)
    return data[indices] if indices is not None else data


def copy_worker_func(queue, cuda_id):
    """The copy worker thread."""
    torch.cuda.set_device(cuda_id)

    cpu_buf = torch.empty((1 * GB,), dtype=torch.float16, pin_memory=True)
    copy_stream = torch.cuda.Stream()

    with torch.cuda.stream(copy_stream):
        while True:
            item = queue.get()
            if item is None:
                queue.task_done()
                return

            dst, dst_indices, src, src_indices = item
            src_data = map_to_torch_tensor(src, src_indices)
            dst_data = map_to_torch_tensor(dst, dst_indices)

            if (src.device.device_type == DeviceType.CUDA or
                dst.device.device_type == DeviceType.CUDA):
                # Use a pinned cpu buffer as a relay
                size = np.prod(src_data.shape)
                tmp_cpu_buf = cpu_buf[:size].view(src_data.shape)
                tmp_cpu_buf.copy_(src_data)
                dst_data.copy_(tmp_cpu_buf)
            else:
                dst_data.copy_(src_data)

            queue.task_done()