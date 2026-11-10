"""V16 layer-streamed KV staging, derived from V12 compact backend.

Full causal SDPA prefill; CPU FP16 KV home; explicit per-layer GPU staging.
Neural selections are shared across heads, distinct for every batch item.
The five trained outputs map to layers 1..5; deeper layers reuse output 5.
Serial and async evaluate the SAME prefix (cached history, without the current
decode input). Async predicts/packs the next step alongside current LLM work.
The newest history KV, unavailable when next-step packing starts, is carried
from the previous decode on GPU rather than read from unwritten CPU memory.
"""
import concurrent.futures
import hashlib
import importlib.util
import math
import resource
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'accuracy/transformers/src'))
sys.path.insert(0, str(ROOT / 'accuracy/lm_eval'))


def load_opt():
    name = 'transformers.models.opt.modeling_opt_reviewer_speed'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / 'accuracy/src/modeling_opt_orig.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def neural_indices(scores, ratio, recent):
    """Historical ceil top-k threshold ties UNION recent; no new sink tokens.

    Input [batch, trained layer outputs, visible cached history].
    Returned CPU indices retain sample-specific variable cardinalities.
    """
    length = scores.shape[-1]
    k = max(1, math.ceil(length * ratio))
    threshold = scores.topk(k, dim=-1).values[..., -1:]
    selected = scores >= threshold
    if recent:
        selected[..., max(0, length-recent):] = True
    return selected.cpu()


def pad_indices(mask, layers):
    """[B, outputs, S] -> [layers, B, K], -1 pads masked attention slots."""
    batch, outputs, _ = mask.shape
    width = int(mask.sum(-1).max())
    indices = torch.full((layers, batch, width), -1, dtype=torch.long)
    for layer in range(layers):
        for b in range(batch):
            idx = mask[b, min(layer, outputs-1)].nonzero().flatten()
            indices[layer, b, :idx.numel()] = idx
    return indices


class Slot:
    def __init__(self, layers, batch, capacity, heads, dim, dtype):
        shape = (layers, 2, batch, capacity, heads, dim)
        self.cpu = torch.empty(shape, dtype=dtype, pin_memory=True)
        # Step-owned pinned CPU storage; GPU layer buffers are shared.
        self.ready = torch.cuda.Event()
        self.used = False


class OffloadEngine:
    def __init__(self, model, predictor=None, predictor_embedding=None):
        self.model = model.eval()
        self.config = model.config
        self.layers = model.model.decoder.layers
        self.nlayers = len(self.layers)
        self.heads = self.config.num_attention_heads
        self.dim = self.config.hidden_size // self.heads
        self.dtype = next(model.parameters()).dtype
        self.predictor = predictor
        self.predictor_embedding = predictor_embedding
        self.predict_stream = torch.cuda.Stream()
        self.copy_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.original_forwards = []
        for layer, block in enumerate(self.layers):
            attn = block.self_attn
            self.original_forwards.append(attn.forward)
            def forward(a, hidden_states, key_value_states=None, past_key_value=None,
                        attention_mask=None, layer_head_mask=None, output_attentions=False,
                        layer=layer):
                if key_value_states is not None or layer_head_mask is not None or output_attentions:
                    raise ValueError('Only unpadded decoder inference is supported')
                return self.attention(layer, a, hidden_states), None, None
            attn.forward = types.MethodType(forward, attn)

    def close(self):
        self.executor.shutdown(wait=True)
        for block, forward in zip(self.layers, self.original_forwards):
            block.self_attn.forward = forward

    def allocate(self, batch, max_length, strategy, ratio, recent):
        self.batch, self.max_length = batch, max_length
        self.strategy, self.ratio, self.recent = strategy, ratio, recent
        shape = (self.nlayers, 2, batch, max_length, self.heads, self.dim)
        self.home = None if strategy == 'full_gpu' else torch.empty(shape, dtype=self.dtype, pin_memory=True)
        self.resident = torch.empty(shape, dtype=self.dtype, device='cuda') if strategy == 'full_gpu' else None
        # Allocate the nominal top-k plus recent union; grow on actual threshold ties.
        self.capacity = (min(max_length, max(1, math.ceil(max_length * ratio)) + recent)
                         if strategy.startswith('neural') else max_length)
        self.staging_growths = []
        self.slots = [] if strategy == 'full_gpu' else [
            Slot(self.nlayers, batch, self.capacity, self.heads, self.dim, self.dtype)
            for _ in range(2 if strategy == 'neural_async' else 1)]
        self.layer_gpu = ([] if strategy == 'full_gpu' else [
            torch.empty((2, batch, self.capacity, self.heads, self.dim),
                        dtype=self.dtype, device='cuda') for _ in range(2)])
        self.layer_consumed = [None, None]
        self.latest = torch.empty((self.nlayers, 2, batch, self.heads, self.dim), dtype=self.dtype, device='cuda')
        self.compute_buffers = []
        self.store_events = [torch.cuda.Event() for _ in self.layers]
        self.valid_cpu = 0
        self.steps = []

    def attention(self, layer, a, hidden):
        b, n, _ = hidden.shape
        q = a.q_proj(hidden).view(b, n, self.heads, self.dim).transpose(1, 2)
        k = a.k_proj(hidden).view(b, n, self.heads, self.dim)
        v = a.v_proj(hidden).view(b, n, self.heads, self.dim)
        if self.prefill:
            out = F.scaled_dot_product_attention(q, k.transpose(1, 2), v.transpose(1, 2),
                                                is_causal=True)
            if self.strategy == 'full_gpu':
                self.resident[layer, 0, :, :n].copy_(k)
                self.resident[layer, 1, :, :n].copy_(v)
            else:
                # Prefill writes are deliberately on the main stream; its timer
                # includes populating CPU home before any decode is launched.
                self.home[layer, 0, :, :n].copy_(k, non_blocking=True)
                self.home[layer, 1, :, :n].copy_(v, non_blocking=True)
            self.latest[layer, 0].copy_(k[:, -1])
            self.latest[layer, 1].copy_(v[:, -1])
        else:
            history = self.history
            if self.strategy == 'full_gpu':
                self.resident[layer, 0, :, history:history+1].copy_(k)
                self.resident[layer, 1, :, history:history+1].copy_(v)
                kh = self.resident[layer, 0, :, :history+1].transpose(1, 2)
                vh = self.resident[layer, 1, :, :history+1].transpose(1, 2)
                out = F.scaled_dot_product_attention(q, kh, vh)
            else:
                plan = self.current_plan
                width, slot = plan['width'], plan['slot']
                torch.cuda.current_stream().wait_event(plan['layer_ready'][layer])
                staged = self.layer_gpu[layer % 2][:, :, :width]
                # Next-step packing may precede computation of the last cached
                # token. Patch precisely that selected tail from GPU carry KV.
                staged.copy_(torch.where(plan['carry'][layer][None, :, :, None, None],
                                         self.latest[layer][:, :, None], staged))
                kh = torch.cat((staged[0], k), dim=1).transpose(1, 2)
                vh = torch.cat((staged[1], v), dim=1).transpose(1, 2)
                valid = torch.cat((plan['valid'][layer], torch.ones((b, 1), dtype=torch.bool, device='cuda')), -1)
                out = F.scaled_dot_product_attention(q, kh, vh,
                                                     attn_mask=valid[:, None, None, :])
                self.latest[layer, 0].copy_(k[:, 0])
                self.latest[layer, 1].copy_(v[:, 0])
                produced = torch.cuda.Event()
                produced.record()
                self.store_stream.wait_event(produced)
                with torch.cuda.stream(self.store_stream):
                    self.home[layer, 0, :, history:history+1].copy_(k, non_blocking=True)
                    self.home[layer, 1, :, history:history+1].copy_(v, non_blocking=True)
                    k.record_stream(self.store_stream)
                    v.record_stream(self.store_stream)
                    stored = torch.cuda.Event()
                    stored.record()
                    self.store_events[layer] = stored
        projected = a.out_proj(out.transpose(1, 2).reshape(b, n, -1))
        if not self.prefill and self.strategy != 'full_gpu':
            # Reuse a lane only after all compute using it is complete.
            consumed = torch.cuda.Event()
            consumed.record()
            self.layer_consumed[layer % 2] = consumed
            if layer + 2 < self.nlayers:
                self.queue_layer_copy(self.current_plan, layer + 2)
        return projected

    def queue_layer_copy(self, plan, layer):
        slot, width = plan['slot'], plan['width']
        lane = layer % 2
        with torch.cuda.stream(self.copy_stream):
            if self.layer_consumed[lane] is not None:
                self.copy_stream.wait_event(self.layer_consumed[lane])
            if layer == 0:
                plan['dma_events'][0].record()
            for kv in range(2):
                for b in range(self.batch):
                    self.layer_gpu[lane][kv, b, :width].copy_(
                        slot.cpu[layer, kv, b, :width], non_blocking=True)
            ready = torch.cuda.Event()
            ready.record()
            plan['layer_ready'][layer] = ready
            if layer == self.nlayers - 1:
                plan['dma_events'][1].record()
                slot.ready.record()

    def start_layer_stream(self, plan):
        width = plan['width']
        if width > self.layer_gpu[0].shape[2]:
            # Rare threshold ties can exceed the nominal selection budget.
            torch.cuda.synchronize()
            previous = self.layer_gpu[0].shape[2]
            self.layer_gpu = [torch.empty((2, self.batch, width, self.heads, self.dim),
                                          dtype=self.dtype, device='cuda') for _ in range(2)]
            self.layer_consumed = [None, None]
            self.staging_growths.append(dict(previous=previous, capacity=width,
                                             history=plan['predictor_prefix_length']))
        plan['layer_ready'] = [None] * self.nlayers
        plan['slot'].used = True
        # Layer 1 transfer can overlap layer 0 computation.
        for layer in range(min(2, self.nlayers)):
            self.queue_layer_copy(plan, layer)

    def forward(self, ids, history=0):
        # Dummy past only communicates the logical position to OPT's decoder;
        # attention data are exclusively in this engine's CPU/GPU buffers.
        if history:
            dummy = ids.new_empty((1, 1, history, 1))
            past = tuple((dummy, dummy) for _ in self.layers)
        else:
            past = None
        mask = torch.ones((ids.shape[0], history+ids.shape[1]), dtype=torch.long, device=ids.device)
        output = self.model.model.decoder(ids, attention_mask=mask, past_key_values=past,
                                          use_cache=False, return_dict=True).last_hidden_state
        return self.model.lm_head(output[:, -1])

    def prepare(self, slot, prefix_ids, history, source_valid, context_ready, store_events):
        """Run in a worker: prediction -> CPU gather -> pinned H2D staging.

        Prefix is immutable, complete visible history. Host waits here affect
        end-to-end time and are NOT disguised as zero-cost GPU event waits.
        """
        started = time.perf_counter()
        with torch.inference_mode(), torch.cuda.device(0):
            pred_events = None
            if self.strategy.startswith('neural'):
                with torch.cuda.stream(self.predict_stream):
                    self.predict_stream.wait_event(context_ready)
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    embedded = F.embedding(prefix_ids, self.predictor_embedding).float()
                    scores = self.predictor(embedded)[:, :5]
                    end.record()
                    mask = neural_indices(scores, self.ratio, self.recent)
                    del embedded, scores
                    pred_events = start, end
                indices = pad_indices(mask, self.nlayers)
            else:
                indices = torch.arange(history).view(1, 1, history).expand(self.nlayers, self.batch, -1).clone()
            selection_end = time.perf_counter()
            width = indices.shape[-1]
            if history > self.max_length:
                raise RuntimeError('Staging history exceeded')
            if width > slot.cpu.shape[3]:
                # GPU lanes grow when this plan begins; preserve all ties.
                if slot.used:
                    slot.ready.synchronize()
                shape = (self.nlayers, 2, self.batch, width, self.heads, self.dim)
                slot.cpu = torch.empty(shape, dtype=self.dtype, pin_memory=True)
                slot.used = False
                self.capacity = max(self.capacity, width)
            valid = indices >= 0
            carry = (indices == history-1) & valid & (indices >= source_valid)
            if bool(((indices >= source_valid) & valid & ~carry).any()):
                raise RuntimeError('Attempted to gather unpublished CPU KV')
            # Before using rows potentially written by the PREVIOUS decode,
            # wait for their D2H completion. Current decode writes a disjoint tail.
            if source_valid > self.prompt_len:
                for event in store_events:
                    event.synchronize()
            if slot.used:
                slot.ready.synchronize()
            pack_start = time.perf_counter()
            for layer in range(self.nlayers):
                for b in range(self.batch):
                    safe = indices[layer, b].clamp(min=0, max=source_valid-1)
                    for kv in range(2):
                        torch.index_select(self.home[layer, kv, b], 0, safe,
                                           out=slot.cpu[layer, kv, b, :width])
            pack_end = time.perf_counter()
            dma_start, dma_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            valid_gpu = valid.to('cuda', non_blocking=False)
            carry_gpu = carry.to('cuda', non_blocking=False)
            return dict(slot=slot, width=width, valid=valid_gpu, carry=carry_gpu,
                        index_hash=hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
                        selected=int(valid.sum()), candidates=self.nlayers*self.batch*history,
                        carry_entries=int(carry.sum()), transferred_bytes=self.nlayers*2*self.batch*width*self.heads*self.dim*2,
                        predictor_events=pred_events, dma_events=(dma_start, dma_end),
                        prediction_selection_host_s=selection_end-started,
                        pack_s=pack_end-pack_start, preparation_host_s=time.perf_counter()-started,
                        predictor_prefix_length=history)

    def generate(self, prompt, gen_len, strategy, ratio=.3, recent=20, forced=None):
        if strategy not in ('full_gpu', 'full_offload', 'neural_serial', 'neural_async'):
            raise ValueError(f'Unknown strategy: {strategy}')
        if not 0 < ratio <= 1 or recent < 0:
            raise ValueError('Invalid selection budget')
        if strategy.startswith('neural') and (self.predictor is None or self.predictor_embedding is None):
            raise ValueError('Neural requires trained predictor and source embeddings')
        batch, prompt_len = prompt.shape
        if gen_len < 2 or prompt_len+gen_len > self.config.max_position_embeddings:
            raise ValueError('Need >=2 output tokens, within OPT position limit')
        setup_start = time.perf_counter()
        torch.cuda.synchronize()
        # Request boundary only; this does not alter selections or KV data.
        torch.cuda.empty_cache()
        self.allocate(batch, prompt_len+gen_len, strategy, ratio, recent)
        self.prompt_len = prompt_len
        tokens = torch.empty((batch, prompt_len+gen_len), dtype=torch.long, device='cuda')
        tokens[:, :prompt_len].copy_(prompt)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        request_setup_s = time.perf_counter() - setup_start
        all_start = time.perf_counter()
        self.prefill = True
        with torch.inference_mode():
            logits = self.forward(prompt)
            prefill_logits = logits.detach()
            tokens[:, prompt_len] = logits.argmax(-1) if forced is None else forced[:, 0]
        torch.cuda.synchronize()
        # Prefill scratch is no longer needed. Include cache release in timing.
        torch.cuda.empty_cache()
        prefill_s = time.perf_counter()-all_start
        prefill_peak = torch.cuda.max_memory_allocated()
        prefill_reserved_peak = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        self.prefill = False
        decode_start = time.perf_counter()
        plans, compute_events, gpu_wait_events, step_cpu = [], [], [], []
        logits_trace = []
        future = None
        try:
            with torch.inference_mode():
                for step in range(1, gen_len):
                    step_start = time.perf_counter()
                    self.history = history = prompt_len+step-1
                    context_ready = torch.cuda.Event()
                    context_ready.record()
                    if strategy != 'full_gpu':
                        wait_start, wait_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        wait_start.record()
                        if strategy == 'neural_async' and future is not None:
                            current = future.result()
                        else:
                            current = self.prepare(self.slots[0], tokens[:, :history], history, history,
                                                   context_ready, tuple(self.store_events))
                        self.current_plan = current
                        plans.append(current)
                        self.start_layer_stream(current)
                        torch.cuda.current_stream().wait_event(current['layer_ready'][0])
                        wait_end.record()
                        gpu_wait_events.append((wait_start, wait_end))
                        if strategy == 'neural_async' and step < gen_len-1:
                            # Include current decode INPUT token, never its not-yet-generated output.
                            next_slot = self.slots[step % 2]
                            future = self.executor.submit(self.prepare, next_slot, tokens[:, :history+1],
                                                          history+1, history, context_ready,
                                                          tuple(self.store_events))
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    logits = self.forward(tokens[:, history:history+1], history)
                    end.record()
                    compute_events.append((start, end))
                    tokens[:, history+1] = logits.argmax(-1) if forced is None else forced[:, step]
                    # Keep only tiny last-position logits for validation; no .cpu() in the loop.
                    logits_trace.append(logits)
                    step_cpu.append(time.perf_counter()-step_start)
            torch.cuda.synchronize()
        finally:
            if future is not None:
                future.result()
            torch.cuda.synchronize()
        decode_s = time.perf_counter()-decode_start
        decode_peak = torch.cuda.max_memory_allocated()
        def milliseconds(events):
            return [a.elapsed_time(b) for a, b in events]
        stats = []
        for plan in plans:
            stats.append({k: v for k, v in plan.items() if k not in ('slot', 'valid', 'carry', 'predictor_events', 'dma_events', 'layer_ready')})
            stats[-1]['predictor_ms'] = milliseconds([plan['predictor_events']])[0] if plan['predictor_events'] else 0.
            stats[-1]['dma_ms'] = milliseconds([plan['dma_events']])[0]
        pinned_home = self.home.numel()*self.home.element_size() if self.home is not None else 0
        pinned_stage = sum(s.cpu.numel()*s.cpu.element_size() for s in self.slots)
        gpu_stage = sum(s.numel()*s.element_size() for s in self.layer_gpu)
        result = dict(strategy=strategy, batch=batch, prompt_len=prompt_len, gen_len=gen_len,
                      nominal_ratio=ratio if strategy.startswith('neural') else 1., recent_window=recent if strategy.startswith('neural') else 0,
                      prefill_s=prefill_s, decode_s=decode_s, total_s=prefill_s+decode_s,
                      request_setup_s=request_setup_s, request_total_including_setup_s=request_setup_s+prefill_s+decode_s,
                      decode_tokens=batch*(gen_len-1), total_output_tokens=batch*gen_len,
                      decode_tokens_per_s=batch*(gen_len-1)/decode_s,
                      end_to_end_output_tokens_per_s=batch*gen_len/(prefill_s+decode_s),
                      prefill_peak_allocated_bytes=prefill_peak, decode_peak_allocated_bytes=decode_peak,
                      peak_allocated_bytes=max(prefill_peak, decode_peak), peak_reserved_bytes=max(prefill_reserved_peak, torch.cuda.max_memory_reserved()),
                      cpu_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      cpu_kv_home_bytes=pinned_home, pinned_staging_bytes=pinned_stage, gpu_staging_bytes=gpu_stage,
                      full_gpu_kv_bytes=self.resident.numel()*self.resident.element_size() if self.resident is not None else 0,
                      staging_capacities=[s.shape[2] for s in self.layer_gpu],
                      staging_growths=list(self.staging_growths),
                      compute_gpu_ms=milliseconds(compute_events), selection_wait_gpu_ms=milliseconds(gpu_wait_events),
                      step_host_submission_s=step_cpu, plans=stats,
                      actual_history_selection_ratio=sum(p['selected'] for p in stats)/sum(p['candidates'] for p in stats) if stats else 1.,
                      h2d_kv_bytes=sum(p['transferred_bytes'] for p in stats),
                      note='V16 layer-streamed GPU KV staging uses two alternating one-layer buffers; pinned CPU staging remains per step. FP32 predictor, selected indices, recent-20, carry and attention match V12. Async predicts and packs the next step during current decode; next-step H2D begins at its decode boundary. dma_ms spans all layer transfers and compute gaps, so it is not pure DMA service time.')
        # Output extraction and diagnostics occur after timed GPU completion.
        result['output_ids'] = tokens[:, prompt_len:].cpu().tolist()
        trace = torch.stack(logits_trace, dim=1).cpu()
        # Release per-run CPU/GPU KV capacity before the next repetition.
        self.current_plan = None
        self.home = self.resident = None
        self.slots = []
        self.layer_gpu = []
        return result, prefill_logits.cpu(), trace
