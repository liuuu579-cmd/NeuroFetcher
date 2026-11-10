#!/usr/bin/env python3
"""Compare original FP32 and experimental FP16 predictor on identical generation."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from reviewer_offload_backend_v12 import load_opt
from reviewer_offload_backend_v15 import OffloadEngine


def result_digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '3'
    assert len(os.sched_getaffinity(0)) <= 10
    torch.set_num_threads(10)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261026)
    torch.cuda.manual_seed_all(20261026)

    model, loading = load_opt().OPTForCausalLM.from_pretrained('/home/liushenyu/opt-13b',
        torch_dtype=torch.float16, device_map={'': 0}, local_files_only=True,
        output_loading_info=True)
    assert not any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
    from predictor_model2A import PredictorTransformer
    predictor = PredictorTransformer(5120, 5, nhead=4, num_encoder_layers=1, d_hid=2048)
    predictor.load_state_dict(torch.load('/home/liushenyu/test/predictor_topk_0.3_iwl_epoch_20.pth',
        map_location='cpu', weights_only=True), strict=True)
    predictor.cuda().eval()
    source_embedding = torch.load('/home/liushenyu/test/opt_13b_embed_weights.pth',
        map_location='cpu', weights_only=True).half().cuda()
    embedding = model.model.decoder.embed_tokens.weight
    assert torch.equal(source_embedding, embedding)
    del source_embedding
    torch.cuda.empty_cache()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/home/liushenyu/opt-13b', use_fast=False, local_files_only=True)
    texts = ('The reader opened the old archive and searched the following page.',
             'A scientist measured the temperature before beginning a new experiment.',
             'The committee reviewed the report and discussed its main conclusion.',
             'Several students walked across campus after the lecture ended.')
    prompt = torch.stack([tokenizer(' '.join([s]*400), add_special_tokens=False,
        return_tensors='pt').input_ids[0, :1920] for s in texts]).cuda()
    engine = OffloadEngine(model, predictor, embedding)
    records = {}
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline, baseline_pre, baseline_logits = engine.generate(
            prompt, 128, 'neural_async', ratio=.3, recent=20)
        records['fp32'] = dict(
            output_sha256=result_digest(baseline['output_ids']),
            output_ids=baseline['output_ids'],
            index_hashes=[p['index_hash'] for p in baseline['plans']],
            peak_allocated_bytes=baseline['peak_allocated_bytes'],
            peak_reserved_bytes=baseline['peak_reserved_bytes'],
            gpu_staging_bytes=baseline['gpu_staging_bytes'],
            predictor_parameter_bytes=sum(p.numel()*p.element_size() for p in predictor.parameters()),
            decode_tokens_per_s=baseline['decode_tokens_per_s'])

        predictor.half()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        compact, compact_pre, compact_logits = engine.generate(
            prompt, 128, 'neural_async', ratio=.3, recent=20)
        records['fp16'] = dict(
            output_sha256=result_digest(compact['output_ids']),
            output_ids=compact['output_ids'],
            index_hashes=[p['index_hash'] for p in compact['plans']],
            peak_allocated_bytes=compact['peak_allocated_bytes'],
            peak_reserved_bytes=compact['peak_reserved_bytes'],
            gpu_staging_bytes=compact['gpu_staging_bytes'],
            predictor_parameter_bytes=sum(p.numel()*p.element_size() for p in predictor.parameters()),
            decode_tokens_per_s=compact['decode_tokens_per_s'])

        index_diffs = [i for i, (a, b) in enumerate(zip(records['fp32']['index_hashes'],
                                                        records['fp16']['index_hashes'])) if a != b]
        token_diffs = [(b, t) for b, (a, t) in enumerate(zip(records['fp32']['output_ids'],
                                                              records['fp16']['output_ids'])) if a != t]
        logits_equal = torch.equal(baseline_pre, compact_pre) and torch.equal(baseline_logits, compact_logits)
        report = dict(complete=True, prompt_len=1920, batch=4, generation_len=128,
            ratio=.3, recent=20, index_difference_steps=index_diffs,
            output_difference_rows=token_diffs, logits_bitwise_equal=logits_equal,
            records=records,
            decision='exact behavior preserved' if not index_diffs and not token_diffs and logits_equal
                     else 'reject as drop-in replacement; evaluate task accuracy before any use')
    finally:
        engine.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2), flush=True)
    print(json.dumps({k: {x: v[x] for x in ('peak_allocated_bytes','peak_reserved_bytes',
        'gpu_staging_bytes','predictor_parameter_bytes','decode_tokens_per_s')}
        for k,v in records.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
