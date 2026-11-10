#!/usr/bin/env python3
"""Check logits and exact selection against the unchanged full-capacity engine."""
import argparse
import json
import os
from pathlib import Path

import torch
from reviewer_offload_backend_v12 import OffloadEngine as OriginalEngine
from reviewer_offload_backend_v16 import OffloadEngine as CompactEngine, load_opt

ROOT = Path(__file__).resolve().parents[1]


def compare(model, predictor, embedding, prompt, forced, ratio=.3, recent=20):
    outputs = {}
    report = {}
    for label, cls in (('compact_v12', OriginalEngine), ('layerwise_v16', CompactEngine)):
        engine = cls(model, predictor, embedding)
        try:
            for strategy in ('neural_serial', 'neural_async'):
                result, pre, logits = engine.generate(prompt, forced.shape[1], strategy,
                    ratio=ratio, recent=recent, forced=forced)
                outputs[label, strategy] = (result, pre, logits)
                report[label + '_' + strategy] = {k: result[k] for k in (
                    'peak_allocated_bytes', 'gpu_staging_bytes', 'actual_history_selection_ratio')}
                if label == 'layerwise_v16':
                    report[label + '_' + strategy]['staging_growths'] = result['staging_growths']
        finally:
            engine.close()
    for strategy in ('neural_serial', 'neural_async'):
        a, b = outputs['compact_v12', strategy], outputs['layerwise_v16', strategy]
        torch.testing.assert_close(a[1], b[1], atol=0, rtol=0)
        torch.testing.assert_close(a[2], b[2], atol=0, rtol=0)
        assert [x['index_hash'] for x in a[0]['plans']] == [x['index_hash'] for x in b[0]['plans']]
        assert a[0]['output_ids'] == b[0]['output_ids']
    torch.testing.assert_close(outputs['layerwise_v16', 'neural_serial'][2],
                               outputs['layerwise_v16', 'neural_async'][2], atol=0, rtol=0)
    report['all_logits_bitwise_equal'] = True
    report['all_index_hashes_equal'] = True
    return report


def tiny():
    # Run all existing correctness checks against the new implementation.
    path = ROOT / 'speedup/test_reviewer_offload_backend.py'
    source = path.read_text().replace('from reviewer_offload_backend import',
                                       'from reviewer_offload_backend_v16 import')
    ns = {'__name__': 'compact_correctness', '__file__': str(path)}
    exec(compile(source, str(path), 'exec'), ns)
    ns['selection_tests']()
    report = {'existing_checks': ns['gpu_tests']()}
    from transformers import OPTConfig
    cfg = OPTConfig(hidden_size=64, word_embed_proj_dim=64, ffn_dim=128,
                    num_hidden_layers=7, num_attention_heads=4, vocab_size=211,
                    max_position_embeddings=128, dropout=0., attention_dropout=0.)
    model = load_opt().OPTForCausalLM(cfg).half().cuda().eval()

    class AllTies(torch.nn.Module):
        def forward(self, x):
            return x.new_zeros((x.shape[0], 5, x.shape[1]))

    ids = torch.randint(3, 211, (2, 64), device='cuda')
    forced = torch.randint(3, 211, (2, 9), device='cuda')
    report['all_ties_adaptive_growth'] = compare(model, AllTies().cuda(),
        torch.randn(211, 64, device='cuda'), ids, forced)
    for strategy in ('neural_serial', 'neural_async'):
        assert report['all_ties_adaptive_growth']['layerwise_v16_' + strategy]['staging_growths']
    return report


def real():
    model, loading = load_opt().OPTForCausalLM.from_pretrained('/home/liushenyu/opt-13b',
        torch_dtype=torch.float16, device_map={'': 0}, local_files_only=True,
        output_loading_info=True)
    assert not any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
    from predictor_model2A import PredictorTransformer
    predictor = PredictorTransformer(5120, 5, nhead=4, num_encoder_layers=1, d_hid=2048)
    predictor.load_state_dict(torch.load('/home/liushenyu/test/predictor_topk_0.3_iwl_epoch_20.pth',
        map_location='cpu', weights_only=True), strict=True)
    predictor.cuda().eval()
    embedding = torch.load('/home/liushenyu/test/opt_13b_embed_weights.pth',
        map_location='cpu', weights_only=True).half().cuda()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/home/liushenyu/opt-13b', use_fast=False, local_files_only=True)
    texts = ('The reader opened the old archive and searched the following page.',
             'A scientist measured the temperature before beginning a new experiment.',
             'The committee reviewed the report and discussed its main conclusion.',
             'Several students walked across campus after the lecture ended.')
    prompt = torch.stack([tokenizer(' '.join([s]*400), add_special_tokens=False,
        return_tensors='pt').input_ids[0, :1920] for s in texts]).cuda()
    forced = torch.randint(3, model.config.vocab_size, (4, 8), device='cuda')
    return compare(model, predictor, embedding, prompt, forced)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '3'
    assert len(os.sched_getaffinity(0)) <= 10
    torch.set_num_threads(10)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261029)
    report = real() if args.real else tiny()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(dict(passed=True, real=args.real, checks=report), stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
