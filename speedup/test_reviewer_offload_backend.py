"""Correctness checks before using the new backend for any timing claims."""
import argparse
import json

import torch

from reviewer_offload_backend import OffloadEngine, load_opt, neural_indices, pad_indices


def selection_tests():
    scores = torch.tensor([[[4., 3., 3., 1., 0.]], [[0., 1., 2., 3., 4.]]])
    mask = neural_indices(scores, .4, 0)
    assert mask.sum(-1).tolist() == [[3], [2]], 'Threshold ties must be preserved'
    assert not torch.equal(mask[0], mask[1]), 'Selections must be batch specific'
    assert neural_indices(scores, .4, 1)[0].sum() == 4, 'Recent union missing'
    idx = pad_indices(mask, 7)
    assert idx[0].tolist() == [[0, 1, 2], [3, 4, -1]]
    assert torch.equal(idx[0], idx[-1])
    depth_mask = torch.eye(5, dtype=torch.bool)[None]
    depth_idx = pad_indices(depth_mask, 7)
    assert depth_idx[:, 0, 0].tolist() == [0, 1, 2, 3, 4, 4, 4]


@torch.inference_mode()
def cached_reference(model, ids, outputs):
    pref = model(ids, use_cache=True, return_dict=True)
    pre = pref.logits[:, -1].cpu()
    past = pref.past_key_values
    logits = []
    for step in range(outputs.shape[1]-1):
        out = model(outputs[:, step:step+1], past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        logits.append(out.logits[:, -1].cpu())
    return pre, torch.stack(logits, 1)


def gpu_tests():
    from transformers import OPTConfig
    from predictor_model2A import PredictorTransformer
    torch.manual_seed(1948)
    torch.set_num_threads(10)
    config = OPTConfig(hidden_size=64, word_embed_proj_dim=64, ffn_dim=128,
                       num_hidden_layers=7, num_attention_heads=4, vocab_size=211,
                       max_position_embeddings=128, dropout=0., attention_dropout=0.)
    model = load_opt().OPTForCausalLM(config).half().cuda().eval()
    predictor = PredictorTransformer(64, 5, nhead=4, num_encoder_layers=1,
                                     d_hid=128, dropout=0.).cuda().eval()
    embedding = torch.randn(211, 64, device='cuda')
    ids = torch.randint(3, 211, (2, 64), device='cuda')
    forced = torch.randint(3, 211, (2, 9), device='cuda')
    ref_pre, ref_logits = cached_reference(model, ids, forced)
    engine = OffloadEngine(model, predictor, embedding)
    report = {}
    try:
        for strategy in ('full_gpu', 'full_offload'):
            result, pre, trace = engine.generate(ids, 9, strategy, forced=forced)
            torch.testing.assert_close(pre, ref_pre, atol=.003, rtol=.01)
            torch.testing.assert_close(trace, ref_logits, atol=.003, rtol=.01)
            report[strategy+'_max_reference_error'] = float((trace-ref_logits).abs().max())
        serial, _, s_logits = engine.generate(ids, 9, 'neural_serial', forced=forced)
        async_result, _, a_logits = engine.generate(ids, 9, 'neural_async', forced=forced)
        assert [p['index_hash'] for p in serial['plans']] == [p['index_hash'] for p in async_result['plans']]
        torch.testing.assert_close(s_logits, a_logits, rtol=0, atol=0)
        assert sum(p['carry_entries'] for p in async_result['plans']) > 0
        assert sum(p['carry_entries'] for p in serial['plans']) == 0
        # Full budget checks every carried row against independently computed
        # native cached OPT, including all layers and both batch items.
        all_result, all_pre, all_trace = engine.generate(ids, 9, 'neural_async', ratio=1., recent=0, forced=forced)
        torch.testing.assert_close(all_pre, ref_pre, atol=.003, rtol=.01)
        torch.testing.assert_close(all_trace, ref_logits, atol=.003, rtol=.01)
        s_free, _, _ = engine.generate(ids, 9, 'neural_serial')
        a_free, _, _ = engine.generate(ids, 9, 'neural_async')
        assert s_free['output_ids'] == a_free['output_ids'], 'Free decoding diverges'
        report.update(serial_async_max_error=float((s_logits-a_logits).abs().max()),
                      all_history_async_max_reference_error=float((all_trace-ref_logits).abs().max()),
                      carry_entries=sum(p['carry_entries'] for p in async_result['plans']),
                      distinct_batch_items=True, free_decode_identical=True, passed=True)
    finally:
        engine.close()
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', action='store_true')
    args = p.parse_args()
    selection_tests()
    print(json.dumps(gpu_tests() if args.gpu else {'cpu_selection_checks': 'passed'}, indent=2))
