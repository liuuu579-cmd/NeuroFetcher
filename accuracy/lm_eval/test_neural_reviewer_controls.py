"""Small deterministic checks for selection semantics and neural equivalence."""

import unittest

import torch

from neural_reviewer_controls import load_model_module


module = load_model_module()


def attention(strategy="oracle", scope="shared", ratio=.3, recent=20):
    from transformers import OPTConfig
    config = OPTConfig(hidden_size=16, num_attention_heads=2,
                       num_hidden_layers=2, ffn_dim=32)
    config.control_strategy = strategy
    config.control_index_scope = scope
    config.control_recent_window = recent
    config.prefetch_budget_ratio = ratio
    return module.OPTAttention(16, 2, config=config, is_decoder=True).eval()


class ControlTests(unittest.TestCase):
    def test_reference_forward_is_identical(self):
        current = attention("full")
        original = module.OriginalOPTAttention(16, 2, config=current.config, is_decoder=True).eval()
        original.load_state_dict(current.state_dict())
        x = torch.randn(1, 35, 16)
        with torch.no_grad():
            for idx in [0, 1]:
                torch.testing.assert_close(current(x, output_attentions=True)[idx],
                                           original(x, output_attentions=True)[idx], rtol=0, atol=0)

    def test_neural_mask_is_identical(self):
        current = attention("neural")
        for logits in [torch.randn(2, 70), torch.zeros(2, 70)]:
            params = dict(predictor_logits=logits, dtype=torch.float32,
                          device=logits.device, bsz=2, tgt_len=70, src_len=70)
            expected = module.OriginalOPTAttention._create_causal_prefetch_mask(current, **params)
            torch.testing.assert_close(current._create_causal_prefetch_mask(**params), expected, rtol=0, atol=0)

    def test_full_budget_matches_neural_full_attention(self):
        oracle = attention(ratio=1.)
        original = module.OriginalOPTAttention(16, 2, config=oracle.config, is_decoder=True).eval()
        original.load_state_dict(oracle.state_dict())
        x = torch.randn(1, 35, 16)
        with torch.no_grad():
            for idx in [0, 1]:
                torch.testing.assert_close(oracle(x, output_attentions=True)[idx],
                                           original(x, output_attentions=True)[idx], rtol=0, atol=0)

    def test_known_topk_and_recency_union(self):
        current = attention(scope="per_head")
        # Equal head scores decline with position: all top-k are old tokens.
        scores = -torch.arange(140).float().view(1, 1, 1, 140).expand(1, 2, 140, 140)
        selected = current.control_mask(scores) == 0
        # torch uses fp32 for k, as the existing neural implementation does.
        k = int(torch.ceil(torch.tensor(140) * .3))
        expected = torch.zeros(140, dtype=torch.bool)
        expected[:k] = True
        expected[-20:] = True
        self.assertTrue(torch.equal(selected[0, 0, -1], expected))
        self.assertEqual(int(selected[0, 0, -1].sum()), k + 20)
        self.assertEqual(int(torch.triu(selected[0, 0], diagonal=1).sum()), 0)
        # Statistics accumulate over requests instead of retaining only the last.
        first = current.control_stats.clone()
        current.control_mask(scores)
        self.assertTrue(torch.equal(current.control_stats, 2 * first))

    def test_shared_uses_mean_probability_not_mean_raw_score(self):
        current = attention(ratio=.2, recent=0)
        scores = torch.randn(1, 2, 10, 10)
        selected = current.control_mask(scores) == 0
        valid = torch.ones(10, 10, dtype=torch.bool).tril()
        importance = scores.masked_fill(~valid, float("-inf")).softmax(-1).mean(1)
        for i in range(10):
            k = int(torch.ceil(torch.tensor(i + 1) * .2))
            expected = torch.zeros(10, dtype=torch.bool)
            expected[importance[0, i, :i+1].topk(k).indices] = True
            self.assertTrue(torch.equal(selected[0, 0, i], expected))

    def test_previous_query_ranking_no_recursive_reuse(self):
        current = attention("previous_step", "per_head", .2, 0)
        scores = torch.zeros(1, 2, 10, 10)
        for i in range(10):
            scores[:, :, i, :i+1] = torch.arange(i + 1).float()
            scores[:, :, i, i // 2] = 100
        selected = current.control_mask(scores) == 0
        for i in range(1, 10):
            k = min(i, int(torch.ceil(torch.tensor(i + 1) * .2)))
            expected = torch.zeros(10, dtype=torch.bool)
            expected[scores[0, 0, i-1, :i].topk(k).indices] = True
            self.assertTrue(torch.equal(selected[0, 0, i], expected))
        self.assertFalse(bool(selected[0, 0, 9, 9]))

    def test_ties_never_unmask_future(self):
        for scope in ["shared", "per_head"]:
            current = attention(scope=scope, ratio=.08)
            selected = current.control_mask(torch.zeros(1, 2, 50, 50)) == 0
            self.assertEqual(int(torch.triu(selected[0, 0], diagonal=1).sum()), 0)
            self.assertTrue(bool(selected[0, 0].diagonal().all()))

    def test_reject_cached_or_padded_control_requests(self):
        current = attention()
        with self.assertRaises(ValueError):
            current.control_mask(torch.zeros(1, 2, 1, 5))
        with self.assertRaises(ValueError):
            current.control_mask(torch.zeros(1, 2, 5, 5), torch.tensor([[1, 1, 0, 0, 0]]))


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.manual_seed(0)
    unittest.main()
