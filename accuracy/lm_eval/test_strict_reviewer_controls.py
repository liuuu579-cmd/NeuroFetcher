"""Deterministic causal-selection and integration checks for strict controls."""

import unittest

import torch

from strict_reviewer_controls import load_module


module = load_module()


def attention(strategy="oracle", ratio=.3, layer=0, state=None):
    from transformers import OPTConfig
    config = OPTConfig(hidden_size=16, num_attention_heads=2,
                       num_hidden_layers=7, ffn_dim=32)
    config.prefetch_strategy = "strict_control"
    config.strict_strategy = strategy
    config.strict_ratio = ratio
    config.strict_reuse_layer = 4
    config.quest_page_size = 16
    current = module.OPTAttention(16, 2, config=config, is_decoder=True,
                                  layer_idx=layer).eval()
    current.control_state = {} if state is None else state
    return current


class StrictControlTests(unittest.TestCase):
    def check_budget(self, mask, ratio=.3):
        length = mask.shape[-1]
        expected = module.budgets(length, ratio, mask.device).expand_as(mask.sum(-1))
        self.assertTrue(torch.equal(mask.sum(-1), expected))
        self.assertFalse(bool(torch.triu(mask, diagonal=1).any()))

    def test_exact_rational_rounding(self):
        k = module.budgets(100, .3, "cpu")
        self.assertEqual(k.tolist(), [max(1, 3 * i // 10) for i in range(1, 101)])

    def test_ties_exact_cardinality_without_recency(self):
        current = attention()
        scores = torch.zeros(2, 2, 140, 140)
        selected = current.selection(scores, None, None)
        self.check_budget(selected)
        self.assertEqual(selected[0, 0, -1].nonzero().flatten().tolist(), list(range(42)))
        self.assertFalse(bool(selected[..., -1, -20:].any()))
        first = current.control_stats.clone()
        current.selection(scores, None, None)
        self.assertTrue(torch.equal(current.control_stats, first * 2))
        self.assertEqual(int(first[1]), 2 * 2 * 140 * 141 // 2)
        self.assertEqual(int(first[3]), 0)

    def test_oracle_heads_have_independent_indices(self):
        scores = torch.zeros(1, 2, 40, 40)
        scores[:, 0] = -torch.arange(40).float()
        scores[:, 1] = torch.arange(40).float()
        selected = attention().selection(scores, None, None)
        self.check_budget(selected)
        self.assertEqual(selected[0, 0, -1].nonzero().flatten().tolist(), list(range(12)))
        self.assertEqual(selected[0, 1, -1].nonzero().flatten().tolist(), list(range(28, 40)))

    def test_previous_row_true_ranking_not_recursive(self):
        length = 40
        scores = torch.randn(1, 2, length, length)
        current = attention("previous_step")
        selected = current.selection(scores, None, None)
        self.check_budget(selected)
        for row in range(length):
            previous = max(0, row - 1)
            k = max(1, 3 * (row + 1) // 10)
            for head in range(2):
                expected = set(scores[0, head, previous, :previous + 1].topk(k).indices.tolist())
                actual = set(selected[0, head, row].nonzero().flatten().tolist())
                self.assertEqual(actual, expected)
        self.assertFalse(bool(selected[..., 1:, :].diagonal(offset=1, dim1=-2, dim2=-1).any()))

    def test_layer5_reuses_current_query_corresponding_heads(self):
        state = {}
        scores = torch.randn(1, 2, 40, 40)
        source = attention("layer5_reuse", layer=4, state=state)
        source_mask = source.selection(scores, None, None)
        for layer in [5, 6, 39]:
            later = attention("layer5_reuse", layer=layer, state=state)
            selected = later.selection(-scores, None, None)
            self.assertTrue(torch.equal(selected, source_mask))
            self.check_budget(selected)
        early = attention("layer5_reuse", layer=3, state=state)
        self.assertTrue(torch.equal(early.selection(-scores, None, None),
                                   attention().selection(-scores, None, None)))
        self.assertFalse(torch.equal(source_mask[:, 0], source_mask[:, 1]))

    def test_new_request_clears_shallow_state(self):
        state = {"shallow_mask": torch.ones(1)}
        first = attention("layer5_reuse", layer=0, state=state)
        first(torch.randn(1, 8, 16))
        self.assertEqual(state, {})
        with self.assertRaisesRegex(RuntimeError, "Missing matching"):
            attention("layer5_reuse", layer=6, state=state).selection(
                torch.zeros(1, 2, 8, 8), None, None)

    def test_quest_bounds_match_brute_force_visible_pages(self):
        q, key = torch.randn(2, 2, 35, 8), torch.randn(2, 2, 35, 8)
        bounds = module.quest_page_scores(q, key, 16)
        for row in range(35):
            for page in range(row // 16 + 1):
                visible = key[:, :, page * 16:min((page + 1) * 16, row + 1)]
                signed = visible * q[:, :, row:row + 1].sign()
                expected = (signed.amax(-2) * q[:, :, row].abs()).sum(-1)
                torch.testing.assert_close(bounds[:, :, row, page], expected)

    def test_future_keys_cannot_change_quest_prefix_selection(self):
        q, key = torch.randn(1, 2, 90, 8), torch.randn(1, 2, 90, 8)
        changed = key.clone()
        changed[:, :, 37:] = torch.randn_like(changed[:, :, 37:]) * 1000
        k = module.budgets(90, .3, "cpu")
        for native in [True, False]:
            first = module.quest_selection(q, key, k, 16, native)
            second = module.quest_selection(q, changed, k, 16, native)
            self.assertTrue(torch.equal(first[..., :37, :], second[..., :37, :]))

    def test_quest_native_minimum_pages_and_shallow_full(self):
        q, key = torch.ones(1, 2, 140, 8), torch.ones(1, 2, 140, 8)
        scores = q @ key.transpose(-1, -2)
        for layer in [0, 1]:
            mask = attention("quest_native", layer=layer).selection(scores, q, key)
            self.assertEqual(mask[0, 0, -1].sum().item(), 140)
        mask = attention("quest_native", layer=2).selection(scores, q, key)
        self.assertEqual(mask[0, 0, -1].sum().item(), 48)
        self.assertTrue(bool(mask[0, 0, 31, :32].all()))
        self.assertFalse(bool(torch.triu(mask, diagonal=1).any()))

    def test_quest_capped_exact_budget_every_layer(self):
        q, key = torch.randn(1, 2, 140, 8), torch.randn(1, 2, 140, 8)
        scores = q @ key.transpose(-1, -2)
        for layer in [0, 1, 5]:
            mask = attention("quest_capped", layer=layer).selection(scores, q, key)
            self.check_budget(mask)
        # Equal page bounds must still choose exactly 42, without recent tokens.
        q.zero_()
        mask = attention("quest_capped").selection(scores, q, key)
        self.assertEqual(mask[0, 0, -1].nonzero().flatten().tolist(), list(range(42)))

    def test_full_budget_matches_original_attention(self):
        current = attention(ratio=1.)
        original = module.OriginalOPTAttention(16, 2, config=current.config,
                                              is_decoder=True).eval()
        original.load_state_dict(current.state_dict())
        x = torch.randn(1, 35, 16)
        with torch.no_grad():
            actual = current(x, output_attentions=True)
            expected = original(x, output_attentions=True)
        for index in [0, 1]:
            torch.testing.assert_close(actual[index], expected[index], rtol=0, atol=0)

    def test_small_model_integration_and_no_predictor(self):
        from transformers import OPTConfig
        config = OPTConfig(vocab_size=30, hidden_size=16, word_embed_proj_dim=16,
                           num_attention_heads=2, num_hidden_layers=7, ffn_dim=32,
                           max_position_embeddings=64, dropout=0.)
        config.prefetch_strategy = "strict_control"
        config.strict_strategy = "layer5_reuse"
        model = module.OPTForCausalLM(config).eval()
        self.assertIsNone(model.predictor)
        state = {}
        for layer in model.model.decoder.layers:
            layer.self_attn.control_state = state
        with torch.no_grad():
            for length in [35, 17]:
                output = model(torch.arange(length).remainder(30)[None], use_cache=False)
                self.assertEqual(output.logits.shape, (1, length, 30))
                self.assertTrue(bool(output.logits.isfinite().all()))
                self.assertEqual(state["shallow_mask"].shape[-1], length)


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.manual_seed(0)
    unittest.main()
