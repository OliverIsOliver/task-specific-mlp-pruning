import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import Gemma2Config, Gemma2ForCausalLM

from src.evaluation import correct
from src.fisher import fisher_scores, global_mask, score
from src.masks import LearnedMask, learn
from src.model import prune_mlp


class PruningTests(unittest.TestCase):
    def test_fisher_gate_sensitivity(self):
        activation = torch.tensor([[[1., 2., 3.], [2., 3., 4.]]], requires_grad=True)
        loss = (activation * torch.tensor([1., 2., -1.])).sum()
        np.testing.assert_allclose(fisher_scores(loss, [activation]), [[4.5, 50., 24.5]])

    def test_global_budget_can_be_uneven_across_layers(self):
        scores = np.array([[1., 2., 3., 4.], [10., 20., 30., 40.]])
        mask = global_mask(scores, .25)
        self.assertEqual(mask.sum(axis=1).tolist(), [2, 0])

    def test_grading_rules(self):
        self.assertTrue(correct('food', '[{"food":"tea","quantity":2},{"food":"cake","quantity":1}]',
                                '[{"food":"cake","quantity":1},{"food":"tea","quantity":2}]'))
        self.assertTrue(correct('addition', '12', '12\n'))
        self.assertFalse(correct('addition', '12', 'The answer is 12'))
        self.assertTrue(correct('appointments', '{"day":"Monday"}', '{"day":"Monday"}'))
        self.assertFalse(correct('appointments', '{"day":"Monday"}', '{"day":"Monday","time":"9"}'))
        self.assertFalse(correct('returns', '{"eligible":true}', '{"eligible":1}'))
        self.assertFalse(correct('returns', '{"eligible":true}', 'not JSON'))

    def test_learning_and_physical_pruning(self):
        # A tiny random Gemma exercises the real architecture without downloading weights.
        torch.manual_seed(17)
        config = Gemma2Config(vocab_size=32, hidden_size=8, intermediate_size=12,
                              num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                              head_dim=4, max_position_embeddings=32, attn_implementation='eager')
        model = Gemma2ForCausalLM(config).eval().requires_grad_(False)
        original = {name: value.clone() for name, value in model.state_dict().items()}

        class Tokenizer:
            def __call__(self, text, **kwargs):
                return SimpleNamespace(input_ids=[1] + [2 + ord(char) % 29 for char in text])

        rows = [dict(id=f'train-{i}', prompt='ab', answer='c') for i in range(4)]
        model.enable_input_require_grads()
        fisher, _ = score(rows[0], Tokenizer(), model, 'cpu')
        model.disable_input_require_grads()
        self.assertEqual(fisher.shape, (2, 12))
        self.assertTrue(np.isfinite(fisher).all())
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            removed = learn(rows, Tokenizer(), model, 'cpu', fisher, global_mask(fisher, .25),
                            Path(folder), 2, 2, .01, .25)
            trace = json.loads((Path(folder) / 'trace.json').read_text())
            self.assertTrue(all(step['muted_neurons_with_nonzero_gradient'] > 0 for step in trace))
        self.assertEqual(int(removed.sum()), 6)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        compact = copy.deepcopy(model)
        gates = LearnedMask(model, fisher, .25)
        gates.apply(removed)
        ids = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            expected = model(ids).logits
        gates.close()
        prune_mlp(compact, removed, 'cpu')
        with torch.no_grad():
            actual = compact(ids).logits
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        self.assertEqual(sum(block.mlp.down_proj.in_features for block in compact.model.layers), 18)
        self.assertFalse(any(block.mlp.down_proj._forward_pre_hooks for block in model.model.layers))


if __name__ == '__main__':
    unittest.main()
