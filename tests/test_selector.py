"""Unit tests for the rank-normalized token selector."""

import unittest

import numpy as np

from rcllm.selector import rank_norm_numpy, select_by_role_numpy, select_tokens_numpy


class TestSelector(unittest.TestCase):
    def test_rank_norm_tie_averaging(self):
        ranks = rank_norm_numpy([10.0, 20.0, 20.0, 30.0])
        np.testing.assert_allclose(ranks, [0.0, 0.5, 0.5, 1.0])

    def test_rank_norm_singleton(self):
        ranks = rank_norm_numpy([42.0])
        np.testing.assert_allclose(ranks, [0.0])

    def test_select_tokens_budget_and_protected(self):
        attn = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        kdev = np.array([0.0, 0.8, 0.1, 0.9, 0.2, 0.6, 0.3, 0.5])
        vdev = np.array([0.0, 0.7, 0.2, 0.8, 0.1, 0.9, 0.4, 0.5])
        eligible = [1, 2, 3, 4, 5, 6]
        protected = [0, 7]
        selected, scores = select_tokens_numpy(attn, kdev, vdev, eligible, protected, recompute_rate_pct=34)
        self.assertEqual(len(scores), len(eligible))
        # ceil(0.34 * 6) = 3 eligible tokens + 2 protected = 5
        self.assertEqual(len(selected), 5)
        self.assertTrue({0, 7}.issubset(set(selected.tolist())))
        # Highest combined rank tokens are positions 1, 3, 5.
        self.assertEqual(selected.tolist(), [0, 1, 3, 5, 7])

    def test_select_by_role_global_vs_role(self):
        vals = np.linspace(0.0, 1.0, 10)
        roles = {"review": [1, 2, 3, 4], "item": [5, 6, 7, 8]}
        protected = [0, 9]
        sel_g, kept_g, _ = select_by_role_numpy(vals, vals, vals, roles, protected, drop_fraction=0.5, global_budget=True)
        self.assertEqual(sum(kept_g.values()), 4)
        self.assertEqual(len(sel_g), 6)
        sel_r, kept_r, _ = select_by_role_numpy(vals, vals, vals, roles, protected, drop_fraction=0.5, global_budget=False)
        self.assertEqual(kept_r, {"review": 2, "item": 2})
        self.assertEqual(len(sel_r), 6)


if __name__ == "__main__":
    unittest.main()
