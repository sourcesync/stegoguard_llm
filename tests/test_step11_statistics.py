from __future__ import annotations

import unittest

from stegoguard_lm.analysis.statistics import bootstrap_mean_ci, paired_parent_difference, parent_means


class Step11StatisticsTests(unittest.TestCase):
    def test_parent_means_clusters_rows(self):
        rows = [
            {"clean_parent_id": "a", "x": 1},
            {"clean_parent_id": "a", "x": 3},
            {"clean_parent_id": "b", "x": 2},
        ]
        self.assertEqual(parent_means(rows, "x"), {"a": 2.0, "b": 2.0})

    def test_bootstrap_mean_ci_is_deterministic(self):
        first = bootstrap_mean_ci([1, 2, 3, 4], replicates=200, seed=7)
        second = bootstrap_mean_ci([1, 2, 3, 4], replicates=200, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean"], 2.5)

    def test_paired_parent_difference_uses_common_parents(self):
        test = [
            {"clean_parent_id": "a", "x": 3},
            {"clean_parent_id": "b", "x": 5},
        ]
        ref = [
            {"clean_parent_id": "a", "x": 1},
            {"clean_parent_id": "b", "x": 2},
        ]
        result = paired_parent_difference(test, ref, "x", replicates=200, seed=11)
        self.assertEqual(result["n_pairs"], 2)
        self.assertAlmostEqual(result["difference"], 2.5)


if __name__ == "__main__":
    unittest.main()
