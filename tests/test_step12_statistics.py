from __future__ import annotations

import unittest

from stegoguard_lm.robustness.statistics import (
    benjamini_hochberg,
    paired_parent_values,
    paired_sign_flip_test,
    summarize_effects,
)


class Step12StatisticsTests(unittest.TestCase):
    def test_sign_flip_is_deterministic(self) -> None:
        a = paired_sign_flip_test([1, 2, -1, 3], replicates=2000, seed=9)
        b = paired_sign_flip_test([1, 2, -1, 3], replicates=2000, seed=9)
        self.assertEqual(a, b)
        self.assertGreaterEqual(a["p_value"], 0.0)
        self.assertLessEqual(a["p_value"], 1.0)

    def test_zero_differences_return_p_one(self) -> None:
        result = paired_sign_flip_test([0, 0, 0], replicates=100, seed=1)
        self.assertEqual(result["p_value"], 1.0)

    def test_bh_adjustment_is_monotone_and_bounded(self) -> None:
        q = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
        self.assertTrue(all(v is not None and 0 <= v <= 1 for v in q))
        ordered = sorted(zip([0.01, 0.04, 0.03, 0.20], q))
        self.assertTrue(all(ordered[i][1] <= ordered[i + 1][1] for i in range(len(ordered) - 1)))

    def test_parent_values_use_common_parents(self) -> None:
        test = [
            {"clean_parent_id": "a", "x": 2},
            {"clean_parent_id": "b", "x": 3},
        ]
        ref = [
            {"clean_parent_id": "a", "x": 1},
            {"clean_parent_id": "c", "x": 5},
        ]
        parents, diffs = paired_parent_values(test, ref, "x")
        self.assertEqual(parents, ["a"])
        self.assertEqual(diffs.tolist(), [1.0])

    def test_effect_summary_reports_sign_stability(self) -> None:
        result = summarize_effects([0.1, 0.2, 0.05], full_effect=0.12)
        self.assertEqual(result["sign_agreement_rate"], 1.0)
        self.assertAlmostEqual(result["min_effect"], 0.05)
        self.assertAlmostEqual(result["max_effect"], 0.2)


if __name__ == "__main__":
    unittest.main()
