from __future__ import annotations

import unittest

from stegoguard_lm.detection.features import fold_cross_model_features, tail_fraction


class DetectionFeatureTests(unittest.TestCase):
    def test_tail_fraction(self):
        self.assertAlmostEqual(tail_fraction([-5, -1, -4, -2], -3), 0.5)

    def test_cross_model_uses_clean_normalized_differences(self):
        row_a = {"mean_logp": -2.0, "ppl": 8.0}
        row_b = {"mean_logp": -4.0, "ppl": 20.0}
        stats = {
            "a": {"mean_logp": (-3.0, 1.0), "ppl": (10.0, 2.0), "tail_05_frac": (0.1, 0.05)},
            "b": {"mean_logp": (-3.0, 1.0), "ppl": (10.0, 2.0), "tail_05_frac": (0.1, 0.05)},
        }
        features = fold_cross_model_features(
            row_a=row_a,
            row_b=row_b,
            tail_a=0.15,
            tail_b=0.05,
            clean_stats=stats,
        )
        self.assertAlmostEqual(features["delta_mean_logp"], 2.0)
        self.assertAlmostEqual(features["abs_delta_mean_logp"], 2.0)


if __name__ == "__main__":
    unittest.main()
