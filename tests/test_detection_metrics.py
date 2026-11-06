from __future__ import annotations

import unittest

import numpy as np

from stegoguard_lm.detection.metrics import classification_metrics, threshold_for_empirical_fpr


class DetectionMetricTests(unittest.TestCase):
    def test_strict_fpr_threshold_when_one_fp_exceeds_target(self):
        clean = np.asarray([0.1, 0.2, 0.3, 0.4])
        threshold = threshold_for_empirical_fpr(clean, 0.01)
        self.assertGreater(threshold, clean.max())

    def test_basic_metrics(self):
        y = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        pred = (scores >= 0.5).astype(int)
        metrics = classification_metrics(
            y_true=y,
            scores=scores,
            pred_f1=pred,
            pred_fpr_05=pred,
            pred_fpr_01=pred,
        )
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
