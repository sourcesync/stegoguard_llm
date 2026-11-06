from __future__ import annotations

import unittest

import numpy as np

from stegoguard_lm.detection.semantic import binary_logprob_to_probability, cosine_similarity


class SemanticDetectionFeatureTests(unittest.TestCase):
    def test_binary_probability(self):
        self.assertGreater(binary_logprob_to_probability(-1.0, -4.0), 0.9)
        self.assertLess(binary_logprob_to_probability(-4.0, -1.0), 0.1)

    def test_cosine_similarity(self):
        self.assertAlmostEqual(cosine_similarity(np.array([1, 0]), np.array([1, 0])), 1.0)
        self.assertAlmostEqual(cosine_similarity(np.array([1, 0]), np.array([0, 1])), 0.0)


if __name__ == "__main__":
    unittest.main()
