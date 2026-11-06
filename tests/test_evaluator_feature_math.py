from __future__ import annotations

import math
import unittest

from stegoguard_lm.detection.evaluator_scoring import sentence_spans, summarize_token_logprobs


class EvaluatorFeatureMathTests(unittest.TestCase):
    def test_sentence_spans(self):
        spans = sentence_spans("First sentence. Second one!")
        self.assertEqual(len(spans), 2)

    def test_distributional_summary(self):
        text = "A B. C D."
        logps = [-1.0, -1.2, -3.0, -3.2]
        offsets = [(0, 1), (2, 4), (5, 6), (7, 9)]
        result = summarize_token_logprobs(
            text=text,
            token_logprobs=logps,
            token_offsets=offsets,
        )
        self.assertEqual(result["response_token_count"], 4)
        self.assertAlmostEqual(result["mean_logp"], -2.1)
        self.assertTrue(math.isfinite(result["ppl"]))
        self.assertGreater(result["logp_var"], 0)
        self.assertGreaterEqual(result["sentence_logp_var"], 0)


if __name__ == "__main__":
    unittest.main()
