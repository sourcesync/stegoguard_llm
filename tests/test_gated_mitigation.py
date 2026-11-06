from __future__ import annotations

import unittest

from stegoguard_lm.mitigation.gated import intervention_decision, selected_oracle_task_id


class GatedMitigationTests(unittest.TestCase):
    def test_unicode_hygiene_overrides_linguistic_detector(self) -> None:
        decision = intervention_decision(
            sample_id="x",
            prediction_row={"pred_f1": 0, "pred_fpr_01": 0},
            unicode_feature_row={"has_suspicious_codepoint": 1},
            operating_point="f1",
        )
        self.assertTrue(decision["intervene"])
        self.assertEqual(decision["route"], "unicode_sanitization")

    def test_unflagged_text_passes_through(self) -> None:
        decision = intervention_decision(
            sample_id="x",
            prediction_row={"pred_f1": 0, "pred_fpr_01": 0},
            unicode_feature_row={"has_suspicious_codepoint": 0},
            operating_point="fpr_01",
        )
        self.assertFalse(decision["intervene"])
        self.assertEqual(decision["route"], "pass_through")

    def test_randomized_task_id_uses_frozen_seed(self) -> None:
        task_id = selected_oracle_task_id(
            sample_id="s1",
            route="linguistic_rewrite",
            rewrite_policy="randomized_rewrite",
            random_seed=9102,
        )
        self.assertEqual(task_id, "randomized_rewrite::seed_9102::s1")


if __name__ == "__main__":
    unittest.main()
