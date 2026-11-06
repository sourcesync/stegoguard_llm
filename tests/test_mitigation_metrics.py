from __future__ import annotations

import unittest

from stegoguard_lm.mitigation.metrics import summarize_payload_results, summarize_randomized_survival


class MitigationMetricTests(unittest.TestCase):
    def test_payload_summary_reports_disruption(self) -> None:
        rows = [
            {
                "condition": "deterministic_rewrite",
                "attack_name": "rank_encoding",
                "clean_parent_id": "a",
                "exact_payload_recovery": False,
                "decoder_success": False,
                "effective_bit_accuracy": 0.0,
                "residual_capacity_fraction": 0.0,
            },
            {
                "condition": "deterministic_rewrite",
                "attack_name": "rank_encoding",
                "clean_parent_id": "b",
                "exact_payload_recovery": True,
                "decoder_success": True,
                "effective_bit_accuracy": 1.0,
                "residual_capacity_fraction": 1.0,
            },
        ]
        summary = summarize_payload_results(rows, bootstrap_replicates=10, seed=42)
        block = summary["deterministic_rewrite::rank_encoding"]
        self.assertAlmostEqual(block["exact_payload_recovery_rate"], 0.5)
        self.assertAlmostEqual(block["payload_disruption_rate"], 0.5)

    def test_randomized_survival_is_aggregated_by_original_sample(self) -> None:
        rows = [
            {"condition": "randomized_rewrite", "original_sample_id": "s", "attack_name": "synonym_slot_codebook", "exact_payload_recovery": True, "effective_bit_accuracy": 1.0},
            {"condition": "randomized_rewrite", "original_sample_id": "s", "attack_name": "synonym_slot_codebook", "exact_payload_recovery": False, "effective_bit_accuracy": 0.0},
            {"condition": "randomized_rewrite", "original_sample_id": "s", "attack_name": "synonym_slot_codebook", "exact_payload_recovery": False, "effective_bit_accuracy": 0.5},
        ]
        result = summarize_randomized_survival(rows)
        sample = result["per_sample"][0]
        self.assertAlmostEqual(sample["payload_survival_frequency"], 1 / 3)
        self.assertTrue(sample["any_seed_survival"])


if __name__ == "__main__":
    unittest.main()
