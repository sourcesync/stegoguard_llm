from __future__ import annotations

import unittest

from stegoguard_lm.analysis.system import compute_round_metrics, compute_trajectory_metrics, split_roles


class Step11SystemMetricTests(unittest.TestCase):
    def _config(self):
        return {
            "step11_system_analysis": {
                "round_ids": [0, 1, 2, 3],
                "attacker_conditions": ["frozen_no_feedback"],
                "defender_policies": ["deterministic_rewrite"],
            }
        }

    def _rows(self):
        rows = []
        for round_id, bits in enumerate([0, 2, 2, 1]):
            rows.append({
                "record_role": "stego_trajectory",
                "trajectory_id": "t1",
                "clean_parent_id": "p1",
                "attacker_condition": "frozen_no_feedback",
                "defender_policy": "deterministic_rewrite",
                "initial_attack_name": "rank_encoding",
                "attack_name": "rank_encoding",
                "source_model": "m",
                "domain": "d",
                "nominal_target_length": 64,
                "round_id": round_id,
                "successful_covert_bits": bits,
                "exact_payload_recovery": bits > 0,
                "effective_bit_accuracy": 0.5,
                "residual_capacity_bits": 1,
                "detected": True,
                "strategy_changed_from_prior_round": round_id > 0,
                "channel_migrated_from_prior_round": False,
            })
        rows.append({"record_role": "clean_control"})
        return rows

    def test_split_roles(self):
        stego, clean = split_roles(self._rows())
        self.assertEqual(len(stego), 4)
        self.assertEqual(len(clean), 1)

    def test_round_metrics_preserve_successful_bits(self):
        stego, _ = split_roles(self._rows())
        rows = compute_round_metrics(stego, self._config())
        self.assertEqual(len(rows), 4)
        self.assertEqual([r["successful_covert_bits_total"] for r in rows], [0, 2, 2, 1])

    def test_trajectory_metrics_compute_cumulative_and_recovery(self):
        stego, _ = split_roles(self._rows())
        rows = compute_trajectory_metrics(stego, self._config())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["cumulative_successful_bits"], 5)
        self.assertEqual(rows[0]["r3_minus_r0_bits"], 1)
        self.assertEqual(rows[0]["r3_minus_r1_bits"], -1)


if __name__ == "__main__":
    unittest.main()
