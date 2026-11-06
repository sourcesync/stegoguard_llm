from __future__ import annotations

import unittest

from stegoguard_lm.mitigation.tasks import (
    account_mitigation_tasks,
    build_mitigation_tasks,
    validate_mitigation_tasks,
)


class MitigationTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "metadata": {"manifest_version": "pilot_v0"},
            "models": {"evaluators": [{"id": "model/r", "role": "evaluator_a"}]},
            "mitigation_evaluation": {
                "rewriter_role": "evaluator_a",
                "expected_detection_records": 2,
                "expected_primary_records": 1,
                "randomized_rewrite": {"seeds": [11, 12, 13]},
            },
        }
        self.rows = [
            {
                "sample_id": "clean_a",
                "clean_parent_id": "clean_a",
                "pair_id": "a",
                "label": "clean",
                "attack_family": "clean",
                "attack_name": "clean",
                "source_model": "source/a",
                "domain": "conversational",
                "topic_id": "conv_001",
                "nominal_target_length": 64,
                "payload_bit_length": 0,
                "payload_rate_bpt": 0.0,
                "is_common_all_three": True,
                "is_balanced_common": False,
            },
            {
                "sample_id": "stego_b",
                "clean_parent_id": "clean_b",
                "pair_id": "b",
                "label": "stego",
                "attack_family": "token_whitebox",
                "attack_name": "rank_encoding",
                "source_model": "source/b",
                "domain": "technical",
                "topic_id": "tech_001",
                "nominal_target_length": 256,
                "payload_bit_length": 256,
                "payload_rate_bpt": 1.0,
                "is_common_all_three": False,
                "is_balanced_common": False,
            },
        ]

    def test_builds_five_tasks_per_detection_record(self) -> None:
        tasks = build_mitigation_tasks(config=self.config, detection_records=self.rows)
        self.assertEqual(len(tasks), 10)
        counts = {}
        for task in tasks:
            counts[task["condition"]] = counts.get(task["condition"], 0) + 1
        self.assertEqual(counts["unicode_sanitization"], 2)
        self.assertEqual(counts["deterministic_rewrite"], 2)
        self.assertEqual(counts["randomized_rewrite"], 6)

    def test_manifest_validation_checks_primary_count(self) -> None:
        tasks = build_mitigation_tasks(config=self.config, detection_records=self.rows)
        result = validate_mitigation_tasks(tasks=tasks, config=self.config, detection_records=self.rows)
        self.assertEqual(result["task_count"], 10)
        self.assertEqual(result["primary_task_count"], 5)

    def test_accounting_separates_pending_and_failures(self) -> None:
        tasks = build_mitigation_tasks(config=self.config, detection_records=self.rows)
        accepted = [{"task_id": tasks[0]["task_id"]}]
        failures = [{"task_id": tasks[1]["task_id"]}]
        result = account_mitigation_tasks(tasks=tasks, accepted_outputs=accepted, terminal_failures=failures)
        self.assertEqual(result["accepted_tasks"], 1)
        self.assertEqual(result["terminal_failures"], 1)
        self.assertEqual(result["pending_tasks"], 8)


if __name__ == "__main__":
    unittest.main()
