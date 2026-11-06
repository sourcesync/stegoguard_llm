from __future__ import annotations

import unittest

from stegoguard_lm.attacks.base import AttackResult
from stegoguard_lm.attacks.full_generation import (
    account_task_inventory,
    attack_protocol_sha256,
    build_attack_task_manifest,
    build_full_attack_record,
    is_expected_terminal_attack_failure,
    validate_full_record_against_task,
)


class FullGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "metadata": {"manifest_version": "pilot_v0"},
            "attacks": {
                "payload": {
                    "payload_type": "bitstring",
                    "derivation": "sha256_deterministic",
                    "base_seed": 4206,
                },
                "out_of_band": {
                    "enabled": True,
                    "name": "unicode_hidden_channel",
                    "attacker_access": "text_transformation",
                    "payload_bits": 8,
                    "framing_version": "unicode_frame_v1",
                },
                "token_blackbox": {
                    "enabled": True,
                    "name": "synonym_slot_codebook",
                    "attacker_access": "black_box_tokenizer_aware",
                    "strategy_id": "synonym_inplace_substitution",
                    "strategy_version": "synonym_slots_v1+inplace_v1",
                    "payload_nonce_attempts": 16,
                },
                "token_whitebox": {
                    "enabled": True,
                    "name": "rank_encoding",
                    "attacker_access": "white_box",
                    "max_encoding_attempts": 5,
                },
            },
            "full_attack_generation": {
                "enabled": True,
                "expected_clean_parents": 2,
                "expected_attack_opportunities": 6,
                "synonym_payload_policy": "use_full_available_capacity",
                "retry_policy": {
                    "resume_completed": True,
                    "retry_terminal_failures": False,
                },
                "immediate_disk_write": True,
                "round_id": 0,
                "adaptive": False,
                "feedback_condition": "none",
            },
        }
        self.clean = [
            {
                "sample_id": "clean_a",
                "pair_id": "a",
                "lineage_id": "a",
                "source_model": "model/a",
                "domain": "conversational",
                "topic_id": "conv_001",
                "target_length": 64,
                "actual_token_length": 64,
                "prompt": "Prompt A",
                "text": "Clean A.",
            },
            {
                "sample_id": "clean_b",
                "pair_id": "b",
                "lineage_id": "b",
                "source_model": "model/b",
                "domain": "technical",
                "topic_id": "tech_001",
                "target_length": 256,
                "actual_token_length": 260,
                "prompt": "Prompt B",
                "text": "Clean B.",
            },
        ]
        self.audit = {
            "matrix": "full",
            "record_count": 2,
            "capacity_distribution": {"0": 1, "2": 1},
            "records_with_capacity": 1,
            "records_without_capacity": 1,
            "rows": [
                {
                    "sample_id": "clean_a",
                    "source_model": "model/a",
                    "domain": "conversational",
                    "nominal_target_length": 64,
                    "actual_token_length": 64,
                    "eligible_slot_ids": ["slot_01", "slot_03"],
                    "capacity_bits": 2,
                    "has_capacity": True,
                },
                {
                    "sample_id": "clean_b",
                    "source_model": "model/b",
                    "domain": "technical",
                    "nominal_target_length": 256,
                    "actual_token_length": 260,
                    "eligible_slot_ids": [],
                    "capacity_bits": 0,
                    "has_capacity": False,
                },
            ],
        }

    def _tasks(self):
        return build_attack_task_manifest(
            config=self.config,
            clean_records=self.clean,
            capacity_audit=self.audit,
        )

    def test_manifest_represents_full_opportunity_space(self) -> None:
        tasks, structural = self._tasks()
        self.assertEqual(len(tasks), 6)
        self.assertEqual(len(structural), 1)
        self.assertEqual(sum(bool(task["eligible"]) for task in tasks), 5)
        self.assertEqual(
            {task["attack_family"] for task in tasks},
            {"out_of_band", "token_blackbox", "token_whitebox"},
        )

    def test_zero_capacity_is_structural_not_failure(self) -> None:
        tasks, structural = self._tasks()
        self.assertEqual(structural[0]["parent_sample_id"], "clean_b")
        self.assertEqual(structural[0]["attack_family"], "token_blackbox")
        self.assertEqual(structural[0]["eligibility_status"], "structural_zero_capacity")
        self.assertEqual(structural[0]["available_capacity_bits"], 0)
        self.assertEqual(structural[0]["expected_payload_bits"], 0)

    def test_payload_policies_are_frozen_per_channel(self) -> None:
        tasks, _ = self._tasks()
        by_family = {
            task["attack_family"]: task
            for task in tasks
            if task["parent_sample_id"] == "clean_a"
        }
        self.assertEqual(by_family["token_whitebox"]["expected_payload_bits"], 64)
        self.assertEqual(by_family["out_of_band"]["expected_payload_bits"], 8)
        self.assertEqual(by_family["token_blackbox"]["expected_payload_bits"], 2)
        self.assertEqual(
            by_family["token_blackbox"]["synonym_eligible_slot_ids"],
            ["slot_01", "slot_03"],
        )

    def test_inventory_keeps_pending_separate_from_structural(self) -> None:
        tasks, _ = self._tasks()
        accounting = account_task_inventory(
            tasks=tasks,
            accepted_records=[],
            terminal_failures=[],
        )
        self.assertEqual(accounting["structural_ineligible_tasks"], 1)
        self.assertEqual(accounting["eligible_tasks"], 5)
        self.assertEqual(accounting["pending_tasks"], 5)
        self.assertEqual(accounting["status"], "incomplete")
        self.assertEqual(accounting["accounting_total"], 6)

    def test_expected_terminal_failure_prefixes_only(self) -> None:
        tasks, _ = self._tasks()
        rank = next(task for task in tasks if task["attack_family"] == "token_whitebox")
        synonym = next(
            task
            for task in tasks
            if task["attack_family"] == "token_blackbox" and task["eligible"]
        )
        unicode = next(task for task in tasks if task["attack_family"] == "out_of_band")
        self.assertTrue(
            is_expected_terminal_attack_failure(
                rank,
                RuntimeError("Rank encoding failed after 5 nonces. Last error: x"),
            )
        )
        self.assertTrue(
            is_expected_terminal_attack_failure(
                synonym,
                RuntimeError(
                    "Synonym full-capacity payload selection failed after 16 deterministic nonce attempts. Last error: x"
                ),
            )
        )
        self.assertFalse(
            is_expected_terminal_attack_failure(
                rank,
                RuntimeError("CUDA out of memory"),
            )
        )
        self.assertFalse(
            is_expected_terminal_attack_failure(
                rank,
                RuntimeError(
                    "Rank encoding failed after 5 nonces. Last error: CUDA out of memory"
                ),
            )
        )
        self.assertFalse(
            is_expected_terminal_attack_failure(
                unicode,
                RuntimeError("Unicode unexpected failure"),
            )
        )

    def test_full_record_adds_task_and_payload_rate_metadata(self) -> None:
        tasks, _ = self._tasks()
        task = next(
            task
            for task in tasks
            if task["parent_sample_id"] == "clean_a"
            and task["attack_family"] == "token_blackbox"
        )
        parent = self.clean[0]
        result = AttackResult(
            attack_family="token_blackbox",
            attack_name="synonym_slot_codebook",
            attacker_access="black_box_tokenizer_aware",
            strategy_id="synonym_inplace_substitution",
            strategy_version="synonym_slots_v1+inplace_v1",
            stego_text="Clean changed.",
            payload="01",
            decoded_payload="01",
            decode_success=True,
            exact_match=True,
            capacity_bits=2,
            payload_bit_length=2,
            encoding_attempts=1,
            source_token_length=64,
            length_invariant="source_tokenizer_exact",
            length_valid=True,
            channel_parameters={"active_slot_ids": ["slot_01", "slot_03"]},
        )
        record = build_full_attack_record(
            config=self.config,
            task=task,
            parent=parent,
            result=result,
        )
        self.assertEqual(record["task_id"], task["task_id"])
        self.assertEqual(record["available_capacity_bits"], 2)
        self.assertEqual(record["expected_payload_bits"], 2)
        self.assertAlmostEqual(record["payload_rate_bpt"], 2 / 64)
        errors = validate_full_record_against_task(
            record=record,
            task=task,
            parent=parent,
        )
        self.assertEqual(errors, [])

    def test_protocol_hash_is_sensitive_to_frozen_attack_settings(self) -> None:
        first = attack_protocol_sha256(self.config)
        changed = {
            **self.config,
            "attacks": {
                **self.config["attacks"],
                "token_whitebox": {
                    **self.config["attacks"]["token_whitebox"],
                    "max_encoding_attempts": 6,
                },
            },
        }
        second = attack_protocol_sha256(changed)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
