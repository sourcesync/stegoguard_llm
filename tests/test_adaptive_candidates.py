import unittest

from stegoguard_lm.adaptive.candidates import (
    available_strategy_indices,
    candidate_index,
    copy_static_candidate,
)


class AdaptiveCandidateTests(unittest.TestCase):
    def test_copy_static_candidate_preserves_payload_and_marks_step10(self):
        spec = {
            "candidate_id": "adaptive_candidate::rank_encoding::clean_1::s00",
            "strategy_index": 0,
            "strategy_id": "rank_key_00",
            "strategy_attack_seed": 42,
            "synonym_mapping_mask": None,
            "static_step7_sample_id": "stego_rank_1",
            "candidate_spec_sha256": "abc",
        }
        static = {
            "sample_id": "stego_rank_1",
            "parent_sample_id": "clean_1",
            "attack_name": "rank_encoding",
            "payload": "101",
            "payload_bit_length": 3,
            "payload_sha256": "hash",
            "source_model": "model",
        }
        row = copy_static_candidate(spec=spec, static_attack=static, protocol_version="step10_adaptive_v1")
        self.assertEqual(row["payload"], "101")
        self.assertEqual(row["sample_id"], spec["candidate_id"])
        self.assertEqual(row["candidate_source"], "frozen_step7_strategy0")
        self.assertTrue(row["adaptive"])

    def test_candidate_index_rejects_duplicate_keys(self):
        row = {"parent_sample_id": "p", "attack_name": "rank_encoding", "strategy_index": 0}
        with self.assertRaises(ValueError):
            candidate_index([row, dict(row)])

    def test_available_strategy_indices_are_sorted_and_unique(self):
        rows = [
            {"parent_sample_id": "p", "attack_name": "rank_encoding", "strategy_index": 2},
            {"parent_sample_id": "p", "attack_name": "rank_encoding", "strategy_index": 0},
            {"parent_sample_id": "p", "attack_name": "rank_encoding", "strategy_index": 2},
            {"parent_sample_id": "p", "attack_name": "synonym_slot_codebook", "strategy_index": 1},
        ]
        self.assertEqual(available_strategy_indices(rows)["p"]["rank_encoding"], [0, 2])


if __name__ == "__main__":
    unittest.main()
