import unittest

from stegoguard_lm.attacks.base import AttackResult
from stegoguard_lm.attacks.records import build_attack_record
from stegoguard_lm.attacks.validation import validate_attack_record_against_parent


class RecordTests(unittest.TestCase):
    def test_token_record_relations(self):
        parent = {
            "sample_id": "clean_conv_001_64_qwen_000",
            "pair_id": "conv_001_64_qwen_000",
            "lineage_id": "conv_001_64_qwen_000",
            "domain": "conversational",
            "topic_id": "conv_001",
            "topic_seed": "topic",
            "target_length": 64,
            "actual_token_length": 60,
            "source_model": "Qwen/test",
            "model_revision": "rev",
            "tokenizer_revision": None,
            "prompt": "Prompt",
            "prompt_id": "prompt_1",
        }
        result = AttackResult(
            attack_family="token_blackbox",
            attack_name="synonym_slot_codebook",
            attacker_access="black_box",
            strategy_id="synonym_slot_codebook",
            strategy_version="v1",
            stego_text="Complete.",
            payload="01",
            decoded_payload="01",
            decode_success=True,
            exact_match=True,
            capacity_bits=2,
            payload_bit_length=2,
            encoding_attempts=1,
            source_token_length=60,
            length_invariant="source_tokenizer_exact",
            length_valid=True,
            channel_parameters={},
        )
        record = build_attack_record(
            parent_record=parent,
            result=result,
            manifest_version="pilot_v0",
            attack_seed=42,
        )
        self.assertEqual(
            validate_attack_record_against_parent(record, parent),
            [],
        )
        self.assertEqual(record["parent_actual_token_length"], 60)
        self.assertEqual(record["stego_actual_token_length"], 60)


if __name__ == "__main__":
    unittest.main()
