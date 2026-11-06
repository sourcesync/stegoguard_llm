import unittest

from stegoguard_lm.attacks.payloads import (
    derive_deterministic_int,
    derive_payload_bits,
    payload_sha256,
    validate_bitstring,
)


class PayloadTests(unittest.TestCase):
    def test_payload_is_deterministic(self):
        kwargs = {
            "manifest_version": "pilot_v0",
            "parent_sample_id": "clean_conv_001_64_qwen_000",
            "attack_name": "unicode_hidden_channel",
            "base_seed": 4206,
            "num_bits": 32,
        }
        first = derive_payload_bits(**kwargs)
        second = derive_payload_bits(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)
        validate_bitstring(first)

    def test_context_changes_payload(self):
        first = derive_payload_bits(
            manifest_version="pilot_v0",
            parent_sample_id="a",
            attack_name="rank_encoding",
            base_seed=1,
            num_bits=32,
        )
        second = derive_payload_bits(
            manifest_version="pilot_v0",
            parent_sample_id="b",
            attack_name="rank_encoding",
            base_seed=1,
            num_bits=32,
        )
        self.assertNotEqual(first, second)

    def test_invalid_bitstring(self):
        with self.assertRaises(ValueError):
            validate_bitstring("")
        with self.assertRaises(ValueError):
            validate_bitstring("0102")

    def test_hash_and_integer_are_stable(self):
        self.assertEqual(payload_sha256("0101"), payload_sha256("0101"))
        kwargs = {
            "namespace": "rank",
            "parent_sample_id": "clean_1",
            "base_seed": 42,
        }
        self.assertEqual(
            derive_deterministic_int(**kwargs),
            derive_deterministic_int(**kwargs),
        )


if __name__ == "__main__":
    unittest.main()
