from __future__ import annotations

import unittest

from stegoguard_lm.attacks.synonym_channel import SynonymCodebook, SynonymSlot
from stegoguard_lm.mitigation.payload_eval import (
    bit_recovery_stats,
    evaluate_synonym_payload_after_mitigation,
    evaluate_unicode_payload_after_mitigation,
)


class MitigationPayloadEvalTests(unittest.TestCase):
    def test_bit_recovery_counts_missing_bits_as_not_recovered(self) -> None:
        result = bit_recovery_stats("0101", "01")
        self.assertEqual(result["correct_bit_count"], 2)
        self.assertAlmostEqual(result["effective_bit_accuracy"], 0.5)

    def test_unicode_decoder_failure_means_zero_effective_recovery(self) -> None:
        attack = {"payload": "0101"}
        cfg = {
            "bit_zero_codepoint": "\u200b",
            "bit_one_codepoint": "\u200c",
            "prefix_codepoint": "\u2063",
            "suffix_codepoint": "\u2064",
        }
        result = evaluate_unicode_payload_after_mitigation(
            original_attack=attack,
            mitigated_text="Visible text only.",
            attack_config=cfg,
        )
        self.assertFalse(result["decoder_success"])
        self.assertFalse(result["exact_payload_recovery"])
        self.assertEqual(result["effective_bit_accuracy"], 0.0)

    def test_synonym_partial_recovery_is_measured_per_active_slot(self) -> None:
        codebook = SynonymCodebook(
            version="v1",
            slots=(
                SynonymSlot("s1", "important", "crucial"),
                SynonymSlot("s2", "use", "apply"),
            ),
        )
        attack = {
            "payload": "01",
            "channel_parameters": {"active_slot_ids": ["s1", "s2"]},
        }
        result = evaluate_synonym_payload_after_mitigation(
            original_attack=attack,
            mitigated_text="This is important. Tools help.",
            codebook=codebook,
        )
        self.assertFalse(result["decoder_success"])
        self.assertEqual(result["correct_bit_count"], 1)
        self.assertAlmostEqual(result["effective_bit_accuracy"], 0.5)
        self.assertEqual(result["residual_capacity_bits"], 1)


if __name__ == "__main__":
    unittest.main()
