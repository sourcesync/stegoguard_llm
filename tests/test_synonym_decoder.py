import tempfile
import unittest
from pathlib import Path

from stegoguard_lm.attacks.synonym_channel import (
    SynonymCodebook,
    SynonymSlot,
    decode_synonym_payload,
    load_synonym_codebook,
    select_synonym_alternatives,
    validate_synonym_codebook,
)


class SynonymDecoderTests(unittest.TestCase):
    def setUp(self):
        self.codebook = SynonymCodebook(
            version="test_v1",
            slots=(
                SynonymSlot("s1", "important", "crucial"),
                SynonymSlot("s2", "begin", "start"),
            ),
        )

    def test_exact_recovery(self):
        result = decode_synonym_payload(
            "It is crucial to begin carefully.",
            self.codebook,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.payload, "10")

    def test_case_and_punctuation(self):
        result = decode_synonym_payload(
            "IMPORTANT, then START.",
            self.codebook,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.payload, "01")

    def test_substrings_do_not_count(self):
        result = decode_synonym_payload(
            "This is importantly stated before restarting.",
            self.codebook,
        )
        self.assertFalse(result.success)
        self.assertTrue(any("neither" in error for error in result.errors))

    def test_missing_ambiguous_and_repeated_slots(self):
        missing = decode_synonym_payload("It is important.", self.codebook)
        self.assertFalse(missing.success)

        ambiguous = decode_synonym_payload(
            "It is important and crucial to begin.", self.codebook
        )
        self.assertFalse(ambiguous.success)

        repeated = decode_synonym_payload(
            "Important work is important; then begin.", self.codebook
        )
        self.assertFalse(repeated.success)

    def test_active_subset_decode(self):
        result = decode_synonym_payload(
            "It is crucial to begin carefully.",
            self.codebook,
            active_slot_ids=["s1"],
        )
        self.assertTrue(result.success)
        self.assertEqual(result.payload, "1")
        self.assertEqual(len(result.slot_results), 1)

    def test_unknown_active_slot_is_rejected(self):
        result = decode_synonym_payload(
            "It is crucial to begin carefully.",
            self.codebook,
            active_slot_ids=["missing"],
        )
        self.assertFalse(result.success)
        self.assertTrue(any("Unknown" in error for error in result.errors))

    def test_payload_selects_expected_words(self):
        selected, excluded = select_synonym_alternatives("10", self.codebook)
        self.assertEqual(selected, ["crucial", "begin"])
        self.assertEqual(excluded, ["important", "start"])

    def test_codebook_validation(self):
        invalid = SynonymCodebook(
            version="x",
            slots=(
                SynonymSlot("same", "use", "apply"),
                SynonymSlot("same", "use", "employ"),
            ),
        )
        self.assertTrue(validate_synonym_codebook(invalid))

    def test_yaml_loader(self):
        content = """
version: test
slots:
  - slot_id: one
    zero: alpha
    one: beta
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "codebook.yaml"
            path.write_text(content, encoding="utf-8")
            loaded = load_synonym_codebook(path)
        self.assertEqual(loaded.version, "test")
        self.assertEqual(len(loaded.slots), 1)


if __name__ == "__main__":
    unittest.main()
