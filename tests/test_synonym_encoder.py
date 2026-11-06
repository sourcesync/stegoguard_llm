import re
import unittest

from stegoguard_lm.attacks.synonym_channel import (
    SynonymCodebook,
    SynonymSlot,
    _has_forbidden_format_codepoints,
    decode_synonym_payload,
    discover_inplace_synonym_slots,
    encode_synonym_payload,
)


class WordTokenizer:
    """Small reversible tokenizer for model-free lexical-channel tests."""

    def __init__(self):
        words = [
            "Managing",
            "time",
            "is",
            "crucial",
            "important",
            "Use",
            "Apply",
            "a",
            "calendar",
            "to",
            "start",
            "begin",
            "work",
            "today",
            ".",
        ]
        self.vocab = {word: i + 3 for i, word in enumerate(words)}
        # Lower-case alternatives used when case preservation is not needed.
        for word in ["use", "apply"]:
            self.vocab.setdefault(word, len(self.vocab) + 3)
        self.reverse = {value: key for key, value in self.vocab.items()}
        self.all_special_ids = [0, 1, 2]

    def _tokens(self, text):
        return re.findall(r"\w+|[^\w\s]", text)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [self.vocab[token] for token in self._tokens(text)]

    def decode(
        self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ):
        del clean_up_tokenization_spaces
        tokens = []
        for token_id in token_ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id in self.all_special_ids:
                continue
            tokens.append(self.reverse[token_id])
        text = " ".join(tokens)
        return text.replace(" .", ".")


class SynonymEncoderIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.codebook = SynonymCodebook(
            version="test_v1",
            slots=(
                SynonymSlot("s1", "important", "crucial"),
                SynonymSlot("s2", "begin", "start"),
                SynonymSlot("s3", "use", "apply"),
            ),
        )
        self.tokenizer = WordTokenizer()
        self.clean = "Managing time is crucial. Use a calendar to start work today."
        self.length = len(self.tokenizer.encode(self.clean))

    def test_discovers_bidirectionally_stable_slots(self):
        eligible = discover_inplace_synonym_slots(
            clean_text=self.clean,
            tokenizer=self.tokenizer,
            codebook=self.codebook,
            required_token_length=self.length,
        )
        self.assertEqual([slot.slot_id for slot in eligible], ["s1", "s3", "s2"])
        self.assertEqual([slot.original_bit for slot in eligible], ["1", "0", "1"])

    def test_inplace_round_trip_preserves_exact_length_and_ending(self):
        result = encode_synonym_payload(
            clean_text=self.clean,
            tokenizer=self.tokenizer,
            payload="010",
            codebook=self.codebook,
            required_token_length=self.length,
            max_payload_bits=3,
            min_payload_bits=1,
            require_text_change=True,
        )
        self.assertNotEqual(result.stego_text, self.clean)
        self.assertEqual(result.source_token_length, self.length)
        self.assertEqual(result.payload, "010")
        self.assertEqual(result.decoded_payload, "010")
        self.assertEqual(result.strategy_id, "synonym_inplace_substitution")
        self.assertEqual(
            result.channel_parameters["active_slot_ids"],
            ["s1", "s3", "s2"],
        )
        decoded = decode_synonym_payload(
            result.stego_text,
            self.codebook,
            active_slot_ids=result.channel_parameters["active_slot_ids"],
        )
        self.assertTrue(decoded.success)
        self.assertEqual(decoded.payload, "010")

    def test_payload_is_truncated_to_available_capacity(self):
        codebook = SynonymCodebook(
            version="one_slot",
            slots=(SynonymSlot("s1", "important", "crucial"),),
        )
        result = encode_synonym_payload(
            clean_text=self.clean,
            tokenizer=self.tokenizer,
            payload="00000000",
            codebook=codebook,
            required_token_length=self.length,
            max_payload_bits=8,
            min_payload_bits=1,
            require_text_change=True,
        )
        self.assertEqual(result.payload, "0")
        self.assertEqual(result.payload_bit_length, 1)
        self.assertEqual(result.capacity_bits, 1)

    def test_no_capacity_is_reported(self):
        codebook = SynonymCodebook(
            version="none",
            slots=(SynonymSlot("x", "help", "assist"),),
        )
        with self.assertRaisesRegex(RuntimeError, "no_capacity"):
            encode_synonym_payload(
                clean_text=self.clean,
                tokenizer=self.tokenizer,
                payload="0",
                codebook=codebook,
                required_token_length=self.length,
            )

    def test_no_text_change_is_rejected_for_same_orientation_payload(self):
        # Original s1 orientation is 1 (crucial). A one-bit payload of 1 leaves
        # the cover unchanged and should be rejected so the runner can derive a
        # deterministic nonce payload instead.
        codebook = SynonymCodebook(
            version="one_slot",
            slots=(SynonymSlot("s1", "important", "crucial"),),
        )
        with self.assertRaisesRegex(RuntimeError, "no_text_change"):
            encode_synonym_payload(
                clean_text=self.clean,
                tokenizer=self.tokenizer,
                payload="1",
                codebook=codebook,
                required_token_length=self.length,
                require_text_change=True,
            )

    def test_hidden_unicode_format_character_is_detected(self):
        self.assertTrue(_has_forbidden_format_codepoints("visible\u200bhidden"))
        self.assertFalse(_has_forbidden_format_codepoints("ordinary visible text"))


if __name__ == "__main__":
    unittest.main()
