import unittest

from stegoguard_lm.attacks.unicode_channel import (
    DEFAULT_PREFIX_CODEPOINT,
    DEFAULT_SUFFIX_CODEPOINT,
    decode_unicode_payload,
    encode_unicode_payload,
    strip_unicode_payload,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]


class UnicodeChannelTests(unittest.TestCase):
    def test_round_trip_and_visible_text_preservation(self):
        clean = "This is a complete response."
        payload = "01011010"
        result = encode_unicode_payload(
            clean_text=clean,
            payload=payload,
            tokenizer=CharacterTokenizer(),
        )
        decoded = decode_unicode_payload(result.stego_text)
        self.assertEqual(decoded.payload, payload)
        self.assertEqual(decoded.stripped_text, clean)
        self.assertEqual(strip_unicode_payload(result.stego_text), clean)
        self.assertTrue(result.exact_match)
        self.assertTrue(result.length_valid)
        self.assertFalse(result.channel_parameters["same_raw_token_length"])

    def test_supported_visible_endings(self):
        for text in ["Done.", "Done!", "Done?", 'Done.")']:
            with self.subTest(text=text):
                encoded = encode_unicode_payload(
                    clean_text=text,
                    payload="01",
                )
                self.assertEqual(strip_unicode_payload(encoded.stego_text), text)

    def test_invalid_payloads_are_rejected(self):
        with self.assertRaises(ValueError):
            encode_unicode_payload(clean_text="Done.", payload="")
        with self.assertRaises(ValueError):
            encode_unicode_payload(clean_text="Done.", payload="012")

    def test_reserved_codepoints_in_cover_are_rejected(self):
        with self.assertRaises(ValueError):
            encode_unicode_payload(
                clean_text=f"Done{DEFAULT_PREFIX_CODEPOINT}.",
                payload="01",
            )

    def test_malformed_frames_are_rejected(self):
        with self.assertRaises(ValueError):
            decode_unicode_payload("No frame.")
        with self.assertRaises(ValueError):
            decode_unicode_payload(
                f"Bad{DEFAULT_PREFIX_CODEPOINT}01{DEFAULT_SUFFIX_CODEPOINT}."
            )
        with self.assertRaises(ValueError):
            decode_unicode_payload(
                f"Bad{DEFAULT_PREFIX_CODEPOINT}{DEFAULT_SUFFIX_CODEPOINT}."
            )


if __name__ == "__main__":
    unittest.main()
