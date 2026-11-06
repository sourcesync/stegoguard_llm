from __future__ import annotations

import unittest

from stegoguard_lm.mitigation.normalization import sanitize_unicode_text


class MitigationNormalizationTests(unittest.TestCase):
    def test_reserved_unicode_frame_is_removed(self) -> None:
        text = "Hello\u2063\u200b\u200c\u2064."
        result = sanitize_unicode_text(
            text,
            reserved_codepoints=["\u200b", "\u200c", "\u2063", "\u2064"],
            remove_all_format_codepoints=True,
        )
        self.assertEqual(result.text, "Hello.")
        self.assertTrue(result.changed)
        self.assertEqual(result.removed_codepoint_count, 4)

    def test_clean_ascii_text_is_unchanged(self) -> None:
        result = sanitize_unicode_text("Ordinary text.")
        self.assertEqual(result.text, "Ordinary text.")
        self.assertFalse(result.changed)

    def test_all_format_controls_are_removed_when_enabled(self) -> None:
        result = sanitize_unicode_text("a\u200db", remove_all_format_codepoints=True)
        self.assertEqual(result.text, "ab")
        self.assertEqual(result.removed_format_codepoint_count, 1)


if __name__ == "__main__":
    unittest.main()
