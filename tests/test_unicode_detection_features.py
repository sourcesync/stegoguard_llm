from __future__ import annotations

import unittest

from stegoguard_lm.detection.unicode_features import extract_unicode_features


class UnicodeDetectionFeatureTests(unittest.TestCase):
    def test_clean_ascii_has_no_suspicious_codepoint(self):
        result = extract_unicode_features("A normal sentence.")
        self.assertEqual(result["has_suspicious_codepoint"], 0)
        self.assertEqual(result["zero_width_count"], 0)

    def test_frozen_attack_codepoints_are_detected(self):
        result = extract_unicode_features("Normal\u2063\u200b\u200c\u2064.")
        self.assertEqual(result["has_suspicious_codepoint"], 1)
        self.assertEqual(result["zero_width_count"], 2)
        self.assertGreaterEqual(result["nonprinting_count"], 4)


if __name__ == "__main__":
    unittest.main()
