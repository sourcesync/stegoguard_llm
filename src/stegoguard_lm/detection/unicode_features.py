from __future__ import annotations

import unicodedata
from typing import Any

ZERO_WIDTH_CODEPOINTS = {
    0x200B,  # ZERO WIDTH SPACE
    0x200C,  # ZERO WIDTH NON-JOINER
    0x200D,  # ZERO WIDTH JOINER
    0x2060,  # WORD JOINER
    0xFEFF,  # ZERO WIDTH NO-BREAK SPACE
}

# Unicode Tags block. Most tag characters are category Cf.
TAG_START = 0xE0000
TAG_END = 0xE007F

# Include the framing characters used by the frozen Unicode attack.
KNOWN_SUSPICIOUS = {
    0x2063,  # INVISIBLE SEPARATOR
    0x2064,  # INVISIBLE PLUS
    *ZERO_WIDTH_CODEPOINTS,
}


def extract_unicode_features(text: str) -> dict[str, Any]:
    nonprinting_count = 0
    zero_width_count = 0
    unicode_tag_count = 0
    suspicious: list[str] = []

    for char in text:
        codepoint = ord(char)
        category = unicodedata.category(char)
        if category in {"Cf", "Cc", "Cs", "Co", "Cn"}:
            nonprinting_count += 1
        if codepoint in ZERO_WIDTH_CODEPOINTS:
            zero_width_count += 1
        if TAG_START <= codepoint <= TAG_END:
            unicode_tag_count += 1
        if (
            codepoint in KNOWN_SUSPICIOUS
            or TAG_START <= codepoint <= TAG_END
            or category == "Cf"
        ):
            suspicious.append(f"U+{codepoint:04X}")

    return {
        "nonprinting_count": int(nonprinting_count),
        "zero_width_count": int(zero_width_count),
        "unicode_tag_count": int(unicode_tag_count),
        "has_suspicious_codepoint": int(bool(suspicious)),
        "suspicious_codepoints": suspicious,
    }
