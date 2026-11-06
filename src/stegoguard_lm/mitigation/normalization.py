from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class UnicodeSanitizationResult:
    text: str
    changed: bool
    normalization_form: str
    removed_codepoint_count: int
    removed_codepoints: tuple[str, ...]
    removed_format_codepoint_count: int


def codepoint_label(ch: str) -> str:
    name = unicodedata.name(ch, "UNNAMED")
    return f"U+{ord(ch):04X} {name}"


def sanitize_unicode_text(
    text: str,
    *,
    normalization_form: str = "NFC",
    reserved_codepoints: Iterable[str] = (),
    remove_all_format_codepoints: bool = True,
) -> UnicodeSanitizationResult:
    """Normalize text and remove hidden formatting controls deterministically.

    NFC is deliberately preferred to compatibility normalization so ordinary
    visible text is changed as little as possible.  The configured covert-channel
    symbols are always removed.  When ``remove_all_format_codepoints`` is true,
    all Unicode category-Cf codepoints are removed as a broader hygiene layer.
    """

    if normalization_form not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError(f"Unsupported Unicode normalization form: {normalization_form}")
    reserved = set(reserved_codepoints)
    if any(len(ch) != 1 for ch in reserved):
        raise ValueError("Every reserved Unicode symbol must be one codepoint.")

    normalized = unicodedata.normalize(normalization_form, text)
    kept: list[str] = []
    removed: list[str] = []
    format_removed = 0
    for ch in normalized:
        is_format = unicodedata.category(ch) == "Cf"
        if ch in reserved or (remove_all_format_codepoints and is_format):
            removed.append(ch)
            if is_format:
                format_removed += 1
            continue
        kept.append(ch)

    sanitized = "".join(kept)
    labels = tuple(codepoint_label(ch) for ch in removed)
    return UnicodeSanitizationResult(
        text=sanitized,
        changed=sanitized != text,
        normalization_form=normalization_form,
        removed_codepoint_count=len(removed),
        removed_codepoints=labels,
        removed_format_codepoint_count=format_removed,
    )
