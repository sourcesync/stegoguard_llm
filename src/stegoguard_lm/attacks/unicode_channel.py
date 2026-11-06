from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .base import AttackResult
from .payloads import validate_bitstring

DEFAULT_ZERO_CODEPOINT = "\u200b"
DEFAULT_ONE_CODEPOINT = "\u200c"
DEFAULT_PREFIX_CODEPOINT = "\u2063"
DEFAULT_SUFFIX_CODEPOINT = "\u2064"

_FINAL_PUNCTUATION = re.compile(r"[.!?][\"'’”)\]]*$")


@dataclass(slots=True)
class UnicodeDecodeResult:
    payload: str
    stripped_text: str
    prefix_index: int
    suffix_index: int
    hidden_codepoint_count: int


def _reserved_codepoints(
    *,
    zero_codepoint: str,
    one_codepoint: str,
    prefix_codepoint: str,
    suffix_codepoint: str,
) -> set[str]:
    values = {
        zero_codepoint,
        one_codepoint,
        prefix_codepoint,
        suffix_codepoint,
    }
    if any(len(value) != 1 for value in values):
        raise ValueError("Every Unicode channel symbol must be one codepoint.")
    if len(values) != 4:
        raise ValueError("Unicode channel symbols must be distinct.")
    return values


def _insertion_index_before_final_punctuation(text: str) -> int:
    match = _FINAL_PUNCTUATION.search(text.rstrip())
    if match is None:
        raise ValueError("Unicode cover text must end in sentence-closing punctuation.")
    return match.start()


def decode_unicode_payload(
    text: str,
    *,
    zero_codepoint: str = DEFAULT_ZERO_CODEPOINT,
    one_codepoint: str = DEFAULT_ONE_CODEPOINT,
    prefix_codepoint: str = DEFAULT_PREFIX_CODEPOINT,
    suffix_codepoint: str = DEFAULT_SUFFIX_CODEPOINT,
) -> UnicodeDecodeResult:
    reserved = _reserved_codepoints(
        zero_codepoint=zero_codepoint,
        one_codepoint=one_codepoint,
        prefix_codepoint=prefix_codepoint,
        suffix_codepoint=suffix_codepoint,
    )

    if text.count(prefix_codepoint) != 1:
        raise ValueError("Expected exactly one Unicode payload prefix.")
    if text.count(suffix_codepoint) != 1:
        raise ValueError("Expected exactly one Unicode payload suffix.")

    prefix_index = text.index(prefix_codepoint)
    suffix_index = text.index(suffix_codepoint)
    if suffix_index <= prefix_index:
        raise ValueError("Unicode payload suffix occurs before its prefix.")

    body = text[prefix_index + 1 : suffix_index]
    if not body:
        raise ValueError("Unicode payload frame is empty.")

    invalid = [ch for ch in body if ch not in {zero_codepoint, one_codepoint}]
    if invalid:
        codepoints = ", ".join(f"U+{ord(ch):04X}" for ch in invalid[:5])
        raise ValueError(
            "Unicode payload frame contains invalid codepoints: " + codepoints
        )

    outside = text[:prefix_index] + text[suffix_index + 1 :]
    unexpected = [ch for ch in outside if ch in reserved]
    if unexpected:
        codepoints = ", ".join(f"U+{ord(ch):04X}" for ch in unexpected[:5])
        raise ValueError(
            "Reserved Unicode channel codepoints occur outside the frame: " + codepoints
        )

    payload = "".join("0" if ch == zero_codepoint else "1" for ch in body)
    validate_bitstring(payload)

    return UnicodeDecodeResult(
        payload=payload,
        stripped_text=outside,
        prefix_index=prefix_index,
        suffix_index=suffix_index,
        hidden_codepoint_count=len(body) + 2,
    )


def strip_unicode_payload(
    text: str,
    **kwargs: Any,
) -> str:
    return decode_unicode_payload(text, **kwargs).stripped_text


def find_unicode_payload(
    text: str,
    **kwargs: Any,
) -> str:
    return decode_unicode_payload(text, **kwargs).payload


def encode_unicode_payload(
    *,
    clean_text: str,
    payload: str,
    tokenizer=None,
    zero_codepoint: str = DEFAULT_ZERO_CODEPOINT,
    one_codepoint: str = DEFAULT_ONE_CODEPOINT,
    prefix_codepoint: str = DEFAULT_PREFIX_CODEPOINT,
    suffix_codepoint: str = DEFAULT_SUFFIX_CODEPOINT,
    framing_version: str = "unicode_frame_v1",
    embed_position: str = "before_final_visible_punctuation",
) -> AttackResult:
    validate_bitstring(payload)
    reserved = _reserved_codepoints(
        zero_codepoint=zero_codepoint,
        one_codepoint=one_codepoint,
        prefix_codepoint=prefix_codepoint,
        suffix_codepoint=suffix_codepoint,
    )

    preexisting = [ch for ch in clean_text if ch in reserved]
    if preexisting:
        codepoints = ", ".join(f"U+{ord(ch):04X}" for ch in preexisting[:5])
        raise ValueError(
            "Clean cover already contains reserved Unicode channel codepoints: "
            + codepoints
        )

    if embed_position != "before_final_visible_punctuation":
        raise ValueError(f"Unsupported Unicode embed_position: {embed_position}")

    insertion_index = _insertion_index_before_final_punctuation(clean_text)
    body = "".join(zero_codepoint if bit == "0" else one_codepoint for bit in payload)
    frame = prefix_codepoint + body + suffix_codepoint
    stego_text = clean_text[:insertion_index] + frame + clean_text[insertion_index:]

    decoded = decode_unicode_payload(
        stego_text,
        zero_codepoint=zero_codepoint,
        one_codepoint=one_codepoint,
        prefix_codepoint=prefix_codepoint,
        suffix_codepoint=suffix_codepoint,
    )

    exact_match = decoded.payload == payload
    visible_exact = decoded.stripped_text == clean_text

    clean_source_tokens = None
    raw_stego_source_tokens = None
    stripped_source_tokens = None
    same_raw_token_length = None
    if tokenizer is not None:
        clean_source_tokens = len(
            tokenizer.encode(clean_text, add_special_tokens=False)
        )
        raw_stego_source_tokens = len(
            tokenizer.encode(stego_text, add_special_tokens=False)
        )
        stripped_source_tokens = len(
            tokenizer.encode(decoded.stripped_text, add_special_tokens=False)
        )
        same_raw_token_length = raw_stego_source_tokens == clean_source_tokens

    codepoint_names = {
        "prefix": f"U+{ord(prefix_codepoint):04X}",
        "zero": f"U+{ord(zero_codepoint):04X}",
        "one": f"U+{ord(one_codepoint):04X}",
        "suffix": f"U+{ord(suffix_codepoint):04X}",
    }

    return AttackResult(
        attack_family="out_of_band",
        attack_name="unicode_hidden_channel",
        attacker_access="text_transformation",
        strategy_id="unicode_hidden_channel",
        strategy_version=framing_version,
        stego_text=stego_text,
        payload=payload,
        decoded_payload=decoded.payload,
        decode_success=True,
        exact_match=exact_match,
        capacity_bits=len(payload),
        payload_bit_length=len(payload),
        encoding_attempts=1,
        source_token_length=raw_stego_source_tokens,
        length_invariant="visible_text_exact_after_hidden_codepoint_strip",
        length_valid=visible_exact,
        channel_parameters={
            "framing_version": framing_version,
            "embed_position": embed_position,
            "hidden_codepoint_count": decoded.hidden_codepoint_count,
            "hidden_codepoints": codepoint_names,
            "raw_character_length": len(stego_text),
            "clean_character_length": len(clean_text),
            "clean_source_token_length": clean_source_tokens,
            "raw_stego_source_token_length": raw_stego_source_tokens,
            "visible_text_source_token_length_after_strip": stripped_source_tokens,
            "same_raw_token_length": same_raw_token_length,
            "visible_text_exact_after_strip": visible_exact,
            "insertion_index": insertion_index,
        },
    )
