from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .base import AttackResult
from .payloads import validate_bitstring

_COMPLETE_ENDING = re.compile(r"[.!?][\"'’”)\]]*$")


@dataclass(frozen=True, slots=True)
class SynonymSlot:
    slot_id: str
    zero: str
    one: str


@dataclass(frozen=True, slots=True)
class SynonymCodebook:
    version: str
    slots: tuple[SynonymSlot, ...]


@dataclass(slots=True)
class SynonymDecodeResult:
    payload: str
    success: bool
    slot_results: list[dict[str, Any]]
    errors: list[str]


@dataclass(frozen=True, slots=True)
class InplaceSynonymSlot:
    """A codebook slot that can encode either bit without changing total length."""

    slot_id: str
    zero: str
    one: str
    original_term: str
    original_bit: str
    start: int
    end: int


# ---------------------------------------------------------------------------
# Codebook loading and lexical matching
# ---------------------------------------------------------------------------


def _whole_word_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)


def _term_matches(text: str, term: str) -> list[re.Match[str]]:
    return list(_whole_word_pattern(term).finditer(text))


def _count_term(text: str, term: str) -> int:
    return len(_term_matches(text, term))


def validate_synonym_codebook(codebook: SynonymCodebook) -> list[str]:
    errors: list[str] = []
    if not codebook.version.strip():
        errors.append("Codebook version is empty.")
    if not codebook.slots:
        errors.append("Codebook has no slots.")

    seen_slot_ids: set[str] = set()
    seen_terms: dict[str, str] = {}

    for slot in codebook.slots:
        if not slot.slot_id.strip():
            errors.append("A slot has an empty slot_id.")
        elif slot.slot_id in seen_slot_ids:
            errors.append(f"Duplicate slot_id: {slot.slot_id}")
        seen_slot_ids.add(slot.slot_id)

        zero = slot.zero.strip()
        one = slot.one.strip()
        if not zero or not one:
            errors.append(f"{slot.slot_id}: alternatives must be nonempty.")
            continue
        if zero.casefold() == one.casefold():
            errors.append(f"{slot.slot_id}: alternatives are identical.")

        for term, label in ((zero, "zero"), (one, "one")):
            folded = term.casefold()
            prior = seen_terms.get(folded)
            if prior is not None:
                errors.append(
                    f"Duplicate alternative {term!r} in {slot.slot_id}:{label}; "
                    f"already used by {prior}."
                )
            else:
                seen_terms[folded] = f"{slot.slot_id}:{label}"

    all_terms = list(seen_terms)
    for i, first in enumerate(all_terms):
        for second in all_terms[i + 1 :]:
            if first in second or second in first:
                errors.append(
                    "Substring collision between alternatives: "
                    f"{first!r} and {second!r}."
                )

    return errors


def load_synonym_codebook(path: str | Path) -> SynonymCodebook:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Synonym codebook not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Synonym codebook root must be a mapping.")

    slots_raw = raw.get("slots")
    if not isinstance(slots_raw, list):
        raise ValueError("Synonym codebook 'slots' must be a list.")

    codebook = SynonymCodebook(
        version=str(raw.get("version", "")).strip(),
        slots=tuple(
            SynonymSlot(
                slot_id=str(item.get("slot_id", "")).strip(),
                zero=str(item.get("zero", "")).strip(),
                one=str(item.get("one", "")).strip(),
            )
            for item in slots_raw
            if isinstance(item, dict)
        ),
    )

    errors = validate_synonym_codebook(codebook)
    if errors:
        raise ValueError("Invalid synonym codebook: " + " | ".join(errors))
    return codebook


def select_synonym_alternatives(
    payload: str,
    codebook: SynonymCodebook,
) -> tuple[list[str], list[str]]:
    """Legacy helper retained for compatibility with codebook tests."""
    validate_bitstring(payload)
    if len(payload) != len(codebook.slots):
        raise ValueError(
            f"Payload has {len(payload)} bits but codebook has "
            f"{len(codebook.slots)} slots."
        )

    selected: list[str] = []
    excluded: list[str] = []
    for bit, slot in zip(payload, codebook.slots, strict=True):
        if bit == "0":
            selected.append(slot.zero)
            excluded.append(slot.one)
        else:
            selected.append(slot.one)
            excluded.append(slot.zero)
    return selected, excluded


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------


def _safe_token_ids(tokenizer, text: str) -> list[int]:
    return [
        int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)
    ]


def _roundtrip_is_stable(tokenizer, text: str) -> tuple[bool, list[int]]:
    token_ids = _safe_token_ids(tokenizer, text)
    reconstructed = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    reconstructed_ids = _safe_token_ids(tokenizer, reconstructed)
    return reconstructed_ids == token_ids, token_ids


def _has_complete_ending(text: str) -> bool:
    return bool(_COMPLETE_ENDING.search(text.rstrip()))


def _format_codepoints(text: str) -> list[str]:
    return [character for character in text if unicodedata.category(character) == "Cf"]


def _has_forbidden_format_codepoints(text: str) -> bool:
    return bool(_format_codepoints(text))


def _preserve_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source.islower():
        return replacement.lower()
    if source[:1].isupper() and source[1:].islower():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _replace_exactly_one(
    text: str,
    *,
    slot: SynonymSlot,
    bit: str,
) -> tuple[str, str, str]:
    """Replace the single occurrence of either slot alternative.

    Returns (new_text, observed_term, replacement_term).
    """
    if bit not in {"0", "1"}:
        raise ValueError(f"Invalid bit: {bit!r}")

    zero_matches = _term_matches(text, slot.zero)
    one_matches = _term_matches(text, slot.one)
    matches = [(match, slot.zero, "0") for match in zero_matches] + [
        (match, slot.one, "1") for match in one_matches
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{slot.slot_id}: expected exactly one codebook alternative in text; "
            f"found zero={len(zero_matches)}, one={len(one_matches)}."
        )

    match, observed_term, _ = matches[0]
    target_term = slot.zero if bit == "0" else slot.one
    replacement = _preserve_case(match.group(0), target_term)
    new_text = text[: match.start()] + replacement + text[match.end() :]
    return new_text, observed_term, target_term


def _validate_text_invariants(
    *,
    text: str,
    tokenizer,
    required_token_length: int,
    require_complete_ending: bool,
    require_tokenizer_roundtrip: bool,
    forbid_format_codepoints: bool,
) -> tuple[bool, str, list[int]]:
    if forbid_format_codepoints and _has_forbidden_format_codepoints(text):
        codepoints = [f"U+{ord(ch):04X}" for ch in _format_codepoints(text)]
        return (
            False,
            "forbidden_format_codepoint: synonym text contains invisible "
            f"Unicode formatting characters {codepoints}.",
            _safe_token_ids(tokenizer, text),
        )

    token_ids = _safe_token_ids(tokenizer, text)
    if len(token_ids) != required_token_length:
        return (
            False,
            f"length_mismatch: text has {len(token_ids)} source-tokenizer tokens, "
            f"expected {required_token_length}.",
            token_ids,
        )

    if require_tokenizer_roundtrip:
        stable, stable_ids = _roundtrip_is_stable(tokenizer, text)
        if not stable:
            return (
                False,
                "tokenizer_roundtrip_failure: token IDs changed after decode/encode.",
                stable_ids,
            )
        token_ids = stable_ids

    if require_complete_ending and not _has_complete_ending(text):
        return (
            False,
            "incomplete_ending: text lacks final sentence punctuation.",
            token_ids,
        )

    return True, "accepted", token_ids


# ---------------------------------------------------------------------------
# V4 in-place synonym channel
# ---------------------------------------------------------------------------


def discover_inplace_synonym_slots(
    *,
    clean_text: str,
    tokenizer,
    codebook: SynonymCodebook,
    required_token_length: int,
    require_complete_ending: bool = True,
    require_tokenizer_roundtrip: bool = True,
    forbid_format_codepoints: bool = True,
) -> list[InplaceSynonymSlot]:
    """Find lexical choices that can carry either bit at exact same length.

    The V1-V3 prompt-driven channel tried to generate a new exact-length answer
    containing all eight codebook words. Live smoke tests showed that this was
    brittle under strict token equality, especially for 256-token technical
    answers. V4 instead treats the already validated clean response as the cover
    and changes only naturally occurring codebook alternatives.

    A slot is eligible only when:
      * exactly one of its two alternatives occurs exactly once in the cover;
      * choosing either alternative preserves the parent's exact token length;
      * both alternatives survive tokenizer decode/re-tokenize checks;
      * the visible-text ending and Unicode hygiene invariants remain valid.

    The resulting capacity is intentionally variable by cover text.
    """
    base_ok, base_reason, _ = _validate_text_invariants(
        text=clean_text,
        tokenizer=tokenizer,
        required_token_length=required_token_length,
        require_complete_ending=require_complete_ending,
        require_tokenizer_roundtrip=require_tokenizer_roundtrip,
        forbid_format_codepoints=forbid_format_codepoints,
    )
    if not base_ok:
        raise ValueError(f"Clean cover violates synonym prerequisites: {base_reason}")

    eligible: list[InplaceSynonymSlot] = []

    for slot in codebook.slots:
        zero_matches = _term_matches(clean_text, slot.zero)
        one_matches = _term_matches(clean_text, slot.one)
        matches = [(m, slot.zero, "0") for m in zero_matches] + [
            (m, slot.one, "1") for m in one_matches
        ]
        if len(matches) != 1:
            continue

        match, original_term, original_bit = matches[0]

        both_valid = True
        for bit in ("0", "1"):
            variant, _, _ = _replace_exactly_one(
                clean_text,
                slot=slot,
                bit=bit,
            )
            valid, _, _ = _validate_text_invariants(
                text=variant,
                tokenizer=tokenizer,
                required_token_length=required_token_length,
                require_complete_ending=require_complete_ending,
                require_tokenizer_roundtrip=require_tokenizer_roundtrip,
                forbid_format_codepoints=forbid_format_codepoints,
            )
            if not valid:
                both_valid = False
                break

        if both_valid:
            eligible.append(
                InplaceSynonymSlot(
                    slot_id=slot.slot_id,
                    zero=slot.zero,
                    one=slot.one,
                    original_term=original_term,
                    original_bit=original_bit,
                    start=int(match.start()),
                    end=int(match.end()),
                )
            )

    # Text order is transparent and deterministic for both encoder and analysis.
    eligible.sort(key=lambda item: (item.start, item.slot_id))
    return eligible


def _slots_by_id(codebook: SynonymCodebook) -> dict[str, SynonymSlot]:
    return {slot.slot_id: slot for slot in codebook.slots}


def decode_synonym_payload(
    text: str,
    codebook: SynonymCodebook,
    active_slot_ids: Iterable[str] | None = None,
) -> SynonymDecodeResult:
    """Decode either all codebook slots (legacy) or a declared active subset."""
    slot_lookup = _slots_by_id(codebook)
    if active_slot_ids is None:
        slots = list(codebook.slots)
    else:
        slots = []
        seen: set[str] = set()
        for slot_id in active_slot_ids:
            slot_id = str(slot_id)
            if slot_id in seen:
                return SynonymDecodeResult(
                    payload="",
                    success=False,
                    slot_results=[],
                    errors=[f"Duplicate active slot_id: {slot_id}"],
                )
            seen.add(slot_id)
            slot = slot_lookup.get(slot_id)
            if slot is None:
                return SynonymDecodeResult(
                    payload="",
                    success=False,
                    slot_results=[],
                    errors=[f"Unknown active slot_id: {slot_id}"],
                )
            slots.append(slot)

    if not slots:
        return SynonymDecodeResult(
            payload="",
            success=False,
            slot_results=[],
            errors=["No active synonym slots were supplied."],
        )

    bits: list[str] = []
    errors: list[str] = []
    slot_results: list[dict[str, Any]] = []

    for slot in slots:
        zero_count = _count_term(text, slot.zero)
        one_count = _count_term(text, slot.one)

        if zero_count == 1 and one_count == 0:
            status = "valid"
            decoded_bit: str | None = "0"
            bits.append("0")
        elif zero_count == 0 and one_count == 1:
            status = "valid"
            decoded_bit = "1"
            bits.append("1")
        elif zero_count == 0 and one_count == 0:
            status = "missing"
            decoded_bit = None
            errors.append(f"{slot.slot_id}: neither alternative is present.")
        elif zero_count > 0 and one_count > 0:
            status = "ambiguous"
            decoded_bit = None
            errors.append(f"{slot.slot_id}: both alternatives are present.")
        else:
            status = "repeated"
            decoded_bit = None
            errors.append(
                f"{slot.slot_id}: an alternative occurs more than once "
                f"(zero={zero_count}, one={one_count})."
            )

        slot_results.append(
            {
                "slot_id": slot.slot_id,
                "zero": slot.zero,
                "one": slot.one,
                "zero_count": zero_count,
                "one_count": one_count,
                "decoded_bit": decoded_bit,
                "status": status,
            }
        )

    return SynonymDecodeResult(
        payload="".join(bits) if not errors else "",
        success=not errors,
        slot_results=slot_results,
        errors=errors,
    )


def encode_synonym_payload(
    *,
    clean_text: str,
    tokenizer,
    payload: str,
    codebook: SynonymCodebook,
    required_token_length: int,
    max_payload_bits: int = 8,
    min_payload_bits: int = 1,
    require_text_change: bool = True,
    require_complete_ending: bool = True,
    require_tokenizer_roundtrip: bool = True,
    forbid_format_codepoints: bool = True,
) -> AttackResult:
    """Encode a variable-length payload by in-place lexical substitution.

    Only source-tokenizer-length-preserving synonym choices already present in
    the validated clean cover are used. No model generation, logits, or tail
    repair is involved. This makes the black-box lexical channel low-capacity
    but robust and directly aligned with the strict same-length threat model.

    `payload` may be longer than the available capacity. The encoder uses the
    longest prefix that can be embedded, capped by `max_payload_bits`.
    """
    validate_bitstring(payload)
    if required_token_length < 1:
        raise ValueError("required_token_length must be positive.")
    if max_payload_bits < 1:
        raise ValueError("max_payload_bits must be positive.")
    if min_payload_bits < 1:
        raise ValueError("min_payload_bits must be positive.")
    if min_payload_bits > max_payload_bits:
        raise ValueError("min_payload_bits cannot exceed max_payload_bits.")

    eligible = discover_inplace_synonym_slots(
        clean_text=clean_text,
        tokenizer=tokenizer,
        codebook=codebook,
        required_token_length=required_token_length,
        require_complete_ending=require_complete_ending,
        require_tokenizer_roundtrip=require_tokenizer_roundtrip,
        forbid_format_codepoints=forbid_format_codepoints,
    )

    usable_count = min(len(eligible), max_payload_bits, len(payload))
    if usable_count < min_payload_bits:
        raise RuntimeError(
            "no_capacity: cover has "
            f"{len(eligible)} bidirectionally length-stable synonym slots; "
            f"minimum required is {min_payload_bits}."
        )

    slot_lookup = _slots_by_id(codebook)
    last_reason = "not_attempted"

    # If interactions among multiple lexical substitutions create an unexpected
    # tokenizer boundary effect, fall back to a shorter payload rather than
    # pretending that the larger capacity was usable.
    for payload_length in range(usable_count, min_payload_bits - 1, -1):
        active = eligible[:payload_length]
        used_payload = payload[:payload_length]
        text = clean_text
        substitutions: list[dict[str, Any]] = []

        try:
            for bit, active_slot in zip(used_payload, active, strict=True):
                slot = slot_lookup[active_slot.slot_id]
                before = text
                text, observed_term, selected_term = _replace_exactly_one(
                    text,
                    slot=slot,
                    bit=bit,
                )

                valid, reason, _ = _validate_text_invariants(
                    text=text,
                    tokenizer=tokenizer,
                    required_token_length=required_token_length,
                    require_complete_ending=require_complete_ending,
                    require_tokenizer_roundtrip=require_tokenizer_roundtrip,
                    forbid_format_codepoints=forbid_format_codepoints,
                )
                if not valid:
                    raise RuntimeError(reason)

                substitutions.append(
                    {
                        "slot_id": slot.slot_id,
                        "bit": bit,
                        "observed_term_before": observed_term,
                        "selected_term": selected_term,
                        "text_changed": before != text,
                    }
                )
        except RuntimeError as exc:
            last_reason = f"combination_validation_failure: {exc}"
            continue

        valid, reason, final_ids = _validate_text_invariants(
            text=text,
            tokenizer=tokenizer,
            required_token_length=required_token_length,
            require_complete_ending=require_complete_ending,
            require_tokenizer_roundtrip=require_tokenizer_roundtrip,
            forbid_format_codepoints=forbid_format_codepoints,
        )
        if not valid:
            last_reason = reason
            continue

        if require_text_change and text == clean_text:
            last_reason = (
                "no_text_change: payload selected the same lexical orientation "
                "already present in every active slot."
            )
            continue

        active_slot_ids = [slot.slot_id for slot in active]
        decoded = decode_synonym_payload(
            text,
            codebook,
            active_slot_ids=active_slot_ids,
        )
        if not decoded.success:
            last_reason = "slot_validation_failure: " + " | ".join(decoded.errors)
            continue
        if decoded.payload != used_payload:
            last_reason = (
                f"payload_mismatch: decoded {decoded.payload!r}, "
                f"expected {used_payload!r}."
            )
            continue

        return AttackResult(
            attack_family="token_blackbox",
            attack_name="synonym_slot_codebook",
            attacker_access="black_box_tokenizer_aware",
            strategy_id="synonym_inplace_substitution",
            strategy_version=f"{codebook.version}+inplace_v1",
            stego_text=text,
            payload=used_payload,
            decoded_payload=decoded.payload,
            decode_success=True,
            exact_match=True,
            capacity_bits=len(active),
            payload_bit_length=len(used_payload),
            encoding_attempts=1,
            source_token_length=len(final_ids),
            length_invariant="source_tokenizer_exact",
            length_valid=len(final_ids) == required_token_length,
            channel_parameters={
                "codebook_version": codebook.version,
                "embedding_method": "inplace_synonym_substitution",
                "eligible_slot_ids": [slot.slot_id for slot in eligible],
                "eligible_capacity_bits": len(eligible),
                "active_slot_ids": active_slot_ids,
                "active_slot_count": len(active),
                "original_orientations": {
                    slot.slot_id: slot.original_bit for slot in active
                },
                "substitutions": substitutions,
                "changed_slot_ids": [
                    item["slot_id"]
                    for item in substitutions
                    if bool(item["text_changed"])
                ],
                "required_token_length": required_token_length,
                "generated_token_ids": final_ids,
                "tokenizer_roundtrip_valid": True,
                "ending_valid": _has_complete_ending(text),
                "forbid_format_codepoints": forbid_format_codepoints,
                "uses_source_tokenizer_for_length_check": True,
                "model_generation_used": False,
            },
        )

    raise RuntimeError("Synonym in-place encoding failed. Last error: " + last_reason)
