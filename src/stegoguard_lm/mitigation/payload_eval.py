from __future__ import annotations

import hashlib
from typing import Any

from stegoguard_lm.attacks.rank_channel import (
    bit_for_rank,
    decode_rank_payload,
    keyed_rank_offset,
)
from stegoguard_lm.attacks.synonym_channel import SynonymCodebook, decode_synonym_payload
from stegoguard_lm.attacks.unicode_channel import decode_unicode_payload


def bit_recovery_stats(expected: str, observed: str | None) -> dict[str, Any]:
    if not expected:
        raise ValueError("Expected payload cannot be empty for stego evaluation.")
    observed = observed or ""
    correct = sum(1 for a, b in zip(expected, observed) if a == b)
    compared = min(len(expected), len(observed))
    return {
        "expected_bit_count": len(expected),
        "observed_bit_count": len(observed),
        "compared_bit_count": compared,
        "correct_bit_count": correct,
        "effective_bit_accuracy": correct / len(expected),
        "effective_bit_error_rate": 1.0 - (correct / len(expected)),
        "conditional_observed_bit_accuracy": (correct / compared) if compared else None,
    }


def evaluate_unicode_payload_after_mitigation(
    *,
    original_attack: dict[str, Any],
    mitigated_text: str,
    attack_config: dict[str, Any],
) -> dict[str, Any]:
    expected = str(original_attack["payload"])
    try:
        decoded = decode_unicode_payload(
            mitigated_text,
            zero_codepoint=attack_config["bit_zero_codepoint"],
            one_codepoint=attack_config["bit_one_codepoint"],
            prefix_codepoint=attack_config["prefix_codepoint"],
            suffix_codepoint=attack_config["suffix_codepoint"],
        )
        observed = decoded.payload
        decoder_success = True
        decoder_error = None
    except Exception as exc:
        observed = ""
        decoder_success = False
        decoder_error = f"{type(exc).__name__}: {exc}"
    stats = bit_recovery_stats(expected, observed)
    return {
        "decoder_success": decoder_success,
        "decoder_error": decoder_error,
        "decoded_payload": observed,
        "exact_payload_recovery": decoder_success and observed == expected,
        **stats,
        "residual_capacity_bits": len(observed) if decoder_success else 0,
        "residual_capacity_fraction": (len(observed) / len(expected)) if decoder_success else 0.0,
    }


def evaluate_synonym_payload_after_mitigation(
    *,
    original_attack: dict[str, Any],
    mitigated_text: str,
    codebook: SynonymCodebook,
) -> dict[str, Any]:
    expected = str(original_attack["payload"])
    active_slot_ids = list(original_attack["channel_parameters"]["active_slot_ids"])
    decoded = decode_synonym_payload(
        mitigated_text,
        codebook,
        active_slot_ids=active_slot_ids,
    )
    observed_bits: list[str] = []
    correct = 0
    recoverable = 0
    slot_details: list[dict[str, Any]] = []
    for expected_bit, slot_result in zip(expected, decoded.slot_results, strict=True):
        bit = slot_result.get("decoded_bit")
        if bit in {"0", "1"}:
            recoverable += 1
            observed_bits.append(str(bit))
            if str(bit) == expected_bit:
                correct += 1
        else:
            observed_bits.append("?")
        slot_details.append(
            {
                "slot_id": slot_result["slot_id"],
                "expected_bit": expected_bit,
                "decoded_bit": bit,
                "status": slot_result["status"],
            }
        )
    effective_accuracy = correct / len(expected)
    observed_exact = decoded.payload if decoded.success else ""
    return {
        "decoder_success": bool(decoded.success),
        "decoder_error": None if decoded.success else " | ".join(decoded.errors),
        "decoded_payload": observed_exact,
        "decoded_slot_pattern": "".join(observed_bits),
        "exact_payload_recovery": bool(decoded.success and decoded.payload == expected),
        "expected_bit_count": len(expected),
        "observed_bit_count": recoverable,
        "compared_bit_count": recoverable,
        "correct_bit_count": correct,
        "effective_bit_accuracy": effective_accuracy,
        "effective_bit_error_rate": 1.0 - effective_accuracy,
        "conditional_observed_bit_accuracy": (correct / recoverable) if recoverable else None,
        "residual_capacity_bits": recoverable,
        "residual_capacity_fraction": recoverable / len(expected),
        "slot_details": slot_details,
    }


def _rank_partial_bits_from_observed_ranks(
    *,
    tokenizer: Any,
    prompt: str,
    mitigated_text: str,
    attack_seed: int,
    encoding_nonce: int,
    observed_ranks: list[int],
) -> str:
    observed_ids = tokenizer.encode(mitigated_text, add_special_tokens=False)
    fingerprint = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    bits: list[str] = []
    prefix: list[int] = []
    for position, rank in enumerate(observed_ranks):
        if position >= len(observed_ids):
            break
        offset = keyed_rank_offset(
            attack_seed=attack_seed,
            encoding_nonce=encoding_nonce,
            token_position=position,
            prompt_fingerprint=fingerprint,
            prefix_token_ids=prefix,
        )
        bits.append(bit_for_rank(int(rank), offset))
        prefix.append(int(observed_ids[position]))
    return "".join(bits)


def evaluate_rank_payload_after_mitigation(
    *,
    original_attack: dict[str, Any],
    mitigated_text: str,
    model: Any,
    tokenizer: Any,
    attack_config: dict[str, Any],
) -> dict[str, Any]:
    expected = str(original_attack["payload"])
    parameters = original_attack["channel_parameters"]
    attack_seed = int(parameters.get("attack_seed", original_attack["attack_seed"]))
    encoding_nonce = int(parameters["encoding_nonce"])
    decoded = decode_rank_payload(
        model=model,
        tokenizer=tokenizer,
        prompt=str(original_attack["prompt"]),
        stego_text=mitigated_text,
        attack_seed=attack_seed,
        encoding_nonce=encoding_nonce,
        top_k=int(parameters.get("top_k", attack_config["top_k"])),
        final_candidate_search_k=int(
            parameters.get("final_candidate_search_k", attack_config["final_candidate_search_k"])
        ),
        require_complete_ending=bool(
            parameters.get("final_token_policy", "") == "sentence_ending_candidates"
            or attack_config.get("require_complete_ending", True)
        ),
    )
    if decoded.success:
        observed = decoded.payload
    else:
        observed = _rank_partial_bits_from_observed_ranks(
            tokenizer=tokenizer,
            prompt=str(original_attack["prompt"]),
            mitigated_text=mitigated_text,
            attack_seed=attack_seed,
            encoding_nonce=encoding_nonce,
            observed_ranks=list(decoded.observed_ranks),
        )
    stats = bit_recovery_stats(expected, observed)
    return {
        "decoder_success": bool(decoded.success),
        "decoder_error": None if decoded.success else " | ".join(decoded.errors),
        "decoded_payload": decoded.payload if decoded.success else "",
        "recovered_prefix_payload": observed,
        "exact_payload_recovery": bool(decoded.success and decoded.payload == expected),
        **stats,
        "residual_capacity_bits": len(observed),
        "residual_capacity_fraction": len(observed) / len(expected),
        "observed_rank_count": len(decoded.observed_ranks),
    }
