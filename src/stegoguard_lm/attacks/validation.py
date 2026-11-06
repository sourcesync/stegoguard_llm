from __future__ import annotations

from typing import Any

COMMON_REQUIRED_FIELDS = {
    "sample_id",
    "pair_id",
    "lineage_id",
    "comparison_id",
    "parent_sample_id",
    "label",
    "experiment_stage",
    "round_id",
    "adaptive",
    "feedback_condition",
    "attack_family",
    "attack_name",
    "attacker_access",
    "strategy_id",
    "strategy_version",
    "domain",
    "topic_id",
    "target_length",
    "parent_actual_token_length",
    "source_model",
    "text",
    "prompt",
    "payload",
    "payload_bit_length",
    "payload_sha256",
    "decoded_payload",
    "decode_success",
    "exact_match",
    "capacity_bits",
    "bits_per_token",
    "length_invariant",
    "length_valid",
    "encoding_attempts",
    "channel_parameters",
    "manifest_version",
    "validation_status",
}


def validate_attack_record_against_parent(
    record: dict[str, Any],
    parent: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    missing = COMMON_REQUIRED_FIELDS - set(record)
    if missing:
        return [f"Missing required fields: {sorted(missing)}"]

    expected_relations = {
        "parent_sample_id": parent["sample_id"],
        "pair_id": parent["pair_id"],
        "lineage_id": parent["lineage_id"],
        "source_model": parent["source_model"],
        "domain": parent["domain"],
        "topic_id": parent["topic_id"],
        "prompt": parent["prompt"],
    }
    for field, expected in expected_relations.items():
        if record.get(field) != expected:
            errors.append(f"{field} mismatch: {record.get(field)!r} != {expected!r}.")

    if record.get("label") != "stego":
        errors.append("label must equal 'stego'.")
    if int(record.get("round_id", -1)) != 0:
        errors.append("round_id must equal 0 for static smoke records.")
    if record.get("adaptive") is not False:
        errors.append("adaptive must be false for static smoke records.")
    if record.get("feedback_condition") != "none":
        errors.append("feedback_condition must equal 'none'.")
    if record.get("decode_success") is not True:
        errors.append("decode_success is not true.")
    if record.get("exact_match") is not True:
        errors.append("exact_match is not true.")
    if record.get("payload") != record.get("decoded_payload"):
        errors.append("decoded_payload does not equal payload.")
    if len(str(record.get("payload", ""))) != int(record.get("payload_bit_length", -1)):
        errors.append("payload_bit_length does not match the payload string.")
    if int(record.get("encoding_attempts", 0)) < 1:
        errors.append("encoding_attempts must be at least 1.")
    if record.get("validation_status") != "accepted":
        errors.append("validation_status must equal 'accepted'.")

    parent_actual = int(parent["actual_token_length"])
    if int(record.get("parent_actual_token_length", -1)) != parent_actual:
        errors.append("parent_actual_token_length does not match clean parent.")

    attack_family = record.get("attack_family")
    if attack_family in {"token_whitebox", "token_blackbox"}:
        if record.get("length_invariant") != "source_tokenizer_exact":
            errors.append("Token channel has an unexpected length_invariant.")
        if record.get("length_valid") is not True:
            errors.append("Token channel length_valid is not true.")
        if int(record.get("stego_actual_token_length", -1)) != parent_actual:
            errors.append(
                "Token-channel stego length does not equal parent actual length."
            )
    elif attack_family == "out_of_band":
        if record.get("length_invariant") != (
            "visible_text_exact_after_hidden_codepoint_strip"
        ):
            errors.append("Unicode channel has an unexpected length_invariant.")
        if record.get("visible_text_exact_after_strip") is not True:
            errors.append("Unicode visible text is not exactly preserved.")
        if record.get("length_valid") is not True:
            errors.append("Unicode length_valid/visible-preservation flag is false.")
    else:
        errors.append(f"Unknown attack_family: {attack_family!r}.")

    return errors
