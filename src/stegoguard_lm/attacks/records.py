from __future__ import annotations

import re
from typing import Any

from .base import AttackResult
from .payloads import payload_sha256


def attack_slug(value: str) -> str:
    slug = value.casefold().replace("/", "_")
    slug = re.sub(r"[^a-z0-9_]+", "_", slug)
    return re.sub(r"_+", "_", slug).strip("_")


def make_attack_sample_id(parent_record: dict[str, Any], attack_name: str) -> str:
    return f"stego_{attack_slug(attack_name)}_{parent_record['pair_id']}"


def make_comparison_id(parent_record: dict[str, Any], attack_name: str) -> str:
    return f"{parent_record['pair_id']}__{attack_slug(attack_name)}"


def build_attack_record(
    *,
    parent_record: dict[str, Any],
    result: AttackResult,
    manifest_version: str,
    experiment_stage: str = "static_attack_smoke",
    attack_seed: int | None = None,
) -> dict[str, Any]:
    parent_actual_length = int(parent_record["actual_token_length"])
    bits_per_token = result.payload_bit_length / parent_actual_length

    record: dict[str, Any] = {
        "sample_id": make_attack_sample_id(parent_record, result.attack_name),
        "pair_id": parent_record["pair_id"],
        "lineage_id": parent_record["lineage_id"],
        "comparison_id": make_comparison_id(parent_record, result.attack_name),
        "parent_sample_id": parent_record["sample_id"],
        "label": "stego",
        "experiment_stage": experiment_stage,
        "round_id": 0,
        "adaptive": False,
        "feedback_condition": "none",
        "attack_family": result.attack_family,
        "attack_name": result.attack_name,
        "attacker_access": result.attacker_access,
        "strategy_id": result.strategy_id,
        "strategy_version": result.strategy_version,
        "domain": parent_record["domain"],
        "topic_id": parent_record["topic_id"],
        "topic_seed": parent_record.get("topic_seed"),
        "target_length": int(parent_record["target_length"]),
        "nominal_target_length": int(parent_record["target_length"]),
        "parent_actual_token_length": parent_actual_length,
        "stego_actual_token_length": result.source_token_length,
        "source_model": parent_record["source_model"],
        "model_revision": parent_record.get("model_revision"),
        "tokenizer_revision": parent_record.get("tokenizer_revision"),
        "text": result.stego_text,
        "prompt": parent_record["prompt"],
        "prompt_id": parent_record.get("prompt_id"),
        "payload": result.payload,
        "payload_bit_length": result.payload_bit_length,
        "payload_sha256": payload_sha256(result.payload),
        "decoded_payload": result.decoded_payload,
        "decode_success": result.decode_success,
        "exact_match": result.exact_match,
        "capacity_bits": result.capacity_bits,
        "bits_per_token": bits_per_token,
        "length_invariant": result.length_invariant,
        "length_valid": result.length_valid,
        "attack_seed": attack_seed,
        "encoding_attempts": result.encoding_attempts,
        "channel_parameters": result.channel_parameters,
        "manifest_version": manifest_version,
        "validation_status": "accepted",
    }

    if result.attack_family == "out_of_band":
        parameters = result.channel_parameters
        record.update(
            {
                "raw_stego_source_token_length": parameters.get(
                    "raw_stego_source_token_length"
                ),
                "visible_text_source_token_length_after_strip": parameters.get(
                    "visible_text_source_token_length_after_strip"
                ),
                "same_raw_token_length": parameters.get("same_raw_token_length"),
                "visible_text_exact_after_strip": parameters.get(
                    "visible_text_exact_after_strip"
                ),
            }
        )

    return record
