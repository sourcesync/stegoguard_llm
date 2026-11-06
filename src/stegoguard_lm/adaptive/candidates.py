from __future__ import annotations

from typing import Any

from stegoguard_lm.attacks.records import build_attack_record
from stegoguard_lm.attacks.rank_channel import encode_rank_payload
from stegoguard_lm.attacks.synonym_channel import (
    SynonymCodebook,
    encode_synonym_payload,
)

from .strategy import RANK, SYNONYM, transformed_codebook


def copy_static_candidate(
    *,
    spec: dict[str, Any],
    static_attack: dict[str, Any],
    protocol_version: str,
) -> dict[str, Any]:
    row = dict(static_attack)
    row.update(
        {
            "candidate_id": str(spec["candidate_id"]),
            "sample_id": str(spec["candidate_id"]),
            "experiment_stage": "step10_adaptive_candidate_catalog",
            "round_id": None,
            "adaptive": True,
            "feedback_condition": "candidate_catalog",
            "strategy_index": int(spec["strategy_index"]),
            "strategy_id": str(spec["strategy_id"]),
            "strategy_attack_seed": int(spec["strategy_attack_seed"]),
            "synonym_mapping_mask": spec.get("synonym_mapping_mask"),
            "static_step7_sample_id": str(spec["static_step7_sample_id"]),
            "candidate_spec_sha256": str(spec["candidate_spec_sha256"]),
            "protocol_version": protocol_version,
            "generation_status": "accepted",
            "candidate_source": "frozen_step7_strategy0",
        }
    )
    return row


def generate_candidate(
    *,
    config: dict[str, Any],
    spec: dict[str, Any],
    parent: dict[str, Any],
    tokenizer: Any,
    model: Any | None,
    codebook: SynonymCodebook,
) -> dict[str, Any]:
    attack_name = str(spec["attack_name"])
    protocol = str(config["adaptive_evaluation"]["protocol_version"])
    required_length = int(parent["actual_token_length"])
    payload = str(spec["payload"])

    if attack_name == RANK:
        if model is None:
            raise ValueError("Rank adaptive candidate generation requires source model.")
        settings = config["attacks"]["token_whitebox"]
        result = encode_rank_payload(
            model=model,
            tokenizer=tokenizer,
            prompt=str(parent["prompt"]),
            payload=payload,
            required_token_length=required_length,
            attack_seed=int(spec["strategy_attack_seed"]),
            top_k=int(settings["top_k"]),
            max_encoding_attempts=int(settings["max_encoding_attempts"]),
            final_candidate_search_k=int(settings["final_candidate_search_k"]),
            require_complete_ending=bool(settings["require_complete_ending"]),
            require_tokenizer_roundtrip=bool(settings["require_tokenizer_roundtrip"]),
        )
    elif attack_name == SYNONYM:
        settings = config["attacks"]["token_blackbox"]
        active_slot_ids = list(map(str, spec["active_slot_ids"]))
        mapped = transformed_codebook(
            codebook,
            active_slot_ids=active_slot_ids,
            mapping_mask=int(spec["synonym_mapping_mask"]),
        )
        result = encode_synonym_payload(
            clean_text=str(parent["text"]),
            tokenizer=tokenizer,
            payload=payload,
            codebook=mapped,
            required_token_length=required_length,
            max_payload_bits=int(spec["payload_bit_length"]),
            min_payload_bits=int(spec["payload_bit_length"]),
            require_text_change=bool(settings.get("require_text_change", True)),
            require_complete_ending=bool(settings["require_complete_ending"]),
            require_tokenizer_roundtrip=bool(settings["require_tokenizer_roundtrip"]),
            forbid_format_codepoints=bool(settings.get("forbid_format_codepoints", True)),
        )
        if list(result.channel_parameters.get("active_slot_ids", [])) != active_slot_ids:
            raise RuntimeError(
                "adaptive_synonym_active_slot_drift: "
                f"observed={result.channel_parameters.get('active_slot_ids')}, expected={active_slot_ids}"
            )
        result.channel_parameters.update(
            {
                "adaptive_mapping_mask": int(spec["synonym_mapping_mask"]),
                "base_codebook_version": codebook.version,
                "adaptive_codebook_version": mapped.version,
            }
        )
    else:
        raise ValueError(f"Unsupported adaptive attack name: {attack_name}")

    if result.payload != payload or not result.exact_match or not result.decode_success:
        raise RuntimeError("Adaptive candidate failed exact pre-defense payload invariant.")
    record = build_attack_record(
        parent_record=parent,
        result=result,
        manifest_version=str(config["metadata"]["manifest_version"]),
        experiment_stage="step10_adaptive_candidate_catalog",
        attack_seed=int(spec["strategy_attack_seed"]),
    )
    record.update(
        {
            "candidate_id": str(spec["candidate_id"]),
            "sample_id": str(spec["candidate_id"]),
            "round_id": None,
            "adaptive": True,
            "feedback_condition": "candidate_catalog",
            "strategy_index": int(spec["strategy_index"]),
            "strategy_id": str(spec["strategy_id"]),
            "strategy_attack_seed": int(spec["strategy_attack_seed"]),
            "synonym_mapping_mask": spec.get("synonym_mapping_mask"),
            "static_step7_sample_id": str(spec["static_step7_sample_id"]),
            "candidate_spec_sha256": str(spec["candidate_spec_sha256"]),
            "protocol_version": protocol,
            "generation_status": "accepted",
            "candidate_source": "step10_generated_strategy",
            "payload_rate_bpt": int(spec["payload_bit_length"]) / required_length,
        }
    )
    return record


def candidate_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["parent_sample_id"]),
            str(row["attack_name"]),
            int(row["strategy_index"]),
        )
        if key in result:
            raise ValueError(f"Duplicate adaptive candidate key: {key}")
        result[key] = row
    return result


def available_strategy_indices(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[int]]]:
    result: dict[str, dict[str, list[int]]] = {}
    for row in rows:
        parent = str(row["parent_sample_id"])
        attack = str(row["attack_name"])
        result.setdefault(parent, {}).setdefault(attack, []).append(int(row["strategy_index"]))
    for by_attack in result.values():
        for attack, indices in by_attack.items():
            by_attack[attack] = sorted(set(indices))
    return result
