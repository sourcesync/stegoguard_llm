from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any

ATTACK_NAMES = (
    "rank_encoding",
    "synonym_slot_codebook",
    "unicode_hidden_channel",
)


def _index_unique(
    records: list[dict[str, Any]],
    field: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record.get(field, ""))
        if not key:
            raise ValueError(f"{label} record lacks {field}: {record}")
        if key in index:
            raise ValueError(f"Duplicate {field} in {label}: {key}")
        index[key] = record
    return index


def _deterministic_order(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{sample_id}".encode("utf-8")).hexdigest()


def _parent_cell(clean: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(clean["source_model"]),
        str(clean["domain"]),
        int(clean["target_length"]),
    )


def _make_clean_detection_record(
    clean: dict[str, Any],
    *,
    is_common_all_three: bool,
    is_balanced_common: bool,
) -> dict[str, Any]:
    return {
        "sample_id": clean["sample_id"],
        "clean_parent_id": clean["sample_id"],
        "pair_id": clean["pair_id"],
        "lineage_id": clean["lineage_id"],
        "label": "clean",
        "y": 0,
        "attack_family": "clean",
        "attack_name": "clean",
        "source_model": clean["source_model"],
        "domain": clean["domain"],
        "topic_id": clean["topic_id"],
        "topic_seed": clean.get("topic_seed"),
        "nominal_target_length": int(clean["target_length"]),
        "parent_actual_token_length": int(clean["actual_token_length"]),
        "prompt_id": clean["prompt_id"],
        "prompt": clean["prompt"],
        "text": clean["text"],
        "payload_bit_length": 0,
        "payload_rate_bpt": 0.0,
        "is_common_all_three": bool(is_common_all_three),
        "is_balanced_common": bool(is_balanced_common),
        "dataset_role": "clean_control",
        "manifest_version": clean["manifest_version"],
    }


def _make_stego_detection_record(
    stego: dict[str, Any],
    clean: dict[str, Any],
    *,
    is_common_all_three: bool,
    is_balanced_common: bool,
) -> dict[str, Any]:
    if stego.get("parent_sample_id") != clean.get("sample_id"):
        raise ValueError(
            f"Stego parent mismatch for {stego.get('sample_id')}: "
            f"{stego.get('parent_sample_id')} != {clean.get('sample_id')}"
        )
    if stego.get("prompt") != clean.get("prompt"):
        raise ValueError(f"Prompt drift for {stego['sample_id']}")
    if stego.get("topic_id") != clean.get("topic_id"):
        raise ValueError(f"Topic drift for {stego['sample_id']}")
    if stego.get("validation_status") != "accepted":
        raise ValueError(f"Non-accepted stego record included: {stego['sample_id']}")

    return {
        "sample_id": stego["sample_id"],
        "clean_parent_id": clean["sample_id"],
        "pair_id": clean["pair_id"],
        "lineage_id": clean["lineage_id"],
        "label": "stego",
        "y": 1,
        "attack_family": stego["attack_family"],
        "attack_name": stego["attack_name"],
        "source_model": clean["source_model"],
        "domain": clean["domain"],
        "topic_id": clean["topic_id"],
        "topic_seed": clean.get("topic_seed"),
        "nominal_target_length": int(clean["target_length"]),
        "parent_actual_token_length": int(clean["actual_token_length"]),
        "prompt_id": clean["prompt_id"],
        "prompt": clean["prompt"],
        "text": stego["text"],
        "payload_bit_length": int(stego.get("payload_bit_length", 0)),
        "payload_rate_bpt": float(stego.get("payload_rate_bpt", 0.0)),
        "is_common_all_three": bool(is_common_all_three),
        "is_balanced_common": bool(is_balanced_common),
        "dataset_role": "accepted_stego",
        "manifest_version": stego["manifest_version"],
    }


def build_detection_dataset(
    *,
    clean_records: list[dict[str, Any]],
    stego_records: list[dict[str, Any]],
    seed: int,
    balanced_per_cell: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clean_by_id = _index_unique(clean_records, "sample_id", "clean")
    _index_unique(stego_records, "sample_id", "stego")

    attacks_by_parent: dict[str, set[str]] = defaultdict(set)
    stego_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for stego in stego_records:
        parent_id = str(stego.get("parent_sample_id", ""))
        if parent_id not in clean_by_id:
            raise ValueError(
                f"Stego record {stego.get('sample_id')} references missing parent {parent_id}."
            )
        attack_name = str(stego.get("attack_name", ""))
        if attack_name not in ATTACK_NAMES:
            raise ValueError(f"Unsupported attack_name in Step 8: {attack_name}")
        attacks_by_parent[parent_id].add(attack_name)
        stego_by_parent[parent_id].append(stego)

    required = set(ATTACK_NAMES)
    common_parent_ids = sorted(
        parent_id
        for parent_id, names in attacks_by_parent.items()
        if required.issubset(names)
    )

    common_by_cell: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for parent_id in common_parent_ids:
        common_by_cell[_parent_cell(clean_by_id[parent_id])].append(parent_id)

    balanced_ids: set[str] = set()
    if common_by_cell:
        minimum_available = min(len(values) for values in common_by_cell.values())
        if minimum_available < balanced_per_cell:
            raise ValueError(
                f"Requested {balanced_per_cell} balanced common parents per cell, "
                f"but the smallest cell has {minimum_available}."
            )
        for parent_ids in common_by_cell.values():
            ordered = sorted(
                parent_ids,
                key=lambda sample_id: _deterministic_order(sample_id, seed),
            )
            balanced_ids.update(ordered[:balanced_per_cell])

    common_set = set(common_parent_ids)
    output: list[dict[str, Any]] = []
    for clean in clean_records:
        parent_id = str(clean["sample_id"])
        is_common = parent_id in common_set
        is_balanced = parent_id in balanced_ids
        output.append(
            _make_clean_detection_record(
                clean,
                is_common_all_three=is_common,
                is_balanced_common=is_balanced,
            )
        )
        for stego in sorted(
            stego_by_parent.get(parent_id, []),
            key=lambda row: ATTACK_NAMES.index(str(row["attack_name"])),
        ):
            output.append(
                _make_stego_detection_record(
                    stego,
                    clean,
                    is_common_all_three=is_common,
                    is_balanced_common=is_balanced,
                )
            )

    output.sort(key=lambda row: (row["clean_parent_id"], row["y"], row["attack_name"]))
    validate_detection_dataset_records(output)

    counts_by_label = Counter(row["label"] for row in output)
    counts_by_attack = Counter(row["attack_name"] for row in output if row["y"] == 1)
    common_rows = [row for row in output if row["is_common_all_three"]]
    balanced_rows = [row for row in output if row["is_balanced_common"]]
    common_cell_counts = Counter(
        (
            row["source_model"],
            row["domain"],
            int(row["nominal_target_length"]),
        )
        for row in output
        if row["label"] == "clean" and row["is_common_all_three"]
    )

    manifest = {
        "protocol_version": "step8_static_v1",
        "seed": int(seed),
        "record_count": len(output),
        "clean_record_count": int(counts_by_label["clean"]),
        "stego_record_count": int(counts_by_label["stego"]),
        "counts_by_attack": dict(sorted(counts_by_attack.items())),
        "common_parent_count_all_three": len(common_parent_ids),
        "common_population_record_count": len(common_rows),
        "balanced_common_parents_per_cell": int(balanced_per_cell),
        "balanced_common_parent_count": len(balanced_ids),
        "balanced_common_population_record_count": len(balanced_rows),
        "common_parent_counts_by_model_domain_length": {
            " | ".join(map(str, key)): int(value)
            for key, value in sorted(common_cell_counts.items(), key=lambda item: str(item[0]))
        },
        "allowed_detector_inputs": ["prompt", "text"],
        "analysis_only_metadata": [
            "label",
            "attack_family",
            "attack_name",
            "source_model",
            "domain",
            "topic_id",
            "nominal_target_length",
            "payload_bit_length",
            "payload_rate_bpt",
        ],
        "forbidden_predictive_fields": [
            "label",
            "y",
            "attack_family",
            "attack_name",
            "clean_parent_id",
            "pair_id",
            "source_model",
            "domain",
            "topic_id",
            "nominal_target_length",
            "payload_bit_length",
            "payload_rate_bpt",
        ],
    }
    return output, manifest


def validate_detection_dataset_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("Detection dataset is empty.")
    ids: set[str] = set()
    clean_ids: set[str] = set()
    for row in records:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError("Detection row lacks sample_id.")
        if sample_id in ids:
            raise ValueError(f"Duplicate detection sample_id: {sample_id}")
        ids.add(sample_id)
        if row.get("label") not in {"clean", "stego"}:
            raise ValueError(f"Invalid detection label for {sample_id}: {row.get('label')}")
        expected_y = 0 if row["label"] == "clean" else 1
        if int(row.get("y", -1)) != expected_y:
            raise ValueError(f"Binary label mismatch for {sample_id}.")
        if not str(row.get("prompt", "")).strip():
            raise ValueError(f"Empty prompt for {sample_id}.")
        if not str(row.get("text", "")).strip():
            raise ValueError(f"Empty text for {sample_id}.")
        if not str(row.get("clean_parent_id", "")):
            raise ValueError(f"Missing clean_parent_id for {sample_id}.")
        if row["label"] == "clean":
            if sample_id != row["clean_parent_id"]:
                raise ValueError(f"Clean row parent mismatch for {sample_id}.")
            clean_ids.add(sample_id)
        elif row.get("attack_name") not in ATTACK_NAMES:
            raise ValueError(f"Invalid attack_name for stego row {sample_id}.")

    for row in records:
        if row["clean_parent_id"] not in clean_ids:
            raise ValueError(
                f"Detection row {row['sample_id']} references unavailable clean parent "
                f"{row['clean_parent_id']}."
            )
