from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from stegoguard_lm.detection.io_utils import canonical_sha256, read_json, read_jsonl, sha256_file

STEP9_PROTOCOL_VERSION = "step9_mitigation_v1"
CONDITION_ORDER = (
    "unicode_sanitization",
    "deterministic_rewrite",
    "randomized_rewrite",
)

STEP9_IMPLEMENTATION_PATHS = (
    "src/stegoguard_lm/mitigation/normalization.py",
    "src/stegoguard_lm/mitigation/rewriting.py",
    "src/stegoguard_lm/mitigation/tasks.py",
    "src/stegoguard_lm/mitigation/payload_eval.py",
    "src/stegoguard_lm/mitigation/utility.py",
    "src/stegoguard_lm/mitigation/gated.py",
    "src/stegoguard_lm/mitigation/metrics.py",
    "scripts/build_mitigation_manifest.py",
    "scripts/validate_mitigation_manifest.py",
    "scripts/generate_mitigated_outputs.py",
    "scripts/validate_mitigated_outputs.py",
    "scripts/validate_mitigation_results.py",
    "scripts/evaluate_mitigation_payloads.py",
    "scripts/extract_mitigation_utility.py",
    "scripts/run_gated_mitigation.py",
    "scripts/summarize_mitigation.py",
    "scripts/freeze_mitigation.py",
)


def _rewriter_model_id(config: dict[str, Any]) -> str:
    role = str(config["mitigation_evaluation"]["rewriter_role"])
    for model in config["models"]["evaluators"]:
        if str(model.get("role")) == role:
            return str(model["id"])
    raise ValueError(f"No evaluator model is configured for rewriter role {role!r}.")


def mitigation_protocol_subset(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": STEP9_PROTOCOL_VERSION,
        "manifest_version": config["metadata"]["manifest_version"],
        "mitigation_evaluation": config["mitigation_evaluation"],
        "rewriter_model_id": _rewriter_model_id(config),
        "unicode_attack_symbols": {
            key: config["attacks"]["out_of_band"][key]
            for key in (
                "bit_zero_codepoint",
                "bit_one_codepoint",
                "prefix_codepoint",
                "suffix_codepoint",
            )
        },
        "synonym_codebook": config["paths"]["synonym_codebook_yaml"],
        "rank_decoder": {
            key: config["attacks"]["token_whitebox"][key]
            for key in ("top_k", "final_candidate_search_k", "require_complete_ending")
        },
    }


def mitigation_protocol_sha256(config: dict[str, Any]) -> str:
    return canonical_sha256(mitigation_protocol_subset(config))


def verify_step8_freeze(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    paths = config["paths"]
    manifest = read_json(paths["detection_freeze_manifest_json"])
    if manifest.get("status") != "frozen":
        raise ValueError("Step 8 must be frozen before Step 9 begins.")
    if manifest.get("protocol_version") != "step8_static_v1":
        raise ValueError("Unexpected Step 8 protocol version.")

    for relative, expected in manifest.get("artifact_hashes", {}).items():
        observed = sha256_file(Path(project_root) / relative)
        if observed != expected:
            raise ValueError(
                f"Frozen Step 8 artifact hash mismatch for {relative}: "
                f"observed={observed}, expected={expected}."
            )
    for relative, expected in manifest.get("code_hashes", {}).items():
        observed = sha256_file(Path(project_root) / relative)
        if observed != expected:
            raise ValueError(
                f"Frozen Step 8 code hash mismatch for {relative}: "
                f"observed={observed}, expected={expected}."
            )

    frozen_config_path = Path(project_root) / paths["step8_frozen_config_yaml"]
    observed_config_hash = sha256_file(frozen_config_path)
    expected_config_hash = str(manifest.get("config_sha256", ""))
    if observed_config_hash != expected_config_hash:
        raise ValueError(
            "Preserved Step 8 frozen config hash mismatch: "
            f"observed={observed_config_hash}, expected={expected_config_hash}."
        )
    return manifest


def step9_implementation_hashes(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root)
    result: dict[str, str] = {}
    for relative in STEP9_IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"Step 9 implementation file not found: {path}")
        result[relative] = sha256_file(path)
    return result


def _task_id(sample_id: str, condition: str, seed: int | None) -> str:
    suffix = "fixed" if seed is None else f"seed_{seed}"
    return f"{condition}::{suffix}::{sample_id}"


def build_mitigation_tasks(
    *,
    config: dict[str, Any],
    detection_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cfg = config["mitigation_evaluation"]
    seeds = [int(value) for value in cfg["randomized_rewrite"]["seeds"]]
    tasks: list[dict[str, Any]] = []
    index = 0
    for row in sorted(detection_records, key=lambda value: str(value["sample_id"])):
        base = {
            "original_sample_id": str(row["sample_id"]),
            "clean_parent_id": str(row["clean_parent_id"]),
            "pair_id": str(row["pair_id"]),
            "label": str(row["label"]),
            "attack_family": str(row["attack_family"]),
            "attack_name": str(row["attack_name"]),
            "source_model": str(row["source_model"]),
            "domain": str(row["domain"]),
            "topic_id": str(row["topic_id"]),
            "nominal_target_length": int(row["nominal_target_length"]),
            "payload_bit_length": int(row["payload_bit_length"]),
            "payload_rate_bpt": float(row["payload_rate_bpt"]),
            "is_primary_common_all": bool(row["is_common_all_three"]),
            "is_balanced_common": bool(row["is_balanced_common"]),
            "protocol_version": STEP9_PROTOCOL_VERSION,
            "manifest_version": config["metadata"]["manifest_version"],
        }
        for condition, seed_values in (
            ("unicode_sanitization", [None]),
            ("deterministic_rewrite", [None]),
            ("randomized_rewrite", seeds),
        ):
            for seed in seed_values:
                task = dict(base)
                task.update(
                    {
                        "task_index": index,
                        "task_id": _task_id(str(row["sample_id"]), condition, seed),
                        "condition": condition,
                        "rewrite_seed": seed,
                        "rewriter_model_id": None
                        if condition == "unicode_sanitization"
                        else _rewriter_model_id(config),
                        "status": "pending",
                    }
                )
                tasks.append(task)
                index += 1
    return tasks


def validate_mitigation_tasks(
    *,
    tasks: list[dict[str, Any]],
    config: dict[str, Any],
    detection_records: list[dict[str, Any]],
) -> dict[str, Any]:
    cfg = config["mitigation_evaluation"]
    expected_records = int(cfg["expected_detection_records"])
    expected_seeds = len(cfg["randomized_rewrite"]["seeds"])
    expected_tasks = expected_records * (2 + expected_seeds)
    if len(detection_records) != expected_records:
        raise ValueError(
            f"Expected {expected_records} detection records, found {len(detection_records)}."
        )
    if len(tasks) != expected_tasks:
        raise ValueError(f"Expected {expected_tasks} Step 9 tasks, found {len(tasks)}.")

    ids = [str(task.get("task_id", "")) for task in tasks]
    if any(not value for value in ids):
        raise ValueError("A Step 9 task has an empty task_id.")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate Step 9 task_id values detected.")

    sample_ids = {str(row["sample_id"]) for row in detection_records}
    grouped: Counter[str] = Counter()
    primary_tasks = 0
    for task in tasks:
        if task["original_sample_id"] not in sample_ids:
            raise ValueError(f"Unknown original_sample_id: {task['original_sample_id']}")
        condition = str(task["condition"])
        if condition not in CONDITION_ORDER:
            raise ValueError(f"Unknown mitigation condition: {condition}")
        grouped[condition] += 1
        if bool(task["is_primary_common_all"]):
            primary_tasks += 1
        if condition == "randomized_rewrite" and task["rewrite_seed"] is None:
            raise ValueError("Randomized rewrite task lacks a seed.")
        if condition != "randomized_rewrite" and task["rewrite_seed"] is not None:
            raise ValueError("Non-randomized mitigation task unexpectedly has a seed.")

    expected_primary_records = int(cfg["expected_primary_records"])
    expected_primary_tasks = expected_primary_records * (2 + expected_seeds)
    if primary_tasks != expected_primary_tasks:
        raise ValueError(
            f"Expected {expected_primary_tasks} primary Step 9 tasks, found {primary_tasks}."
        )
    return {
        "detection_records": len(detection_records),
        "task_count": len(tasks),
        "primary_task_count": primary_tasks,
        "counts_by_condition": dict(sorted(grouped.items())),
        "randomized_seed_count": expected_seeds,
    }


def account_mitigation_tasks(
    *,
    tasks: list[dict[str, Any]],
    accepted_outputs: list[dict[str, Any]],
    terminal_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    task_ids = {str(task["task_id"]) for task in tasks}
    accepted = {str(row["task_id"]) for row in accepted_outputs}
    failed = {str(row["task_id"]) for row in terminal_failures}
    if not accepted <= task_ids or not failed <= task_ids:
        raise ValueError("Mitigation outputs/failures contain unknown task IDs.")
    overlap = accepted & failed
    if overlap:
        raise ValueError(f"Task IDs appear as both accepted and failed: {sorted(overlap)[:5]}")
    pending = sorted(task_ids - accepted - failed)
    return {
        "task_count": len(tasks),
        "accepted_tasks": len(accepted),
        "terminal_failures": len(failed),
        "pending_tasks": len(pending),
        "pending_task_ids": pending,
        "status": "complete" if not pending else "incomplete",
    }
