from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from stegoguard_lm.detection.io_utils import canonical_sha256, read_jsonl

from .io import derive_int, verify_freeze_manifest
from .strategy import RANK, SYNONYM, assign_initial_channels, rank_strategy_seed, synonym_strategy_masks


def build_manifests(config: dict[str, Any], project_root: str | Path = ".") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(project_root)
    verify_freeze_manifest(root, "outputs/pilot_v0/manifests/step8_static_detection_manifest.json")
    verify_freeze_manifest(root, "outputs/pilot_v0/manifests/step9_mitigation_freeze_manifest.json")

    paths = config["paths"]
    cfg = config["adaptive_evaluation"]
    dataset = read_jsonl(root / paths["detection_dataset_jsonl"])
    attacks = read_jsonl(root / paths["attack_full_samples_jsonl"])
    attack_by_key = {(str(r["parent_sample_id"]), str(r["attack_name"])): r for r in attacks}

    parents = [
        row for row in dataset
        if row.get("label") == "clean" and bool(row.get("is_balanced_common"))
    ]
    parents = sorted(parents, key=lambda row: str(row["sample_id"]))
    if len(parents) != int(cfg["expected_parent_count"]):
        raise ValueError(f"Expected {cfg['expected_parent_count']} balanced clean parents; found {len(parents)}")

    initial = assign_initial_channels(parents)
    budget = int(cfg["candidate_budget_per_family"])
    protocol = str(cfg["protocol_version"])
    candidate_rows: list[dict[str, Any]] = []
    counts_by_parent: dict[str, dict[str, int]] = {}
    for parent in parents:
        parent_id = str(parent["sample_id"])
        counts_by_parent[parent_id] = {}
        for attack_name in (RANK, SYNONYM):
            static = attack_by_key.get((parent_id, attack_name))
            if static is None:
                raise ValueError(f"Balanced parent lacks frozen Step 7 {attack_name}: {parent_id}")
            payload = str(static["payload"])
            if attack_name == RANK:
                strategies = list(range(budget))
            else:
                strategies = synonym_strategy_masks(int(static["payload_bit_length"]), budget)
            counts_by_parent[parent_id][attack_name] = len(strategies)
            for strategy_index in strategies:
                static_seed = int(static.get("attack_seed") or 0)
                strategy_seed = (
                    rank_strategy_seed(
                        protocol=protocol,
                        parent_id=parent_id,
                        static_seed=static_seed,
                        strategy_index=int(strategy_index),
                    ) if attack_name == RANK else static_seed
                )
                row = {
                    "candidate_id": f"adaptive_candidate::{attack_name}::{parent_id}::s{int(strategy_index):02d}",
                    "clean_parent_id": parent_id,
                    "pair_id": str(parent["pair_id"]),
                    "source_model": str(parent["source_model"]),
                    "domain": str(parent["domain"]),
                    "topic_id": str(parent["topic_id"]),
                    "nominal_target_length": int(parent["nominal_target_length"]),
                    "parent_actual_token_length": int(parent["parent_actual_token_length"]),
                    "prompt": str(parent["prompt"]),
                    "attack_name": attack_name,
                    "strategy_index": int(strategy_index),
                    "strategy_id": (
                        f"rank_key_{int(strategy_index):02d}" if attack_name == RANK
                        else f"synonym_map_{int(strategy_index):02d}"
                    ),
                    "payload": payload,
                    "payload_bit_length": int(static["payload_bit_length"]),
                    "payload_sha256": str(static["payload_sha256"]),
                    "static_step7_sample_id": str(static["sample_id"]),
                    "static_step7_attack_seed": static_seed,
                    "strategy_attack_seed": int(strategy_seed),
                    "synonym_mapping_mask": int(strategy_index) if attack_name == SYNONYM else None,
                    "active_slot_ids": list(static.get("channel_parameters", {}).get("active_slot_ids", [])),
                    "protocol_version": protocol,
                }
                row["candidate_spec_sha256"] = canonical_sha256(row)
                candidate_rows.append(row)

    trajectory_rows: list[dict[str, Any]] = []
    for parent in parents:
        parent_id = str(parent["sample_id"])
        for attacker_condition in cfg["attacker_conditions"]:
            for defender_policy in cfg["defender_policies"]:
                trajectory_id = f"traj::{parent_id}::{attacker_condition}::{defender_policy}"
                seed_schedule = {
                    str(round_id): (
                        derive_int(
                            protocol,
                            trajectory_id,
                            int(round_id),
                            int(cfg["randomized_defender"]["master_seed"]),
                        ) if defender_policy == "randomized_rewrite" else None
                    )
                    for round_id in cfg["round_ids"]
                }
                row = {
                    "trajectory_id": trajectory_id,
                    "clean_parent_id": parent_id,
                    "pair_id": str(parent["pair_id"]),
                    "source_model": str(parent["source_model"]),
                    "domain": str(parent["domain"]),
                    "topic_id": str(parent["topic_id"]),
                    "nominal_target_length": int(parent["nominal_target_length"]),
                    "initial_attack_name": initial[parent_id],
                    "attacker_condition": str(attacker_condition),
                    "defender_policy": str(defender_policy),
                    "round_ids": list(map(int, cfg["round_ids"])),
                    "candidate_counts": counts_by_parent[parent_id],
                    "rewrite_seed_schedule": seed_schedule,
                    "feedback_fields": list(cfg["feedback_fields"]),
                    "protocol_version": protocol,
                }
                row["trajectory_spec_sha256"] = canonical_sha256(row)
                trajectory_rows.append(row)

    expected_trajectories = int(cfg["expected_parent_count"]) * len(cfg["attacker_conditions"]) * len(cfg["defender_policies"])
    if len(trajectory_rows) != expected_trajectories:
        raise ValueError(f"Expected {expected_trajectories} trajectories; found {len(trajectory_rows)}")
    return candidate_rows, trajectory_rows


def validate_manifests(config: dict[str, Any], candidate_rows: list[dict[str, Any]], trajectory_rows: list[dict[str, Any]]) -> list[str]:
    cfg = config["adaptive_evaluation"]
    errors: list[str] = []
    candidate_ids = [str(row["candidate_id"]) for row in candidate_rows]
    if len(set(candidate_ids)) != len(candidate_ids):
        errors.append("Duplicate candidate_id detected.")
    trajectory_ids = [str(row["trajectory_id"]) for row in trajectory_rows]
    if len(set(trajectory_ids)) != len(trajectory_ids):
        errors.append("Duplicate trajectory_id detected.")

    parent_ids = sorted({str(row["clean_parent_id"]) for row in trajectory_rows})
    if len(parent_ids) != int(cfg["expected_parent_count"]):
        errors.append(f"Expected {cfg['expected_parent_count']} trajectory parents; found {len(parent_ids)}")

    starts = Counter(str(row["initial_attack_name"]) for row in trajectory_rows)
    blocks = len(cfg["attacker_conditions"]) * len(cfg["defender_policies"])
    if starts[RANK] != 24 * blocks or starts[SYNONYM] != 24 * blocks:
        errors.append(f"Initial-channel balance mismatch: {dict(starts)}")

    for row in candidate_rows:
        if row["protocol_version"] != cfg["protocol_version"]:
            errors.append(f"Candidate protocol mismatch: {row['candidate_id']}")
        if row["attack_name"] == RANK and not (0 <= int(row["strategy_index"]) < int(cfg["candidate_budget_per_family"])):
            errors.append(f"Rank strategy out of budget: {row['candidate_id']}")
        if row["attack_name"] == SYNONYM:
            bits = int(row["payload_bit_length"])
            if int(row["strategy_index"]) >= min(int(cfg["candidate_budget_per_family"]), 1 << bits):
                errors.append(f"Synonym strategy exceeds unique mapping space: {row['candidate_id']}")

    for row in trajectory_rows:
        if list(map(int, row["round_ids"])) != list(map(int, cfg["round_ids"])):
            errors.append(f"Round schedule mismatch: {row['trajectory_id']}")
        if row["defender_policy"] == "randomized_rewrite":
            seeds = [row["rewrite_seed_schedule"][str(r)] for r in cfg["round_ids"]]
            if any(seed is None for seed in seeds) or len(set(map(int, seeds))) != len(seeds):
                errors.append(f"Invalid randomized seed schedule: {row['trajectory_id']}")
    return errors
