from __future__ import annotations

from pathlib import Path
from typing import Any

from stegoguard_lm.detection.io_utils import read_jsonl

from .candidates import available_strategy_indices, candidate_index
from .io import derive_int
from .strategy import next_action


def _prior_round_feedback(config: dict[str, Any], round_id: int) -> list[dict[str, Any]]:
    if round_id <= 0:
        return []
    path = Path(config["paths"]["adaptive_output_dir"]) / f"round_{round_id - 1}" / "feedback_results.jsonl"
    return read_jsonl(path)


def build_round_tasks(
    *,
    config: dict[str, Any],
    round_id: int,
    trajectories: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    detector_scores: list[dict[str, Any]],
    prior_all_feedback: list[dict[str, Any]],
    clean_rows: list[dict[str, Any]],
    static_unicode_features: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cfg = config["adaptive_evaluation"]
    if int(round_id) not in list(map(int, cfg["round_ids"])):
        raise ValueError(f"Round {round_id} is outside frozen schedule {cfg['round_ids']}")
    cindex = candidate_index(candidates)
    available = available_strategy_indices(candidates)
    score_by_id = {str(row["sample_id"]): row for row in detector_scores}
    unicode_by_id = {str(row["sample_id"]): row for row in static_unicode_features}
    clean_by_id = {str(row["sample_id"]): row for row in clean_rows}

    feedback_by_traj: dict[str, list[dict[str, Any]]] = {}
    for row in prior_all_feedback:
        if row.get("record_role") != "stego_trajectory":
            continue
        feedback_by_traj.setdefault(str(row["trajectory_id"]), []).append(row)
    for rows in feedback_by_traj.values():
        rows.sort(key=lambda r: int(r["round_id"]))

    tasks: list[dict[str, Any]] = []
    for trajectory in trajectories:
        trajectory_id = str(trajectory["trajectory_id"])
        parent_id = str(trajectory["clean_parent_id"])
        prior_feedback = [r for r in feedback_by_traj.get(trajectory_id, []) if int(r["round_id"]) < round_id]
        if len(prior_feedback) != round_id:
            raise ValueError(
                f"Trajectory {trajectory_id} requires exactly {round_id} prior feedback rows; found {len(prior_feedback)}"
            )
        prior_actions = [
            {
                "attack_name": str(r["attack_name"]),
                "strategy_index": int(r["strategy_index"]),
            }
            for r in prior_feedback
        ]
        bounded_feedback = [
            {
                "detected": bool(r["attacker_feedback"]["detected"]),
                "intervention_route": str(r["attacker_feedback"]["intervention_route"]),
                "exact_payload_recovery": bool(r["attacker_feedback"]["exact_payload_recovery"]),
            }
            for r in prior_feedback
        ]
        family, strategy_index, reason = next_action(
            attacker_condition=str(trajectory["attacker_condition"]),
            initial_family=str(trajectory["initial_attack_name"]),
            prior_actions=prior_actions,
            prior_feedback=bounded_feedback,
            available_indices=available[parent_id],
        )
        candidate = cindex.get((parent_id, family, strategy_index))
        if candidate is None:
            raise ValueError(f"Selected adaptive candidate is unavailable: {parent_id} {family} s{strategy_index}")
        score = score_by_id.get(str(candidate["sample_id"]))
        if score is None:
            raise ValueError(f"Missing detector score for candidate: {candidate['sample_id']}")
        detected = bool(score["detected_f1"])
        route = "linguistic_rewrite" if detected else "pass_through"
        seed = trajectory["rewrite_seed_schedule"][str(round_id)]
        tasks.append(
            {
                "task_id": f"{trajectory_id}::r{round_id}",
                "record_role": "stego_trajectory",
                "trajectory_id": trajectory_id,
                "round_id": int(round_id),
                "clean_parent_id": parent_id,
                "candidate_id": str(candidate["sample_id"]),
                "attack_name": family,
                "strategy_index": int(strategy_index),
                "strategy_id": str(candidate["strategy_id"]),
                "attacker_condition": str(trajectory["attacker_condition"]),
                "initial_attack_name": str(trajectory["initial_attack_name"]),
                "defender_policy": str(trajectory["defender_policy"]),
                "selection_reason": reason,
                "detected": detected,
                "intervention_route": route,
                "rewrite_seed": int(seed) if seed is not None else None,
                "source_model": str(candidate["source_model"]),
                "domain": str(candidate["domain"]),
                "topic_id": str(candidate["topic_id"]),
                "nominal_target_length": int(candidate["nominal_target_length"]),
                "payload_bit_length": int(candidate["payload_bit_length"]),
                "payload_rate_bpt": float(candidate["payload_rate_bpt"]),
                "ensemble_mean_score": float(score["ensemble_mean_score"]),
                "flagged_fold_count": int(score["flagged_fold_count"]),
                "fold_count": int(score["fold_count"]),
                "protocol_version": str(cfg["protocol_version"]),
            }
        )

    # Clean controls are repeated per defender policy and round, but not per attacker condition.
    defender_policies = list(map(str, cfg["defender_policies"]))
    for clean_id, clean in sorted(clean_by_id.items()):
        score = score_by_id.get(clean_id)
        if score is None:
            raise ValueError(f"Missing Step 10 detector score for clean control: {clean_id}")
        unicode = unicode_by_id.get(clean_id, {})
        unicode_flag = bool(unicode.get("has_suspicious_codepoint", False))
        for defender_policy in defender_policies:
            control_id = f"clean_control::{clean_id}::{defender_policy}"
            seed = (
                None if defender_policy == "deterministic_rewrite"
                else derive_int(
                    str(cfg["protocol_version"]),
                    control_id,
                    int(round_id),
                    int(cfg["randomized_defender"]["master_seed"]),
                )
            )
            detected = bool(score["detected_f1"])
            if unicode_flag:
                route = "unicode_sanitization"
            else:
                route = "linguistic_rewrite" if detected else "pass_through"
            tasks.append(
                {
                    "task_id": f"{control_id}::r{round_id}",
                    "record_role": "clean_control",
                    "trajectory_id": control_id,
                    "round_id": int(round_id),
                    "clean_parent_id": clean_id,
                    "candidate_id": clean_id,
                    "attack_name": "clean",
                    "strategy_index": 0,
                    "strategy_id": "clean_control",
                    "attacker_condition": "none",
                    "defender_policy": defender_policy,
                    "selection_reason": "clean_control",
                    "detected": detected or unicode_flag,
                    "intervention_route": route,
                    "rewrite_seed": int(seed) if seed is not None else None,
                    "source_model": str(clean["source_model"]),
                    "domain": str(clean["domain"]),
                    "topic_id": str(clean["topic_id"]),
                    "nominal_target_length": int(clean["nominal_target_length"]),
                    "payload_bit_length": 0,
                    "payload_rate_bpt": 0.0,
                    "ensemble_mean_score": float(score["ensemble_mean_score"]),
                    "flagged_fold_count": int(score["flagged_fold_count"]),
                    "fold_count": int(score["fold_count"]),
                    "protocol_version": str(cfg["protocol_version"]),
                }
            )
    return tasks
