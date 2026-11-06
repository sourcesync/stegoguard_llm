from __future__ import annotations

import argparse
from pathlib import Path

from stegoguard_lm.adaptive.io import load_yaml, round_paths, verify_step10_protocol
from stegoguard_lm.detection.io_utils import atomic_write_json, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one finalized Step 10 adaptive round.")
    parser.add_argument("--config", default="config_step10.yaml")
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    verify_step10_protocol(config)
    cfg = config["adaptive_evaluation"]
    rp = round_paths(config, args.round)
    tasks = read_jsonl(rp["tasks"])
    outputs = read_jsonl(rp["outputs"])
    feedback = read_jsonl(rp["feedback"])
    utility = read_jsonl(rp["utility"])
    payload = read_jsonl(rp["payload"])
    failures = read_jsonl(rp["failures"]) if rp["failures"].exists() else []

    errors = []
    expected_stego = int(cfg["expected_parent_count"]) * len(cfg["attacker_conditions"]) * len(cfg["defender_policies"])
    expected_clean = int(cfg["expected_parent_count"]) * len(cfg["defender_policies"])
    expected_total = expected_stego + expected_clean
    if len(tasks) != expected_total:
        errors.append(f"Expected {expected_total} tasks; found {len(tasks)}")
    for label, rows in (("outputs", outputs), ("utility", utility), ("feedback", feedback)):
        ids = [str(r["task_id"]) for r in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"Duplicate task IDs in {label}")
        if len(rows) != expected_total:
            errors.append(f"Expected {expected_total} {label} rows; found {len(rows)}")
    stego_feedback = [r for r in feedback if r["record_role"] == "stego_trajectory"]
    clean_feedback = [r for r in feedback if r["record_role"] == "clean_control"]
    if len(stego_feedback) != expected_stego:
        errors.append(f"Expected {expected_stego} stego feedback rows; found {len(stego_feedback)}")
    if len(clean_feedback) != expected_clean:
        errors.append(f"Expected {expected_clean} clean feedback rows; found {len(clean_feedback)}")
    if len(payload) != expected_stego:
        errors.append(f"Expected {expected_stego} payload rows; found {len(payload)}")
    if failures:
        errors.append(f"Terminal defense-generation failures exist: {len(failures)}")
    required_feedback_keys = set(cfg["feedback_fields"])
    forbidden = set(cfg["feedback_excludes"])
    for row in stego_feedback:
        fb = row.get("attacker_feedback")
        if not isinstance(fb, dict) or set(fb) != required_feedback_keys:
            errors.append(f"Attacker feedback schema mismatch: {row['task_id']}")
            continue
        if forbidden & set(fb):
            errors.append(f"Forbidden feedback leaked to attacker: {row['task_id']}")
        if int(row["round_id"]) != args.round:
            errors.append(f"Round ID mismatch: {row['task_id']}")
        if row["attacker_condition"] == "frozen_no_feedback":
            if row["attack_name"] != row["initial_attack_name"] or int(row["strategy_index"]) != 0:
                errors.append(f"Frozen attacker changed strategy/family: {row['task_id']}")
        if row["attacker_condition"] == "fixed_channel_feedback" and row["attack_name"] != row["initial_attack_name"]:
            errors.append(f"Fixed-channel attacker migrated: {row['task_id']}")
    if errors:
        print(f"Step 10 round {args.round} validation failed.")
        for error in errors[:100]:
            print(f"- {error}")
        raise SystemExit(1)
    summary = {
        "protocol_version": cfg["protocol_version"],
        "round_id": int(args.round),
        "status": "validated",
        "total_tasks": expected_total,
        "stego_trajectories": expected_stego,
        "clean_controls": expected_clean,
        "defense_failures": 0,
        "successful_covert_bits": sum(int(r["successful_covert_bits"]) for r in stego_feedback),
        "exact_payload_survival_count": sum(int(bool(r["exact_payload_recovery"])) for r in stego_feedback),
        "migration_count": sum(int(bool(r["channel_migrated_from_prior_round"])) for r in stego_feedback),
        "clean_intervention_count": sum(int(bool(r["detected"])) for r in clean_feedback),
    }
    atomic_write_json(summary, rp["validation"])
    print(f"Step 10 round {args.round} validation passed.")
    print(summary)


if __name__ == "__main__":
    main()
