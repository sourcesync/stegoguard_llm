from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stegoguard_lm.adaptive.io import load_yaml, round_paths, verify_step10_protocol
from stegoguard_lm.detection.io_utils import atomic_write_json, read_jsonl


def mean_ci_by_parent(rows: list[dict[str, Any]], value_field: str, *, replicates: int, seed: int) -> dict[str, Any]:
    by_parent: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_parent[str(row["clean_parent_id"])].append(float(row[value_field]))
    parents = sorted(by_parent)
    values = np.asarray([np.mean(by_parent[parent]) for parent in parents], dtype=float)
    if not len(values):
        return {"mean": None, "ci95": [None, None], "n_parents": 0}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(int(replicates)):
        sample = rng.choice(values, size=len(values), replace=True)
        boot.append(float(np.mean(sample)))
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "n_parents": len(parents),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Step 10 adaptive trajectories and system-level outcomes.")
    parser.add_argument("--config", default="config_step10.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    verify_step10_protocol(config)
    cfg = config["adaptive_evaluation"]
    all_rows: list[dict[str, Any]] = []
    for round_id in cfg["round_ids"]:
        validation = Path(round_paths(config, int(round_id))["validation"])
        if not validation.exists():
            raise FileNotFoundError(f"Round {round_id} must be validated before summary: {validation}")
        all_rows.extend(read_jsonl(round_paths(config, int(round_id))["feedback"]))

    stego = [r for r in all_rows if r["record_role"] == "stego_trajectory"]
    clean = [r for r in all_rows if r["record_role"] == "clean_control"]
    round_groups: dict[str, Any] = {}
    replicates = int(cfg["statistics"]["bootstrap_replicates"])
    seed0 = int(config["seed"])
    for attacker_condition in cfg["attacker_conditions"]:
        for defender_policy in cfg["defender_policies"]:
            for round_id in cfg["round_ids"]:
                subset = [
                    r for r in stego
                    if r["attacker_condition"] == attacker_condition
                    and r["defender_policy"] == defender_policy
                    and int(r["round_id"]) == int(round_id)
                ]
                key = f"{attacker_condition} | {defender_policy} | R{round_id}"
                channel_counts = Counter(str(r["attack_name"]) for r in subset)
                round_groups[key] = {
                    "n": len(subset),
                    "exact_payload_survival_rate": float(np.mean([bool(r["exact_payload_recovery"]) for r in subset])) if subset else None,
                    "successful_covert_bits_total": int(sum(int(r["successful_covert_bits"]) for r in subset)),
                    "successful_covert_bits_per_parent": mean_ci_by_parent(subset, "successful_covert_bits", replicates=replicates, seed=seed0 + int(round_id) + sum(map(ord, attacker_condition + defender_policy))),
                    "mean_effective_bit_accuracy": float(np.mean([float(r["effective_bit_accuracy"]) for r in subset])) if subset else None,
                    "mean_residual_capacity_bits": float(np.mean([float(r["residual_capacity_bits"]) for r in subset])) if subset else None,
                    "intervention_rate": float(np.mean([bool(r["detected"]) for r in subset])) if subset else None,
                    "strategy_change_rate": float(np.mean([bool(r["strategy_changed_from_prior_round"]) for r in subset])) if subset else None,
                    "channel_migration_rate": float(np.mean([bool(r["channel_migrated_from_prior_round"]) for r in subset])) if subset else None,
                    "channel_counts": dict(channel_counts),
                    "channel_shares": {k: v / len(subset) for k, v in channel_counts.items()} if subset else {},
                }

    trajectory_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stego:
        trajectory_rows[str(row["trajectory_id"])].append(row)
    trajectory_metrics = []
    for trajectory_id, rows in trajectory_rows.items():
        rows.sort(key=lambda r: int(r["round_id"]))
        if len(rows) != len(cfg["round_ids"]):
            raise ValueError(f"Incomplete trajectory: {trajectory_id}")
        throughput = [int(r["successful_covert_bits"]) for r in rows]
        trajectory_metrics.append(
            {
                "trajectory_id": trajectory_id,
                "clean_parent_id": str(rows[0]["clean_parent_id"]),
                "attacker_condition": str(rows[0]["attacker_condition"]),
                "defender_policy": str(rows[0]["defender_policy"]),
                "initial_attack_name": str(rows[0]["initial_attack_name"]),
                "successful_covert_bits_by_round": throughput,
                "throughput_auc_sum": int(sum(throughput)),
                "recovery_r3_minus_r1": int(throughput[3] - throughput[1]),
                "migration_count": int(sum(bool(r["channel_migrated_from_prior_round"]) for r in rows)),
                "strategy_change_count": int(sum(bool(r["strategy_changed_from_prior_round"]) for r in rows)),
            }
        )

    clean_groups = {}
    for defender_policy in cfg["defender_policies"]:
        for round_id in cfg["round_ids"]:
            subset = [r for r in clean if r["defender_policy"] == defender_policy and int(r["round_id"]) == int(round_id)]
            clean_groups[f"{defender_policy} | R{round_id}"] = {
                "n": len(subset),
                "intervention_rate": float(np.mean([bool(r["detected"]) for r in subset])) if subset else None,
                "mean_semantic_similarity": float(np.mean([float(r["semantic_similarity_original_to_mitigated"]) for r in subset])) if subset else None,
                "mean_instruction_retention_delta": float(np.mean([float(r["visible_instruction_retention_delta"]) for r in subset])) if subset else None,
                "mean_length_ratio": float(np.mean([float(r["length_ratio"]) for r in subset])) if subset else None,
            }

    summary = {
        "protocol_version": cfg["protocol_version"],
        "status": "complete" if len(trajectory_metrics) == int(cfg["expected_parent_count"]) * len(cfg["attacker_conditions"]) * len(cfg["defender_policies"]) else "incomplete",
        "round_ids": cfg["round_ids"],
        "stego_feedback_rows": len(stego),
        "clean_control_rows": len(clean),
        "trajectory_count": len(trajectory_metrics),
        "round_condition_metrics": round_groups,
        "clean_control_metrics": clean_groups,
        "trajectory_metrics": trajectory_metrics,
        "overall_channel_transitions": dict(Counter(
            f"{prev['attack_name']}->{cur['attack_name']}"
            for rows in trajectory_rows.values()
            for prev, cur in zip(sorted(rows, key=lambda r: int(r['round_id']))[:-1], sorted(rows, key=lambda r: int(r['round_id']))[1:])
        )),
    }
    atomic_write_json(summary, config["paths"]["adaptive_summary_json"])
    qa = Path(config["paths"]["adaptive_qa_review_txt"])
    qa.parent.mkdir(parents=True, exist_ok=True)
    qa.write_text(
        "StegoGuard-LM Step 10 Adaptive QA Review\n"
        "========================================\n\n"
        f"Protocol: {cfg['protocol_version']}\n"
        f"Trajectories: {len(trajectory_metrics)}\n"
        f"Stego round rows: {len(stego)}\n"
        f"Clean control round rows: {len(clean)}\n\n"
        "Interpretation guardrails:\n"
        "- Do not claim adaptation unless feedback-enabled trajectories differ from frozen controls.\n"
        "- Do not claim channel migration unless observed transition counts support it.\n"
        "- Do not claim resilience from low one-shot survival alone; inspect the four-round trajectories and clean burden.\n"
        "- Exact payload failure is not zero information leakage; retain effective bit accuracy and residual capacity.\n",
        encoding="utf-8",
    )
    print("Step 10 adaptive summary rebuilt.")
    print(f"Trajectories: {len(trajectory_metrics)}")
    print(f"Stego feedback rows: {len(stego)}")
    print(f"Clean control rows: {len(clean)}")


if __name__ == "__main__":
    main()
