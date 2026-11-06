from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from stegoguard_lm.detection.io_utils import atomic_write_json, read_jsonl
from stegoguard_lm.mitigation.metrics import (
    summarize_payload_results,
    summarize_randomized_survival,
    summarize_utility_results,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping.")
    return value


def gated_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    groups: dict[tuple[str, str, int | None, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        requested_seed = row.get(
            "requested_rewrite_seed",
            row.get("rewrite_seed"),
        )

        groups[
            (
                str(row["operating_point"]),
                str(row["rewrite_policy"]),
                requested_seed,
                str(row["attack_name"]),
            )
        ].append(row)
    for key_tuple, subset in sorted(groups.items(), key=lambda item: str(item[0])):
        operating_point, policy, seed, attack = key_tuple
        clean = [row for row in subset if row["label"] == "clean"]
        stego = [row for row in subset if row["label"] == "stego"]
        block: dict[str, Any] = {
            "n": len(subset),
            "intervention_rate": float(
                np.mean([float(row["intervention_applied"]) for row in subset])
            ),
        }
        if clean:
            block["clean_intervention_rate"] = float(
                np.mean([float(row["intervention_applied"]) for row in clean])
            )
        if stego:
            exact = [float(row["exact_payload_recovery"]) for row in stego]
            block["stego_exact_payload_survival_rate"] = float(np.mean(exact))
            block["stego_payload_disruption_rate"] = float(1.0 - np.mean(exact))
            block["stego_effective_bit_accuracy_mean"] = float(
                np.mean([float(row["effective_bit_accuracy"]) for row in stego])
            )
        result[f"{operating_point}::{policy}::{seed}::{attack}"] = block
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Step 9 mitigation security and utility outcomes."
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    cfg = config["mitigation_evaluation"]
    payload_rows = read_jsonl(paths["mitigation_payload_results_jsonl"])

    utility_rows = read_jsonl(paths["mitigation_utility_jsonl"])

    attack_records = read_jsonl(paths["attack_full_samples_jsonl"])

    task_rows = read_jsonl(paths["mitigation_task_manifest_jsonl"])

    gated_rows = (
        read_jsonl(paths["mitigation_gated_results_jsonl"])
        if Path(paths["mitigation_gated_results_jsonl"]).exists()
        else []
    )
    primary_payload = [
        row for row in payload_rows if bool(row["is_primary_common_all"])
    ]
    primary_utility = [
        row for row in utility_rows if bool(row["is_primary_common_all"])
    ]
    full_payload = list(payload_rows)
    full_utility = list(utility_rows)

    expected_full_payload_rows = sum(1 for row in task_rows if row["label"] == "stego")

    expected_full_utility_rows = len(task_rows)
    baseline_by_attack = {}
    for attack in sorted({row["attack_name"] for row in attack_records}):
        subset = [row for row in attack_records if row["attack_name"] == attack]
        baseline_by_attack[attack] = {
            "n": len(subset),
            "exact_payload_recovery_rate": float(
                np.mean([float(bool(row["exact_match"])) for row in subset])
            ),
            "effective_bit_accuracy": 1.0,
            "payload_disruption_rate": 0.0,
        }

    summary = {
        "protocol_version": cfg["protocol_version"],
        "baseline_no_mitigation": baseline_by_attack,
        "primary_population": "104-parent common all-three matched set",
        "primary_payload_result_rows": len(primary_payload),
        "primary_utility_result_rows": len(primary_utility),
        "oracle_payload_security_primary": summarize_payload_results(
            primary_payload,
            bootstrap_replicates=int(cfg["bootstrap_replicates"]),
            seed=int(config["seed"]),
        ),
        "randomized_survival_primary": summarize_randomized_survival(primary_payload),
        "utility_primary": summarize_utility_results(primary_utility),
        "gated_primary": (gated_summary(gated_rows) if gated_rows else {}),
        "full_inventory_population": (
            "all 583 frozen Step 8 records; "
            "channel-specific feasible inventory used as a "
            "robustness/generalization extension"
        ),
        "full_inventory_payload_result_rows": len(full_payload),
        "full_inventory_payload_expected_rows": expected_full_payload_rows,
        "full_inventory_utility_result_rows": len(full_utility),
        "full_inventory_utility_expected_rows": expected_full_utility_rows,
        "full_inventory_complete": (
            len(full_payload) == expected_full_payload_rows
            and len(full_utility) == expected_full_utility_rows
        ),
        "oracle_payload_security_full": summarize_payload_results(
            full_payload,
            bootstrap_replicates=int(cfg["bootstrap_replicates"]),
            seed=int(config["seed"]),
        ),
        "randomized_survival_full": summarize_randomized_survival(full_payload),
        "utility_full": summarize_utility_results(full_utility),
    }

    atomic_write_json(summary, paths["mitigation_summary_json"])

    qa_limit = 20
    lines = [
        "Step 9 Mitigation QA Review",
        "",
        "Highest residual payload survival among primary stego mitigation outputs",
        "",
    ]
    payload_ranked = sorted(
        primary_payload,
        key=lambda row: (
            float(bool(row["exact_payload_recovery"])),
            float(row["effective_bit_accuracy"]),
        ),
        reverse=True,
    )[:qa_limit]
    for row in payload_ranked:
        lines.append(
            f"{row['task_id']} | attack={row['attack_name']} | exact={row['exact_payload_recovery']} "
            f"| bit_accuracy={float(row['effective_bit_accuracy']):.4f} "
            f"| residual={float(row['residual_capacity_fraction']):.4f}"
        )
    lines.extend(["", "Lowest semantic preservation among primary outputs", ""])
    utility_ranked = sorted(
        primary_utility,
        key=lambda row: float(row["semantic_similarity_original_to_mitigated"]),
    )[:qa_limit]
    for row in utility_ranked:
        lines.append(
            f"{row['task_id']} | label={row['label']} | attack={row['attack_name']} "
            f"| semantic={float(row['semantic_similarity_original_to_mitigated']):.4f} "
            f"| instruction_delta={float(row['visible_instruction_retention_delta']):+.4f} "
            f"| length_ratio={float(row['length_ratio']):.4f}"
        )
    clean_ranked = sorted(
        [row for row in primary_utility if row["label"] == "clean"],
        key=lambda row: float(row["semantic_similarity_original_to_mitigated"]),
    )[:qa_limit]
    lines.extend(["", "Lowest semantic preservation among clean controls", ""])
    for row in clean_ranked:
        lines.append(
            f"{row['task_id']} | semantic={float(row['semantic_similarity_original_to_mitigated']):.4f} "
            f"| instruction_delta={float(row['visible_instruction_retention_delta']):+.4f} "
            f"| length_ratio={float(row['length_ratio']):.4f}"
        )
    Path(paths["mitigation_qa_review_txt"]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["mitigation_qa_review_txt"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("Step 9 mitigation summary rebuilt.")
    print(f"Primary payload result rows: {len(primary_payload)}")
    print(f"Primary utility result rows: {len(primary_utility)}")

    print(
        f"Full payload result rows: "
        f"{len(full_payload)} / {expected_full_payload_rows}"
    )

    print(
        f"Full utility result rows: "
        f"{len(full_utility)} / {expected_full_utility_rows}"
    )

    print(
        "Full inventory complete: "
        f"{len(full_payload) == expected_full_payload_rows and len(full_utility) == expected_full_utility_rows}"
    )

    print(f"Gated rows: {len(gated_rows)}")
    print(f"Wrote: {paths['mitigation_summary_json']}")
    print(f"Wrote: {paths['mitigation_qa_review_txt']}")


if __name__ == "__main__":
    main()
