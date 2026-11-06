from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from stegoguard_lm.detection.io_utils import read_jsonl


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping.")
    return value


def selected(task: dict[str, Any], population: str) -> bool:
    if population == "all":
        return True
    if population == "primary_common_all":
        return bool(task["is_primary_common_all"])
    if population == "balanced_common":
        return bool(task["is_balanced_common"])
    raise ValueError(population)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Step 9 security/utility evaluation coverage."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--population",
        choices=["all", "primary_common_all", "balanced_common"],
        default="primary_common_all",
    )
    parser.add_argument("--require-gated", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    tasks = [
        row
        for row in read_jsonl(paths["mitigation_task_manifest_jsonl"])
        if selected(row, args.population)
    ]
    outputs = (
        read_jsonl(paths["mitigation_outputs_jsonl"])
        if Path(paths["mitigation_outputs_jsonl"]).exists()
        else []
    )
    output_by_id = {
        str(row["task_id"]): row for row in outputs if selected(row, args.population)
    }
    failures = (
        read_jsonl(paths["mitigation_failures_jsonl"])
        if Path(paths["mitigation_failures_jsonl"]).exists()
        else []
    )
    failure_ids = {str(row["task_id"]) for row in failures}
    payload = (
        read_jsonl(paths["mitigation_payload_results_jsonl"])
        if Path(paths["mitigation_payload_results_jsonl"]).exists()
        else []
    )
    payload_ids = {str(row["task_id"]) for row in payload}
    utility = (
        read_jsonl(paths["mitigation_utility_jsonl"])
        if Path(paths["mitigation_utility_jsonl"]).exists()
        else []
    )
    utility_ids = {str(row["task_id"]) for row in utility}
    errors: list[str] = []
    task_ids = {str(row["task_id"]) for row in tasks}
    completed = set(output_by_id) | (failure_ids & task_ids)
    pending = task_ids - completed
    if pending:
        errors.append(f"{len(pending)} selected mitigation tasks remain pending")
    accepted_stego_ids = {
        task_id for task_id, row in output_by_id.items() if row["label"] == "stego"
    }
    missing_payload = accepted_stego_ids - payload_ids
    if missing_payload:
        errors.append(
            f"{len(missing_payload)} accepted stego outputs lack payload evaluation"
        )
    missing_utility = set(output_by_id) - utility_ids
    if missing_utility:
        errors.append(
            f"{len(missing_utility)} accepted mitigation outputs lack utility evaluation"
        )
    gated_count = 0
    if args.require_gated:
        gated_path = Path(paths["mitigation_gated_results_jsonl"])
        if not gated_path.exists():
            errors.append("Gated results file is missing")
        else:
            gated = read_jsonl(gated_path)
            gated_count = len(gated)
            frozen_seeds = {
                int(value)
                for value in config["mitigation_evaluation"]["randomized_rewrite"][
                    "seeds"
                ]
            }

        for row in gated:
            if "requested_rewrite_seed" not in row or "applied_rewrite_seed" not in row:
                errors.append(
                    "Gated result is missing "
                    "requested_rewrite_seed/applied_rewrite_seed provenance fields"
                )
                break

            policy = str(row["rewrite_policy"])
            route = str(row["intervention_route"])

            requested_seed = row["requested_rewrite_seed"]
            applied_seed = row["applied_rewrite_seed"]

            if policy == "randomized_rewrite":
                if requested_seed is None or int(requested_seed) not in frozen_seeds:
                    errors.append(
                        "Invalid requested randomized rewrite seed "
                        f"in gated result: {requested_seed}"
                    )
                    break

                if route == "linguistic_rewrite":
                    if applied_seed is None or int(applied_seed) != int(requested_seed):
                        errors.append(
                            "Gated randomized linguistic rewrite has "
                            "mismatched requested/applied seed: "
                            f"requested={requested_seed}, "
                            f"applied={applied_seed}"
                        )
                        break

                elif applied_seed is not None:
                    errors.append(
                        "Gated pass-through/Unicode-sanitization row "
                        "should not report an applied rewrite seed"
                    )
                    break

            else:
                if requested_seed is not None or applied_seed is not None:
                    errors.append(
                        "Deterministic gated policy should have null "
                        "requested/applied rewrite seeds"
                    )
                    break
            if args.population == "primary_common_all":
                seed_count = len(frozen_seeds)
                expected = (
                    int(config["mitigation_evaluation"]["expected_primary_records"])
                    * 2
                    * (1 + seed_count)
                )
                if gated_count != expected:
                    errors.append(
                        f"Expected {expected} gated primary rows, found {gated_count}"
                    )
    if errors:
        print("Step 9 mitigation result validation failed.")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Step 9 mitigation result validation passed.")
    print(f"Population: {args.population}")
    print(f"Selected tasks: {len(tasks)}")
    print(f"Accepted outputs: {len(output_by_id)}")
    print(f"Terminal generation failures: {len(failure_ids & task_ids)}")
    print(f"Stego payload evaluations: {len(accepted_stego_ids)}")
    print(f"Utility evaluations: {len(output_by_id)}")
    if args.require_gated:
        print(f"Gated rows: {gated_count}")


if __name__ == "__main__":
    main()
