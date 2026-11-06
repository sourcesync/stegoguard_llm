from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from stegoguard_lm.detection.io_utils import atomic_write_jsonl, read_jsonl
from stegoguard_lm.mitigation.gated import (
    index_primary_full_fusion_predictions,
    intervention_decision,
    selected_oracle_task_id,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping.")
    return value


def baseline_payload(original_attack: dict[str, Any] | None) -> dict[str, Any]:
    if original_attack is None:
        return {}
    bits = int(original_attack["payload_bit_length"])
    return {
        "decoder_success": True,
        "exact_payload_recovery": True,
        "effective_bit_accuracy": 1.0,
        "effective_bit_error_rate": 0.0,
        "residual_capacity_bits": bits,
        "residual_capacity_fraction": 1.0,
    }


def baseline_utility() -> dict[str, Any]:
    return {
        "semantic_similarity_original_to_mitigated": 1.0,
        "visible_instruction_retention_delta": 0.0,
        "mean_logp_delta": 0.0,
        "ppl_delta": 0.0,
        "length_ratio": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate detector-gated Step 9 deployment using frozen Step 8 predictions."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    output_path = Path(paths["mitigation_gated_results_jsonl"])
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Gated results already exist: {output_path}. Use --overwrite intentionally."
        )

    dataset = read_jsonl(paths["detection_dataset_jsonl"])
    primary = [row for row in dataset if bool(row["is_common_all_three"])]
    predictions = index_primary_full_fusion_predictions(
        read_jsonl(paths["detection_predictions_jsonl"])
    )
    unicode_by_id = {
        str(row["sample_id"]): row
        for row in read_jsonl(paths["detection_unicode_features_jsonl"])
    }
    mitigation_outputs = {
        str(row["task_id"]): row
        for row in read_jsonl(paths["mitigation_outputs_jsonl"])
    }
    payload_results = {
        str(row["task_id"]): row
        for row in read_jsonl(paths["mitigation_payload_results_jsonl"])
    }
    utility_results = {
        str(row["task_id"]): row
        for row in read_jsonl(paths["mitigation_utility_jsonl"])
    }
    attacks = {
        str(row["sample_id"]): row
        for row in read_jsonl(paths["attack_full_samples_jsonl"])
    }
    seeds = [
        int(value)
        for value in config["mitigation_evaluation"]["randomized_rewrite"]["seeds"]
    ]
    rows: list[dict[str, Any]] = []
    for original in primary:
        sample_id = str(original["sample_id"])
        prediction = predictions[sample_id]
        unicode_feature = unicode_by_id[sample_id]
        original_attack = attacks.get(sample_id)
        for operating_point in ("f1", "fpr_01"):
            decision = intervention_decision(
                sample_id=sample_id,
                prediction_row=prediction,
                unicode_feature_row=unicode_feature,
                operating_point=operating_point,
            )
            policies: list[tuple[str, int | None]] = [("deterministic_rewrite", None)]
            policies.extend(("randomized_rewrite", seed) for seed in seeds)
            for rewrite_policy, seed in policies:
                task_id = selected_oracle_task_id(
                    sample_id=sample_id,
                    route=decision["route"],
                    rewrite_policy=rewrite_policy,
                    random_seed=seed,
                )
                applied_rewrite_seed: int | None = None

                if task_id is None:
                    payload = baseline_payload(original_attack)
                    utility = baseline_utility()
                    final_source = "original_pass_through"
                else:
                    if task_id not in mitigation_outputs:
                        raise ValueError(
                            f"Required mitigation output missing for gated simulation: {task_id}"
                        )

                    mitigation_output = mitigation_outputs[task_id]

                    applied_rewrite_seed = mitigation_output.get("rewrite_seed")
                    if applied_rewrite_seed is not None:
                        applied_rewrite_seed = int(applied_rewrite_seed)

                    if (
                        decision["route"] == "linguistic_rewrite"
                        and rewrite_policy == "randomized_rewrite"
                        and applied_rewrite_seed != seed
                    ):
                        raise ValueError(
                            "Randomized rewrite seed mismatch for gated simulation: "
                            f"requested={seed}, "
                            f"applied={applied_rewrite_seed}, "
                            f"task_id={task_id}"
                        )

                    payload = (
                        payload_results.get(task_id, {}) if original_attack else {}
                    )

                    if original_attack and task_id not in payload_results:
                        raise ValueError(
                            f"Required payload result missing for gated simulation: {task_id}"
                        )

                    utility = utility_results.get(task_id)
                    if utility is None:
                        raise ValueError(
                            f"Required utility result missing for gated simulation: {task_id}"
                        )

                    final_source = task_id
                rows.append(
                    {
                        "sample_id": sample_id,
                        "clean_parent_id": str(original["clean_parent_id"]),
                        "label": str(original["label"]),
                        "attack_name": str(original["attack_name"]),
                        "source_model": str(original["source_model"]),
                        "domain": str(original["domain"]),
                        "topic_id": str(original["topic_id"]),
                        "nominal_target_length": int(original["nominal_target_length"]),
                        "operating_point": operating_point,
                        "rewrite_policy": rewrite_policy,
                        "intervention_applied": bool(decision["intervene"]),
                        "intervention_route": str(decision["route"]),
                        "decision_reason": str(decision["decision_reason"]),
                        "final_output_source": final_source,
                        **payload,
                        **utility,
                        # In gated results, rewrite_seed remains a backward-compatible alias
                        # for the requested policy seed. The actual applied rewrite seed is
                        # recorded separately. It is None when the response passed through
                        # unchanged or when Unicode sanitation intercepted the response.
                        "rewrite_seed": seed,
                        "requested_rewrite_seed": seed,
                        "applied_rewrite_seed": applied_rewrite_seed,
                        "protocol_version": config["mitigation_evaluation"][
                            "protocol_version"
                        ],
                    }
                )
    atomic_write_jsonl(rows, output_path)
    print("Step 9 detector-gated simulation complete.")
    print(f"Rows: {len(rows)}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
