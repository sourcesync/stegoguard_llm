from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from stegoguard_lm.detection.io_utils import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl, sha256_file
from stegoguard_lm.detection.metrics import classification_metrics, cluster_bootstrap_ci
from stegoguard_lm.detection.modeling import FeatureStore, VARIANTS, run_topic_disjoint_cv

ATTACKS = ("rank_encoding", "synonym_slot_codebook", "unicode_hidden_channel")


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        key = str(row["sample_id"])
        if key in result:
            raise ValueError(f"Duplicate {label} sample_id: {key}")
        result[key] = row
    return result


def _matched_population_ids(dataset: list[dict[str, Any]], attack_name: str) -> set[str]:
    parents = {
        str(row["clean_parent_id"])
        for row in dataset
        if row["label"] == "stego" and row["attack_name"] == attack_name
    }
    return {
        str(row["sample_id"])
        for row in dataset
        if (
            row["label"] == "clean" and row["sample_id"] in parents
        ) or (
            row["label"] == "stego"
            and row["attack_name"] == attack_name
            and row["clean_parent_id"] in parents
        )
    }


def _metric_block(rows: list[dict[str, Any]], bootstrap_replicates: int, seed: int) -> dict[str, Any]:
    y = np.asarray([int(row["y"]) for row in rows], dtype=int)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=float)
    pred_f1 = np.asarray([int(row["pred_f1"]) for row in rows], dtype=int)
    pred_05 = np.asarray([int(row["pred_fpr_05"]) for row in rows], dtype=int)
    pred_01 = np.asarray([int(row["pred_fpr_01"]) for row in rows], dtype=int)
    metrics = classification_metrics(
        y_true=y,
        scores=scores,
        pred_f1=pred_f1,
        pred_fpr_05=pred_05,
        pred_fpr_01=pred_01,
    )
    bootstrap_rows = [dict(row) for row in rows]
    metrics["roc_auc_ci95_parent_cluster"] = cluster_bootstrap_ci(
        rows=bootstrap_rows,
        metric_fn=roc_auc_score,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    metrics["pr_auc_ci95_parent_cluster"] = cluster_bootstrap_ci(
        rows=bootstrap_rows,
        metric_fn=average_precision_score,
        replicates=bootstrap_replicates,
        seed=seed + 1,
    )
    return metrics


def _stratified_metrics(
    rows: list[dict[str, Any]],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    # Per attack comparisons reuse the same clean rows, which is intentional.
    for attack in ATTACKS:
        subset = [row for row in rows if row["label"] == "clean" or row["attack_name"] == attack]
        if any(row["attack_name"] == attack for row in subset):
            result[f"attack::{attack}"] = _metric_block(subset, bootstrap_replicates, seed + sum(ord(ch) for ch in attack))

    group_specs = {
        "source_model": lambda row: str(row["source_model"]),
        "domain": lambda row: str(row["domain"]),
        "nominal_length": lambda row: str(int(row["nominal_target_length"])),
    }
    for group_name, getter in group_specs.items():
        values = sorted({getter(row) for row in rows})
        for value in values:
            subset = [row for row in rows if getter(row) == value]
            if len({int(row["y"]) for row in subset}) == 2:
                result[f"{group_name}::{value}"] = _metric_block(
                    subset,
                    bootstrap_replicates,
                    seed + sum(ord(ch) for ch in f"{group_name}:{value}"),
                )

    for bits in sorted({int(row["payload_bit_length"]) for row in rows if row["attack_name"] == "synonym_slot_codebook"}):
        parents = {
            row["clean_parent_id"]
            for row in rows
            if row["attack_name"] == "synonym_slot_codebook" and int(row["payload_bit_length"]) == bits
        }
        subset = [
            row for row in rows
            if (
                row["label"] == "clean" and row["clean_parent_id"] in parents
            ) or (
                row["attack_name"] == "synonym_slot_codebook"
                and row["clean_parent_id"] in parents
            )
        ]
        if subset and len({int(row["y"]) for row in subset}) == 2:
            result[f"synonym_payload_bits::{bits}"] = _metric_block(
                subset,
                bootstrap_replicates,
                seed + 500 + bits,
            )
    return result


def _unicode_hygiene_predictions(
    dataset: list[dict[str, Any]],
    unicode_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    parents = {
        row["clean_parent_id"]
        for row in dataset
        if row["attack_name"] == "unicode_hidden_channel"
    }
    rows = [
        row for row in dataset
        if (row["label"] == "clean" and row["sample_id"] in parents)
        or row["attack_name"] == "unicode_hidden_channel"
    ]
    predictions = []
    for row in rows:
        score = float(unicode_by_id[row["sample_id"]]["has_suspicious_codepoint"])
        predictions.append(
            {
                **{key: row[key] for key in (
                    "sample_id", "clean_parent_id", "pair_id", "topic_id", "source_model",
                    "domain", "nominal_target_length", "attack_name", "label", "y",
                    "payload_bit_length", "payload_rate_bpt",
                )},
                "fold_id": "deterministic_unicode_hygiene",
                "experiment": "unicode_hygiene",
                "variant": "unicode_hygiene",
                "score": score,
                "threshold_f1": 0.5,
                "pred_f1": int(score >= 0.5),
                "threshold_fpr_05": 0.5,
                "pred_fpr_05": int(score >= 0.5),
                "threshold_fpr_01": 0.5,
                "pred_fpr_01": int(score >= 0.5),
            }
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 8 static detection with topic-disjoint evaluation.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    paths = config["paths"]
    static_cfg = config["static_detection"]
    prediction_path = Path(paths["detection_predictions_jsonl"])
    metrics_path = Path(paths["detection_metrics_json"])
    if prediction_path.exists() and not args.overwrite:
        raise FileExistsError(f"Predictions already exist: {prediction_path}. Use --overwrite intentionally.")

    feature_manifest = read_json(paths["detection_feature_manifest_json"])
    if not feature_manifest.get("complete"):
        raise ValueError("Step 8 feature manifest is not complete.")
    expected_feature_paths = {
        "evaluator_a": paths["detection_evaluator_a_features_jsonl"],
        "evaluator_b": paths["detection_evaluator_b_features_jsonl"],
        "semantic": paths["detection_semantic_features_jsonl"],
        "unicode": paths["detection_unicode_features_jsonl"],
    }
    for label, path in expected_feature_paths.items():
        observed = sha256_file(path)
        expected = feature_manifest["feature_hashes"].get(label)
        if observed != expected:
            raise ValueError(
                f"Feature hash mismatch for {label}: observed={observed}, expected={expected}."
            )

    dataset = read_jsonl(paths["detection_dataset_jsonl"])
    splits = read_json(paths["detection_splits_json"])
    store = FeatureStore(
        dataset_by_id=_index(dataset, "dataset"),
        evaluator_a_by_id=_index(read_jsonl(paths["detection_evaluator_a_features_jsonl"]), "evaluator_a"),
        evaluator_b_by_id=_index(read_jsonl(paths["detection_evaluator_b_features_jsonl"]), "evaluator_b"),
        semantic_by_id=_index(read_jsonl(paths["detection_semantic_features_jsonl"]), "semantic"),
        unicode_by_id=_index(read_jsonl(paths["detection_unicode_features_jsonl"]), "unicode"),
    )
    all_ids = set(store.dataset_by_id)
    for name, index in (
        ("evaluator_a", store.evaluator_a_by_id),
        ("evaluator_b", store.evaluator_b_by_id),
        ("semantic", store.semantic_by_id),
        ("unicode", store.unicode_by_id),
    ):
        if set(index) != all_ids:
            raise ValueError(f"{name} feature IDs do not match the detection dataset.")

    common_ids = {
        row["sample_id"] for row in dataset if bool(row["is_common_all_three"])
    }
    balanced_ids = {
        row["sample_id"] for row in dataset if bool(row["is_balanced_common"])
    }
    expected_common_records = int(static_cfg["expected_common_population_records"])
    expected_balanced_records = int(static_cfg["expected_balanced_population_records"])
    if len(common_ids) != expected_common_records:
        raise ValueError(f"Expected {expected_common_records} common rows, found {len(common_ids)}.")
    if len(balanced_ids) != expected_balanced_records:
        raise ValueError(f"Expected {expected_balanced_records} balanced rows, found {len(balanced_ids)}.")

    c = float(static_cfg["logistic_regression"]["C"])
    max_iter = int(static_cfg["logistic_regression"]["max_iter"])
    seed = int(config["seed"])
    predictions: list[dict[str, Any]] = []

    # Primary experiment: all variants on the same 104-parent matched population.
    primary_full_fusion: list[dict[str, Any]] | None = None
    for variant in VARIANTS:
        print(f"Running primary_common_all / {variant}")
        rows = run_topic_disjoint_cv(
            store=store,
            split_definition=splits,
            population_ids=common_ids,
            variant=variant,
            c=c,
            max_iter=max_iter,
            seed=seed,
        )
        for row in rows:
            row["experiment"] = "primary_common_all"
        predictions.extend(rows)
        if variant == "full_fusion":
            primary_full_fusion = rows

    # Balanced robustness check.
    print("Running balanced_common_all / full_fusion")
    balanced_rows = run_topic_disjoint_cv(
        store=store,
        split_definition=splits,
        population_ids=balanced_ids,
        variant="full_fusion",
        c=c,
        max_iter=max_iter,
        seed=seed + 100,
    )
    for row in balanced_rows:
        row["experiment"] = "balanced_common_all"
    predictions.extend(balanced_rows)

    # Channel-specific feasible-domain models.
    for attack in ATTACKS:
        population = _matched_population_ids(dataset, attack)
        print(f"Running channel_specific_{attack} / full_fusion records={len(population)}")
        rows = run_topic_disjoint_cv(
            store=store,
            split_definition=splits,
            population_ids=population,
            variant="full_fusion",
            c=c,
            max_iter=max_iter,
            seed=seed + 200 + ATTACKS.index(attack),
        )
        for row in rows:
            row["experiment"] = f"channel_specific_{attack}"
        predictions.extend(rows)

    unicode_rows = _unicode_hygiene_predictions(dataset, store.unicode_by_id)
    predictions.extend(unicode_rows)

    # Deployed layer: deterministic Unicode hygiene override on primary core fusion.
    if primary_full_fusion is None:
        raise RuntimeError("Primary full-fusion predictions were not generated.")
    deployed_rows = []
    for row in primary_full_fusion:
        unicode_flag = int(store.unicode_by_id[row["sample_id"]]["has_suspicious_codepoint"])
        deployed = dict(row)
        deployed["experiment"] = "primary_common_all"
        deployed["variant"] = "full_deployed_with_unicode_hygiene"
        if unicode_flag:
            deployed["score"] = 1.0
            deployed["pred_f1"] = 1
            deployed["pred_fpr_05"] = 1
            deployed["pred_fpr_01"] = 1
        deployed_rows.append(deployed)
    predictions.extend(deployed_rows)

    atomic_write_jsonl(predictions, prediction_path)

    bootstrap_replicates = int(static_cfg["bootstrap_replicates"])
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(row["experiment"], row["variant"])].append(row)

    metrics: dict[str, Any] = {
        "protocol_version": "step8_static_v1",
        "prediction_record_count": len(predictions),
        "experiments": {},
        "dataset_sha256": sha256_file(paths["detection_dataset_jsonl"]),
        "feature_manifest_sha256": sha256_file(paths["detection_feature_manifest_json"]),
        "splits_sha256": sha256_file(paths["detection_splits_json"]),
        "notes": {
            "primary_cross_validation": "leave-one-topic-out",
            "threshold_calibration": "nested leave-one-topic-out on outer-training topics",
            "cross_model_normalization": "clean-training-fold z normalization per evaluator",
            "tail_05_threshold": "5th percentile of clean training token log-probabilities per evaluator",
            "unicode_hygiene": "reported separately and as a deterministic deployed override",
            "fpr_0_1_percent": "not used as a primary operating point because 160 clean controls cannot support stable empirical estimation",
        },
    }

    for (experiment, variant), rows in sorted(grouped.items()):
        key = f"{experiment}::{variant}"
        block = {
            "overall": _metric_block(rows, bootstrap_replicates, seed + sum(map(ord, key))),
        }
        if experiment == "primary_common_all" and variant in {"full_fusion", "full_deployed_with_unicode_hygiene"}:
            block["stratified"] = _stratified_metrics(
                rows,
                bootstrap_replicates=bootstrap_replicates,
                seed=seed + 800,
            )
        metrics["experiments"][key] = block

    atomic_write_json(metrics, metrics_path)
    print("Step 8 static detection run complete.")
    print(f"Wrote: {prediction_path}")
    print(f"Wrote: {metrics_path}")
    primary_key = "primary_common_all::full_fusion"
    if primary_key in metrics["experiments"]:
        print("Primary full-fusion metrics:")
        print(metrics["experiments"][primary_key]["overall"])


if __name__ == "__main__":
    main()
