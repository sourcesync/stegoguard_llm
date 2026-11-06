from __future__ import annotations

import argparse
from pathlib import Path

from stegoguard_lm.adaptive.detector import fit_step8_fold_ensemble, score_new_sample
from stegoguard_lm.adaptive.io import load_yaml, verify_step10_protocol
from stegoguard_lm.detection.io_utils import atomic_write_jsonl, read_json, read_jsonl
from stegoguard_lm.detection.modeling import FeatureStore


def _index(rows, label):
    result = {str(r["sample_id"]): r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate sample IDs in {label}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Step 10 candidates using the frozen Step 8 outer-fold ensemble.")
    parser.add_argument("--config", default="config_step10.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    verify_step10_protocol(config)
    paths = config["paths"]
    output = Path(paths["adaptive_detector_scores_jsonl"])
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Adaptive detector scores exist: {output}")

    static_dataset = read_jsonl(paths["detection_dataset_jsonl"])
    static_store = FeatureStore(
        dataset_by_id=_index(static_dataset, "static dataset"),
        evaluator_a_by_id=_index(read_jsonl(paths["detection_evaluator_a_features_jsonl"]), "static evaluator A"),
        evaluator_b_by_id=_index(read_jsonl(paths["detection_evaluator_b_features_jsonl"]), "static evaluator B"),
        semantic_by_id=_index(read_jsonl(paths["detection_semantic_features_jsonl"]), "static semantic"),
        unicode_by_id=_index(read_jsonl(paths["detection_unicode_features_jsonl"]), "static unicode"),
    )
    primary_ids = {str(r["sample_id"]) for r in static_dataset if bool(r["is_common_all_three"])}
    static_cfg = config["static_detection"]
    detectors = fit_step8_fold_ensemble(
        static_store=static_store,
        split_definition=read_json(paths["detection_splits_json"]),
        population_ids=primary_ids,
        variant=str(config["adaptive_evaluation"]["detector_variant"]),
        c=float(static_cfg["logistic_regression"]["C"]),
        max_iter=int(static_cfg["logistic_regression"]["max_iter"]),
        seed=int(config["seed"]),
    )

    adaptive_dataset = read_jsonl(paths["adaptive_detection_dataset_jsonl"])
    new_store = FeatureStore(
        dataset_by_id=_index(adaptive_dataset, "adaptive dataset"),
        evaluator_a_by_id=_index(read_jsonl(paths["adaptive_detection_evaluator_a_features_jsonl"]), "adaptive evaluator A"),
        evaluator_b_by_id=_index(read_jsonl(paths["adaptive_detection_evaluator_b_features_jsonl"]), "adaptive evaluator B"),
        semantic_by_id=_index(read_jsonl(paths["adaptive_detection_semantic_features_jsonl"]), "adaptive semantic"),
        unicode_by_id=_index(read_jsonl(paths["adaptive_detection_unicode_features_jsonl"]), "adaptive unicode"),
    )
    rows = []
    for index, record in enumerate(adaptive_dataset, start=1):
        sample_id = str(record["sample_id"])
        score = score_new_sample(
            new_store=new_store,
            sample_id=sample_id,
            detectors=detectors,
            variant=str(config["adaptive_evaluation"]["detector_variant"]),
        )
        unicode_flag = bool(new_store.unicode_by_id[sample_id].get("has_suspicious_codepoint"))
        rows.append(
            {
                "sample_id": sample_id,
                "clean_parent_id": str(record["clean_parent_id"]),
                "label": str(record["label"]),
                "attack_name": str(record["attack_name"]),
                "source_model": str(record["source_model"]),
                "domain": str(record["domain"]),
                "topic_id": str(record["topic_id"]),
                "nominal_target_length": int(record["nominal_target_length"]),
                **score,
                "unicode_hygiene_flag": unicode_flag,
                "deployed_detected": bool(score["detected_f1"] or unicode_flag),
                "detector_policy": config["adaptive_evaluation"]["detector_policy"],
                "protocol_version": config["adaptive_evaluation"]["protocol_version"],
            }
        )
        print(f"[{index}/{len(adaptive_dataset)}] scored: {sample_id}")
    atomic_write_jsonl(rows, output)
    print("Step 10 candidate scoring complete.")
    print(f"Records: {len(rows)}")
    print(f"Detected: {sum(int(r['deployed_detected']) for r in rows)}")


if __name__ == "__main__":
    main()
