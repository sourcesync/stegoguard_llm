from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from stegoguard_lm.detection.io_utils import atomic_write_json, read_json, read_jsonl, sha256_file


def _format_review(title: str, rows: list[dict], dataset_by_id: dict[str, dict], limit: int) -> str:
    lines = [title, "=" * len(title), ""]
    for row in rows[:limit]:
        source = dataset_by_id[row["sample_id"]]
        lines.extend(
            [
                f"sample_id: {row['sample_id']}",
                f"label: {row['label']}",
                f"attack_name: {row['attack_name']}",
                f"score: {row['score']:.6f}",
                f"pred_f1: {row['pred_f1']}",
                f"source_model: {row['source_model']}",
                f"domain: {row['domain']}",
                f"nominal_length: {row['nominal_target_length']}",
                f"topic_id: {row['topic_id']}",
                f"payload_bits: {row['payload_bit_length']}",
                f"payload_rate_bpt: {row['payload_rate_bpt']}",
                "PROMPT:",
                source["prompt"],
                "TEXT:",
                source["text"],
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Step 8 static detector results and QA cases.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    paths = config["paths"]
    metrics = read_json(paths["detection_metrics_json"])
    predictions = read_jsonl(paths["detection_predictions_jsonl"])
    dataset = read_jsonl(paths["detection_dataset_jsonl"])
    dataset_by_id = {row["sample_id"]: row for row in dataset}

    primary = [
        row for row in predictions
        if row["experiment"] == "primary_common_all" and row["variant"] == "full_fusion"
    ]
    clean_ranked = sorted(
        [row for row in primary if row["label"] == "clean"],
        key=lambda row: float(row["score"]),
        reverse=True,
    )
    stego_ranked = sorted(
        [row for row in primary if row["label"] == "stego"],
        key=lambda row: float(row["score"]),
    )
    review_limit = int(config["static_detection"]["qa_review_limit"])
    fp_text = _format_review(
        "Highest-scoring clean responses (false-positive risk review)",
        clean_ranked,
        dataset_by_id,
        review_limit,
    )
    fn_text = _format_review(
        "Lowest-scoring stego responses (false-negative risk review)",
        stego_ranked,
        dataset_by_id,
        review_limit,
    )
    Path(paths["detection_false_positive_review_txt"]).write_text(fp_text, encoding="utf-8")
    Path(paths["detection_false_negative_review_txt"]).write_text(fn_text, encoding="utf-8")

    primary_core = metrics["experiments"]["primary_common_all::full_fusion"]["overall"]
    primary_deployed = metrics["experiments"]["primary_common_all::full_deployed_with_unicode_hygiene"]["overall"]
    unicode_hygiene = metrics["experiments"]["unicode_hygiene::unicode_hygiene"]["overall"]
    summary = {
        "protocol_version": "step8_static_v1",
        "primary_population": "104-parent common all-three matched set",
        "primary_population_records": len(primary),
        "primary_core_full_fusion": primary_core,
        "primary_deployed_with_unicode_hygiene": primary_deployed,
        "unicode_hygiene_channel_specific": unicode_hygiene,
        "ablation_keys": sorted(
            key for key in metrics["experiments"] if key.startswith("primary_common_all::")
        ),
        "qa_review_limit_each": review_limit,
        "prediction_sha256": sha256_file(paths["detection_predictions_jsonl"]),
        "metrics_sha256": sha256_file(paths["detection_metrics_json"]),
        "false_positive_review_path": paths["detection_false_positive_review_txt"],
        "false_negative_review_path": paths["detection_false_negative_review_txt"],
    }
    atomic_write_json(summary, paths["detection_summary_json"])
    print("Step 8 static detection summary built.")
    print(f"Primary common-set rows: {len(primary)}")
    print(f"Core full-fusion AUROC: {primary_core.get('roc_auc')}")
    print(f"Core full-fusion AUPRC: {primary_core.get('pr_auc')}")
    print(f"Wrote: {paths['detection_summary_json']}")
    print(f"Wrote: {paths['detection_false_positive_review_txt']}")
    print(f"Wrote: {paths['detection_false_negative_review_txt']}")


if __name__ == "__main__":
    main()
