from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import torch
import yaml

from stegoguard_lm.detection.evaluator_scoring import load_evaluator
from stegoguard_lm.detection.io_utils import append_jsonl_record, read_jsonl
from stegoguard_lm.mitigation.utility import evaluate_utility_after_mitigation


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping.")
    return value


def collect_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Step 9 post-mitigation utility features with frozen evaluator A.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--condition", choices=["all", "unicode_sanitization", "deterministic_rewrite", "randomized_rewrite"], default="all")
    parser.add_argument("--population", choices=["all", "primary_common_all", "balanced_common"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    outputs = read_jsonl(paths["mitigation_outputs_jsonl"])
    dataset = read_jsonl(paths["detection_dataset_jsonl"])
    semantic = read_jsonl(paths["detection_semantic_features_jsonl"])
    dist_a = read_jsonl(paths["detection_evaluator_a_features_jsonl"])
    dataset_by_id = {str(row["sample_id"]): row for row in dataset}
    semantic_by_id = {str(row["sample_id"]): row for row in semantic}
    dist_by_id = {str(row["sample_id"]): row for row in dist_a}
    output_path = Path(paths["mitigation_utility_jsonl"])
    existing = read_jsonl(output_path) if output_path.exists() else []
    existing_ids = {str(row["task_id"]) for row in existing}
    if existing_ids and not args.resume:
        raise FileExistsError("Mitigation utility output already exists. Use --resume.")

    selected: list[dict[str, Any]] = []
    for row in outputs:
        if str(row["task_id"]) in existing_ids:
            continue
        if args.condition != "all" and row["condition"] != args.condition:
            continue
        if args.population == "primary_common_all" and not bool(row["is_primary_common_all"]):
            continue
        if args.population == "balanced_common" and not bool(row["is_balanced_common"]):
            continue
        selected.append(row)
    if args.limit is not None:
        selected = selected[: max(0, int(args.limit))]

    evaluator_role = str(config["mitigation_evaluation"]["utility"]["evaluator_role"])
    evaluator_id = next(
        str(model["id"])
        for model in config["models"]["evaluators"]
        if str(model["role"]) == evaluator_role
    )
    bundle = load_evaluator(evaluator_id)
    original_embedding_cache: dict[str, Any] = {}
    processed = 0
    try:
        for index, row in enumerate(selected, start=1):
            sample_id = str(row["original_sample_id"])
            original = dataset_by_id[sample_id]
            utility, embedding = evaluate_utility_after_mitigation(
                model=bundle.model,
                tokenizer=bundle.tokenizer,
                visible_prompt=str(original["prompt"]),
                original_text=str(original["text"]),
                mitigated_text=str(row["mitigated_text"]),
                original_semantic_row=semantic_by_id[sample_id],
                original_distribution_row=dist_by_id[sample_id],
                original_embedding=original_embedding_cache.get(sample_id),
            )
            original_embedding_cache[sample_id] = embedding
            result = {
                "task_id": str(row["task_id"]),
                "original_sample_id": sample_id,
                "clean_parent_id": str(row["clean_parent_id"]),
                "label": str(row["label"]),
                "attack_name": str(row["attack_name"]),
                "source_model": str(row["source_model"]),
                "domain": str(row["domain"]),
                "topic_id": str(row["topic_id"]),
                "nominal_target_length": int(row["nominal_target_length"]),
                "condition": str(row["condition"]),
                "rewrite_seed": row["rewrite_seed"],
                "is_primary_common_all": bool(row["is_primary_common_all"]),
                "is_balanced_common": bool(row["is_balanced_common"]),
                **utility,
                "evaluator_role": evaluator_role,
                "evaluator_id": evaluator_id,
                "protocol_version": config["mitigation_evaluation"]["protocol_version"],
            }
            append_jsonl_record(result, output_path)
            processed += 1
            print(f"[{index}/{len(selected)}] Utility evaluated: {row['task_id']}")
    finally:
        del bundle
        collect_memory()
    print(f"Step 9 utility extraction complete. New rows: {processed}")


if __name__ == "__main__":
    main()
