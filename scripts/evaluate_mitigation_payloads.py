from __future__ import annotations

import argparse
import gc
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

from stegoguard_lm.attacks.synonym_channel import load_synonym_codebook
from stegoguard_lm.detection.io_utils import append_jsonl_record, read_jsonl
from stegoguard_lm.mitigation.payload_eval import (
    evaluate_rank_payload_after_mitigation,
    evaluate_synonym_payload_after_mitigation,
    evaluate_unicode_payload_after_mitigation,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping.")
    return value


def load_model_and_tokenizer(model_id: str):
    from stegoguard_lm.clean_generation import load_model_and_tokenizer as loader
    return loader(model_id)


def collect_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode Step 9 mitigated stego outputs with the frozen attack decoders.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--channel", choices=["all", "unicode", "synonym", "rank"], default="all")
    parser.add_argument("--condition", choices=["all", "unicode_sanitization", "deterministic_rewrite", "randomized_rewrite"], default="all")
    parser.add_argument("--population", choices=["all", "primary_common_all", "balanced_common"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    outputs = read_jsonl(paths["mitigation_outputs_jsonl"])
    attacks = read_jsonl(paths["attack_full_samples_jsonl"])
    attack_by_id = {str(row["sample_id"]): row for row in attacks}
    result_path = Path(paths["mitigation_payload_results_jsonl"])
    existing = read_jsonl(result_path) if result_path.exists() else []
    existing_ids = {str(row["task_id"]) for row in existing}
    if existing_ids and not args.resume:
        raise FileExistsError("Mitigation payload results already exist. Use --resume.")

    channel_to_attack = {
        "unicode": "unicode_hidden_channel",
        "synonym": "synonym_slot_codebook",
        "rank": "rank_encoding",
    }
    selected: list[dict[str, Any]] = []
    for row in outputs:
        if row["label"] != "stego":
            continue
        if str(row["task_id"]) in existing_ids:
            continue
        if args.channel != "all" and row["attack_name"] != channel_to_attack[args.channel]:
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

    codebook = load_synonym_codebook(paths["synonym_codebook_yaml"])
    unicode_cfg = config["attacks"]["out_of_band"]
    rank_cfg = config["attacks"]["token_whitebox"]

    non_rank = [row for row in selected if row["attack_name"] != "rank_encoding"]
    rank_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        if row["attack_name"] == "rank_encoding":
            rank_rows[str(row["source_model"])].append(row)

    processed = 0
    for row in non_rank:
        original = attack_by_id[str(row["original_sample_id"])]
        if row["attack_name"] == "unicode_hidden_channel":
            metrics = evaluate_unicode_payload_after_mitigation(
                original_attack=original,
                mitigated_text=str(row["mitigated_text"]),
                attack_config=unicode_cfg,
            )
        elif row["attack_name"] == "synonym_slot_codebook":
            metrics = evaluate_synonym_payload_after_mitigation(
                original_attack=original,
                mitigated_text=str(row["mitigated_text"]),
                codebook=codebook,
            )
        else:
            raise ValueError(f"Unexpected non-rank attack: {row['attack_name']}")
        result = {
            "task_id": str(row["task_id"]),
            "original_sample_id": str(row["original_sample_id"]),
            "clean_parent_id": str(row["clean_parent_id"]),
            "attack_name": str(row["attack_name"]),
            "source_model": str(row["source_model"]),
            "domain": str(row["domain"]),
            "topic_id": str(row["topic_id"]),
            "nominal_target_length": int(row["nominal_target_length"]),
            "condition": str(row["condition"]),
            "rewrite_seed": row["rewrite_seed"],
            "is_primary_common_all": bool(row["is_primary_common_all"]),
            "is_balanced_common": bool(row["is_balanced_common"]),
            **metrics,
            "protocol_version": config["mitigation_evaluation"]["protocol_version"],
        }
        append_jsonl_record(result, result_path)
        processed += 1
        print(f"Payload evaluated: {row['task_id']}")

    for model_id, rows in rank_rows.items():
        print(f"Loading source model for rank mitigation decoding: {model_id}")
        model, tokenizer, _, _ = load_model_and_tokenizer(model_id)
        try:
            for row in rows:
                original = attack_by_id[str(row["original_sample_id"])]
                metrics = evaluate_rank_payload_after_mitigation(
                    original_attack=original,
                    mitigated_text=str(row["mitigated_text"]),
                    model=model,
                    tokenizer=tokenizer,
                    attack_config=rank_cfg,
                )
                result = {
                    "task_id": str(row["task_id"]),
                    "original_sample_id": str(row["original_sample_id"]),
                    "clean_parent_id": str(row["clean_parent_id"]),
                    "attack_name": str(row["attack_name"]),
                    "source_model": str(row["source_model"]),
                    "domain": str(row["domain"]),
                    "topic_id": str(row["topic_id"]),
                    "nominal_target_length": int(row["nominal_target_length"]),
                    "condition": str(row["condition"]),
                    "rewrite_seed": row["rewrite_seed"],
                    "is_primary_common_all": bool(row["is_primary_common_all"]),
                    "is_balanced_common": bool(row["is_balanced_common"]),
                    **metrics,
                    "protocol_version": config["mitigation_evaluation"]["protocol_version"],
                }
                append_jsonl_record(result, result_path)
                processed += 1
                print(f"Payload evaluated: {row['task_id']}")
        finally:
            del model
            del tokenizer
            collect_memory()

    print(f"Step 9 payload evaluation complete. New rows: {processed}")


if __name__ == "__main__":
    main()
