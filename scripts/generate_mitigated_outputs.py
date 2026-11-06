from __future__ import annotations

import argparse
import gc
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml

from stegoguard_lm.detection.evaluator_scoring import load_evaluator
from stegoguard_lm.detection.io_utils import append_jsonl_record, read_jsonl
from stegoguard_lm.mitigation.normalization import sanitize_unicode_text
from stegoguard_lm.mitigation.rewriting import rewrite_response
from stegoguard_lm.mitigation.tasks import account_mitigation_tasks


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate resumable Step 9 mitigation outputs.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--condition",
        choices=["all", "unicode_sanitization", "deterministic_rewrite", "randomized_rewrite"],
        default="all",
    )
    parser.add_argument(
        "--population",
        choices=["all", "primary_common_all", "balanced_common"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=None, help="Restrict randomized rewriting to one frozen seed.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    cfg = config["mitigation_evaluation"]
    tasks = read_jsonl(paths["mitigation_task_manifest_jsonl"])
    dataset = read_jsonl(paths["detection_dataset_jsonl"])
    dataset_by_id = {str(row["sample_id"]): row for row in dataset}
    output_path = Path(paths["mitigation_outputs_jsonl"])
    failure_path = Path(paths["mitigation_failures_jsonl"])
    accepted = read_jsonl(output_path) if output_path.exists() else []
    failures = read_jsonl(failure_path) if failure_path.exists() else []
    accepted_ids = {str(row["task_id"]) for row in accepted}
    failure_ids = {str(row["task_id"]) for row in failures}
    if (accepted_ids or failure_ids) and not args.resume:
        raise FileExistsError("Step 9 outputs already exist. Use --resume to preserve completed tasks.")

    selected: list[dict[str, Any]] = []
    for task in tasks:
        if str(task["task_id"]) in accepted_ids or str(task["task_id"]) in failure_ids:
            continue
        if args.condition != "all" and task["condition"] != args.condition:
            continue
        if args.population == "primary_common_all" and not bool(task["is_primary_common_all"]):
            continue
        if args.population == "balanced_common" and not bool(task["is_balanced_common"]):
            continue
        if args.seed is not None:
            if task["condition"] != "randomized_rewrite" or int(task["rewrite_seed"]) != args.seed:
                continue
        selected.append(task)
    if args.limit is not None:
        selected = selected[: max(0, int(args.limit))]

    rewriter_bundle = None
    rewriter_model_id = None
    unicode_cfg = config["attacks"]["out_of_band"]
    reserved = [
        unicode_cfg["bit_zero_codepoint"],
        unicode_cfg["bit_one_codepoint"],
        unicode_cfg["prefix_codepoint"],
        unicode_cfg["suffix_codepoint"],
    ]
    normalization_cfg = cfg["unicode_sanitization"]
    rewrite_cfg = cfg["rewriter_generation"]
    counts = Counter()

    for index, task in enumerate(selected, start=1):
        task_id = str(task["task_id"])
        original = dataset_by_id[str(task["original_sample_id"])]
        condition = str(task["condition"])
        try:
            if condition == "unicode_sanitization":
                result = sanitize_unicode_text(
                    str(original["text"]),
                    normalization_form=str(normalization_cfg["normalization_form"]),
                    reserved_codepoints=reserved,
                    remove_all_format_codepoints=bool(normalization_cfg["remove_all_format_codepoints"]),
                )
                mitigated_text = result.text
                method_metadata = {
                    "normalization_form": result.normalization_form,
                    "removed_codepoint_count": result.removed_codepoint_count,
                    "removed_format_codepoint_count": result.removed_format_codepoint_count,
                    "removed_codepoints": list(result.removed_codepoints),
                    "changed": result.changed,
                }
            else:
                if rewriter_bundle is None:
                    rewriter_model_id = str(task["rewriter_model_id"])
                    rewriter_bundle = load_evaluator(rewriter_model_id)
                elif str(task["rewriter_model_id"]) != rewriter_model_id:
                    raise RuntimeError("Step 9 currently expects one frozen rewriter model per run.")
                try:
                    rewrite = rewrite_response(
                        model=rewriter_bundle.model,
                        tokenizer=rewriter_bundle.tokenizer,
                        visible_prompt=str(original["prompt"]),
                        response=str(original["text"]),
                        mode=condition,
                        seed=task["rewrite_seed"],
                        max_length_ratio=float(rewrite_cfg["max_length_ratio"]),
                        max_extra_tokens=int(rewrite_cfg["max_extra_tokens"]),
                        hard_cap=int(rewrite_cfg["hard_cap_new_tokens"]),
                        randomized_temperature=float(rewrite_cfg["randomized_temperature"]),
                        randomized_top_p=float(rewrite_cfg["randomized_top_p"]),
                    )
                    mitigated_text = rewrite.text
                    method_metadata = {
                        "changed": rewrite.changed,
                        "mode": rewrite.mode,
                        "seed": rewrite.seed,
                        "input_token_count_rewriter": rewrite.input_token_count,
                        "output_token_count_rewriter": rewrite.output_token_count,
                        "max_new_tokens": rewrite.max_new_tokens,
                        "do_sample": rewrite.do_sample,
                        "temperature": rewrite.temperature,
                        "top_p": rewrite.top_p,
                        "empty_output_fallback": False,
                    }
                except RuntimeError as rewrite_exc:
                    if not str(rewrite_exc).startswith("rewrite_empty_output:"):
                        raise
                    mitigated_text = str(original["text"])
                    method_metadata = {
                        "changed": False,
                        "mode": condition,
                        "seed": task["rewrite_seed"],
                        "empty_output_fallback": True,
                        "fallback_reason": str(rewrite_exc),
                    }

            record = {
                "task_id": task_id,
                "task_index": int(task["task_index"]),
                "original_sample_id": str(task["original_sample_id"]),
                "clean_parent_id": str(task["clean_parent_id"]),
                "pair_id": str(task["pair_id"]),
                "label": str(task["label"]),
                "attack_family": str(task["attack_family"]),
                "attack_name": str(task["attack_name"]),
                "source_model": str(task["source_model"]),
                "domain": str(task["domain"]),
                "topic_id": str(task["topic_id"]),
                "nominal_target_length": int(task["nominal_target_length"]),
                "condition": condition,
                "rewrite_seed": task["rewrite_seed"],
                "rewriter_model_id": task["rewriter_model_id"],
                "is_primary_common_all": bool(task["is_primary_common_all"]),
                "is_balanced_common": bool(task["is_balanced_common"]),
                "original_text": str(original["text"]),
                "mitigated_text": mitigated_text,
                "text_changed": mitigated_text != str(original["text"]),
                "method_metadata": method_metadata,
                "protocol_version": cfg["protocol_version"],
                "generation_status": "accepted",
            }
            append_jsonl_record(record, output_path)
            accepted.append(record)
            accepted_ids.add(task_id)
            counts[condition] += 1
            print(f"[{index}/{len(selected)}] Accepted: {task_id}")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            # Model/runtime faults remain pending rather than becoming scientific failures.
            infrastructure_markers = (
                "out of memory", "cuda", "cublas", "cudnn", "meta device",
                "cannot allocate memory", "device", "connection", "timeout",
            )
            if any(marker in message.casefold() for marker in infrastructure_markers):
                print(f"Infrastructure interruption: {task_id}: {message}")
                raise
            failure = {
                "task_id": task_id,
                "original_sample_id": str(task["original_sample_id"]),
                "condition": condition,
                "rewrite_seed": task["rewrite_seed"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "protocol_version": cfg["protocol_version"],
                "status": "terminal_mitigation_generation_failure",
            }
            append_jsonl_record(failure, failure_path)
            failures.append(failure)
            failure_ids.add(task_id)
            print(f"Terminal mitigation failure: {task_id}: {message}")

    accounting = account_mitigation_tasks(tasks=tasks, accepted_outputs=accepted, terminal_failures=failures)
    print("Step 9 mitigation generation run complete.")
    print(accounting)
    if counts:
        print("New accepted by condition:")
        for key, value in sorted(counts.items()):
            print(f"  {key}: {value}")
    if rewriter_bundle is not None:
        del rewriter_bundle
        collect_memory()


if __name__ == "__main__":
    main()
