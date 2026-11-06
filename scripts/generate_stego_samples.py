from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

from stegoguard_lm.attacks.full_generation import (
    account_task_inventory,
    append_jsonl_record,
    atomic_write_json,
    build_corpus_summary,
    build_full_attack_record,
    build_terminal_failure_record,
    execute_eligible_task,
    is_expected_terminal_attack_failure,
    read_json,
    read_jsonl,
    task_matches_filters,
    validate_attack_task_manifest,
    validate_full_record_against_task,
    verify_generation_manifest,
    write_full_preview,
)
from stegoguard_lm.attacks.synonym_channel import load_synonym_codebook

CHANNEL_ALIASES = {
    "unicode": "out_of_band",
    "synonym": "token_blackbox",
    "rank": "token_whitebox",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the resumable Step 7 full static steganographic corpus."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--channel",
        choices=["all", "unicode", "synonym", "rank"],
        default="all",
        help="Generate only one attack family or all pending eligible families.",
    )
    parser.add_argument(
        "--source-model",
        default="all",
        help="all, qwen, mistral, or an exact configured source-model ID.",
    )
    parser.add_argument(
        "--domain",
        choices=["all", "conversational", "technical"],
        default="all",
    )
    parser.add_argument(
        "--nominal-length",
        choices=["all", "64", "256"],
        default="all",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional scheduling limit. Does not change the frozen protocol.",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Preserve accepted/terminal tasks and run only pending eligible tasks.",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete Step 7 generated samples/failures and restart generation from zero.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping.")
    return value


def load_tokenizer(model_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for Step 7 generation.") from exc

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(model_id: str):
    from stegoguard_lm.clean_generation import load_model_and_tokenizer as loader

    return loader(model_id)


def collect_model_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def configured_source_models(config: dict[str, Any]) -> list[str]:
    return [str(item["id"]) for item in config["models"]["sources"]]


def resolve_source_model_filter(config: dict[str, Any], requested: str) -> set[str] | None:
    if requested.casefold() == "all":
        return None
    source_models = configured_source_models(config)
    if requested in source_models:
        return {requested}

    needle = requested.casefold()
    matches = [model for model in source_models if needle in model.casefold()]
    if len(matches) != 1:
        raise ValueError(
            f"--source-model {requested!r} matched {len(matches)} configured models: "
            f"{matches}. Use qwen, mistral, or the exact ID."
        )
    return {matches[0]}


def selected_family_filter(channel: str) -> set[str] | None:
    if channel == "all":
        return None
    return {CHANNEL_ALIASES[channel]}


def validate_existing_outputs(
    *,
    tasks: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    clean_by_id: dict[str, dict[str, Any]],
) -> None:
    task_by_id = {task["task_id"]: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("Frozen task manifest contains duplicate task_id values.")

    seen_samples: set[str] = set()
    for record in accepted:
        task_id = str(record.get("task_id", ""))
        task = task_by_id.get(task_id)
        if task is None:
            raise ValueError(f"Accepted record has unknown task_id: {task_id}")
        parent = clean_by_id.get(str(task["parent_sample_id"]))
        if parent is None:
            raise ValueError(f"Clean parent missing for accepted task: {task_id}")
        errors = validate_full_record_against_task(
            record=record,
            task=task,
            parent=parent,
        )
        if errors:
            raise ValueError(
                f"Existing accepted record {record.get('sample_id')} is incompatible "
                "with the frozen Step 7 protocol: " + " | ".join(errors)
            )
        sample_id = str(record["sample_id"])
        if sample_id in seen_samples:
            raise ValueError(f"Duplicate accepted sample_id: {sample_id}")
        seen_samples.add(sample_id)

    seen_failure_tasks: set[str] = set()
    for failure in failures:
        task_id = str(failure.get("task_id", ""))
        task = task_by_id.get(task_id)
        if task is None:
            raise ValueError(f"Terminal failure has unknown task_id: {task_id}")
        if failure.get("status") != "terminal_encoding_failure":
            raise ValueError(f"Failure {task_id} is not marked terminal_encoding_failure.")
        if task_id in seen_failure_tasks:
            raise ValueError(f"Duplicate terminal failure task_id: {task_id}")
        seen_failure_tasks.add(task_id)

    account_task_inventory(
        tasks=tasks,
        accepted_records=accepted,
        terminal_failures=failures,
    )


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")

    config = load_config(args.config)
    project_root = Path(args.config).resolve().parent
    paths = config["paths"]

    clean_path = Path(paths["clean_samples_full_jsonl"])
    task_path = Path(paths["attack_task_manifest_jsonl"])
    structural_path = Path(paths["attack_structural_ineligible_jsonl"])
    generation_manifest_path = Path(paths["attack_generation_manifest_json"])
    capacity_path = Path(paths["synonym_capacity_json"])
    codebook_path = Path(paths["synonym_codebook_yaml"])

    output_path = Path(paths["attack_full_samples_jsonl"])
    failure_path = Path(paths["attack_full_failures_jsonl"])
    preview_path = Path(paths["attack_full_preview_txt"])
    summary_path = Path(paths["attack_full_summary_json"])

    if args.overwrite:
        for path in (output_path, failure_path, preview_path, summary_path):
            path.unlink(missing_ok=True)
    elif not args.resume and any(
        path.exists() for path in (output_path, failure_path, preview_path, summary_path)
    ):
        raise FileExistsError(
            "Step 7 generated outputs already exist. Use --resume to continue or "
            "--overwrite only for an intentional fresh restart."
        )

    tasks = read_jsonl(task_path)
    structural = read_jsonl(structural_path)
    generation_manifest = read_json(generation_manifest_path)
    clean_records = read_jsonl(clean_path)
    clean_by_id = {str(record["sample_id"]): record for record in clean_records}
    if len(clean_by_id) != len(clean_records):
        raise ValueError("Frozen clean corpus has duplicate sample_id values.")

    validate_attack_task_manifest(
        tasks=tasks,
        config=config,
        expected_structural_count=len(structural),
    )
    structural_ids = {task["task_id"] for task in structural}
    manifest_structural_ids = {
        task["task_id"] for task in tasks if not bool(task["eligible"])
    }
    if structural_ids != manifest_structural_ids:
        raise ValueError("Structural exclusion file drifted from the frozen task manifest.")

    verify_generation_manifest(
        config=config,
        generation_manifest=generation_manifest,
        task_manifest_path=task_path,
        structural_path=structural_path,
        clean_path=clean_path,
        capacity_audit_path=capacity_path,
        codebook_path=codebook_path,
        project_root=project_root,
    )

    accepted = read_jsonl(output_path, allow_missing=True)
    terminal_failures = read_jsonl(failure_path, allow_missing=True)
    validate_existing_outputs(
        tasks=tasks,
        accepted=accepted,
        failures=terminal_failures,
        clean_by_id=clean_by_id,
    )

    accepted_by_task = {record["task_id"]: record for record in accepted}
    failure_by_task = {record["task_id"]: record for record in terminal_failures}
    completed_ids = set(accepted_by_task) | set(failure_by_task) | structural_ids

    attack_families = selected_family_filter(args.channel)
    source_models = resolve_source_model_filter(config, args.source_model)
    domains = None if args.domain == "all" else {args.domain}
    nominal_lengths = (
        None if args.nominal_length == "all" else {int(args.nominal_length)}
    )

    requested_tasks = [
        task
        for task in tasks
        if bool(task["eligible"])
        and task_matches_filters(
            task,
            attack_families=attack_families,
            source_models=source_models,
            domains=domains,
            nominal_lengths=nominal_lengths,
        )
    ]
    requested_task_ids = {str(task["task_id"]) for task in requested_tasks}
    pending_tasks = [
        task for task in requested_tasks if str(task["task_id"]) not in completed_ids
    ]
    pending_tasks.sort(key=lambda task: int(task["task_index"]))
    if args.limit is not None:
        pending_tasks = pending_tasks[: args.limit]

    records_at_start = len(accepted)
    failures_at_start = len(terminal_failures)
    new_accepted = 0
    new_terminal = 0

    codebook = None
    if any(task["attack_family"] == "token_blackbox" for task in pending_tasks):
        codebook = load_synonym_codebook(codebook_path)

    pending_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in pending_tasks:
        pending_by_model[str(task["source_model"])].append(task)

    if not pending_tasks:
        print("All requested eligible Step 7 tasks are already final or no tasks match the filter.")

    model_order = configured_source_models(config)
    for source_model in model_order:
        model_tasks = pending_by_model.get(source_model, [])
        if not model_tasks:
            continue

        lightweight_tasks = [
            task
            for task in model_tasks
            if task["attack_family"] in {"out_of_band", "token_blackbox"}
        ]
        rank_tasks = [
            task for task in model_tasks if task["attack_family"] == "token_whitebox"
        ]

        # Unicode and synonym need only the source tokenizer.
        if lightweight_tasks:
            tokenizer = load_tokenizer(source_model)
            try:
                for task in lightweight_tasks:
                    parent = clean_by_id[str(task["parent_sample_id"])]
                    try:
                        result = execute_eligible_task(
                            config=config,
                            task=task,
                            parent=parent,
                            tokenizer=tokenizer,
                            model=None,
                            codebook=codebook,
                        )
                        record = build_full_attack_record(
                            config=config,
                            task=task,
                            parent=parent,
                            result=result,
                        )
                        errors = validate_full_record_against_task(
                            record=record,
                            task=task,
                            parent=parent,
                        )
                        if errors:
                            raise RuntimeError(
                                "Post-encoding protocol validation failed: "
                                + " | ".join(errors)
                            )
                        append_jsonl_record(record, output_path)
                        accepted.append(record)
                        accepted_by_task[task["task_id"]] = record
                        completed_ids.add(task["task_id"])
                        new_accepted += 1
                        print(f"Accepted: {record['sample_id']}")
                    except Exception as exc:
                        if is_expected_terminal_attack_failure(task, exc):
                            failure = build_terminal_failure_record(
                                task=task,
                                parent=parent,
                                exc=exc,
                            )
                            append_jsonl_record(failure, failure_path)
                            terminal_failures.append(failure)
                            failure_by_task[task["task_id"]] = failure
                            completed_ids.add(task["task_id"])
                            new_terminal += 1
                            print(f"Terminal failure: {task['task_id']}: {exc}")
                        else:
                            print(
                                f"Non-terminal execution/protocol error for {task['task_id']}. "
                                "The task remains pending and no terminal failure was written."
                            )
                            raise
            finally:
                tokenizer = None
                collect_model_memory()

        # Rank tasks load the causal model exactly once for this source model.
        if rank_tasks:
            model = None
            tokenizer = None
            try:
                model, tokenizer, _, _ = load_model_and_tokenizer(source_model)
                for task in rank_tasks:
                    parent = clean_by_id[str(task["parent_sample_id"])]
                    try:
                        result = execute_eligible_task(
                            config=config,
                            task=task,
                            parent=parent,
                            tokenizer=tokenizer,
                            model=model,
                            codebook=None,
                        )
                        record = build_full_attack_record(
                            config=config,
                            task=task,
                            parent=parent,
                            result=result,
                        )
                        errors = validate_full_record_against_task(
                            record=record,
                            task=task,
                            parent=parent,
                        )
                        if errors:
                            raise RuntimeError(
                                "Post-encoding protocol validation failed: "
                                + " | ".join(errors)
                            )
                        append_jsonl_record(record, output_path)
                        accepted.append(record)
                        accepted_by_task[task["task_id"]] = record
                        completed_ids.add(task["task_id"])
                        new_accepted += 1
                        print(f"Accepted: {record['sample_id']}")
                    except Exception as exc:
                        if is_expected_terminal_attack_failure(task, exc):
                            failure = build_terminal_failure_record(
                                task=task,
                                parent=parent,
                                exc=exc,
                            )
                            append_jsonl_record(failure, failure_path)
                            terminal_failures.append(failure)
                            failure_by_task[task["task_id"]] = failure
                            completed_ids.add(task["task_id"])
                            new_terminal += 1
                            print(f"Terminal failure: {task['task_id']}: {exc}")
                        else:
                            print(
                                f"Non-terminal execution/protocol error for {task['task_id']}. "
                                "The task remains pending and no terminal failure was written."
                            )
                            raise
            finally:
                model = None
                tokenizer = None
                collect_model_memory()

    # Reload exact persisted state before writing summary/preview.
    persisted_accepted = read_jsonl(output_path, allow_missing=True)
    persisted_failures = read_jsonl(failure_path, allow_missing=True)
    validate_existing_outputs(
        tasks=tasks,
        accepted=persisted_accepted,
        failures=persisted_failures,
        clean_by_id=clean_by_id,
    )

    final_completed = {
        record["task_id"] for record in persisted_accepted
    } | {
        record["task_id"] for record in persisted_failures
    } | structural_ids
    requested_remaining = sorted(requested_task_ids - final_completed)
    run_metadata = {
        "channel_filter": args.channel,
        "source_model_filter": args.source_model,
        "domain_filter": args.domain,
        "nominal_length_filter": args.nominal_length,
        "limit": args.limit,
        "records_at_start": records_at_start,
        "terminal_failures_at_start": failures_at_start,
        "new_accepted_this_run": new_accepted,
        "new_terminal_failures_this_run": new_terminal,
        "requested_eligible_tasks": len(requested_task_ids),
        "requested_tasks_remaining": len(requested_remaining),
        "requested_task_ids_remaining": requested_remaining,
        "complete_for_requested_tasks": len(requested_remaining) == 0,
    }
    summary = build_corpus_summary(
        tasks=tasks,
        accepted_records=persisted_accepted,
        terminal_failures=persisted_failures,
        run_metadata=run_metadata,
    )
    summary.update(
        {
            "output_path": str(output_path),
            "failure_path": str(failure_path),
            "preview_path": str(preview_path),
            "task_manifest_path": str(task_path),
        }
    )
    atomic_write_json(summary, summary_path)
    write_full_preview(persisted_accepted, preview_path)

    print("Step 7 steganographic generation run complete.")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
