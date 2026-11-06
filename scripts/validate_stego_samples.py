from __future__ import annotations

import argparse
import gc
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

from stegoguard_lm.attacks.full_generation import (
    account_task_inventory,
    read_json,
    read_jsonl,
    validate_attack_task_manifest,
    validate_full_record_against_task,
    verify_generation_manifest,
)
from stegoguard_lm.attacks.payloads import payload_sha256, validate_bitstring
from stegoguard_lm.attacks.rank_channel import decode_rank_payload
from stegoguard_lm.attacks.synonym_channel import (
    decode_synonym_payload,
    discover_inplace_synonym_slots,
    load_synonym_codebook,
)
from stegoguard_lm.attacks.unicode_channel import decode_unicode_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the persisted Step 7 full static attack corpus."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Allow incomplete eligible tasks. Useful only for interim checks.",
    )
    parser.add_argument(
        "--require-zero-terminal-failures",
        action="store_true",
        help="Fail validation if any eligible task exhausted its frozen attack budget.",
    )
    parser.add_argument(
        "--verify-rank-with-model",
        action="store_true",
        help="Independently decode every accepted rank record with its source model.",
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
        raise RuntimeError("transformers is required for Step 7 validation.") from exc
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    project_root = Path(args.config).resolve().parent
    paths = config["paths"]

    clean_path = Path(paths["clean_samples_full_jsonl"])
    task_path = Path(paths["attack_task_manifest_jsonl"])
    structural_path = Path(paths["attack_structural_ineligible_jsonl"])
    generation_manifest_path = Path(paths["attack_generation_manifest_json"])
    capacity_path = Path(paths["synonym_capacity_json"])
    codebook_path = Path(paths["synonym_codebook_yaml"])
    samples_path = Path(paths["attack_full_samples_jsonl"])
    failures_path = Path(paths["attack_full_failures_jsonl"])

    clean_records = read_jsonl(clean_path)
    clean_by_id = {record["sample_id"]: record for record in clean_records}
    if len(clean_by_id) != len(clean_records):
        raise ValueError("Clean corpus contains duplicate sample_id values.")

    tasks = read_jsonl(task_path)
    task_by_id = {task["task_id"]: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("Task manifest contains duplicate task_id values.")
    structural = read_jsonl(structural_path)
    structural_ids = {task["task_id"] for task in structural}
    if len(structural_ids) != len(structural):
        raise ValueError("Structural exclusion file contains duplicate task_id values.")

    validate_attack_task_manifest(
        tasks=tasks,
        config=config,
        expected_structural_count=len(structural),
    )
    manifest_structural = {
        task["task_id"] for task in tasks if not bool(task["eligible"])
    }
    if structural_ids != manifest_structural:
        raise ValueError("Structural exclusion file does not match frozen task manifest.")

    generation_manifest = read_json(generation_manifest_path)
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

    attack_records = read_jsonl(samples_path, allow_missing=True)
    terminal_failures = read_jsonl(failures_path, allow_missing=True)
    accounting = account_task_inventory(
        tasks=tasks,
        accepted_records=attack_records,
        terminal_failures=terminal_failures,
    )

    errors: list[str] = []
    seen_samples: set[str] = set()
    seen_comparison_ids: set[str] = set()
    tokenizer_cache: dict[str, Any] = {}
    codebook = load_synonym_codebook(codebook_path)
    attacks_config = config["attacks"]
    rank_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for line_number, record in enumerate(attack_records, start=1):
        sample_id = str(record.get("sample_id", ""))
        prefix = f"Line {line_number} ({sample_id or 'missing sample_id'})"

        if sample_id in seen_samples:
            errors.append(f"{prefix}: duplicate sample_id.")
        seen_samples.add(sample_id)
        comparison_id = str(record.get("comparison_id", ""))
        if comparison_id in seen_comparison_ids:
            errors.append(f"{prefix}: duplicate comparison_id.")
        seen_comparison_ids.add(comparison_id)

        task_id = str(record.get("task_id", ""))
        task = task_by_id.get(task_id)
        if task is None:
            errors.append(f"{prefix}: task_id not found in frozen manifest: {task_id!r}.")
            continue
        if not bool(task["eligible"]):
            errors.append(f"{prefix}: structurally ineligible task was accepted.")
            continue
        parent = clean_by_id.get(str(task["parent_sample_id"]))
        if parent is None:
            errors.append(f"{prefix}: clean parent not found.")
            continue

        for error in validate_full_record_against_task(
            record=record,
            task=task,
            parent=parent,
        ):
            errors.append(f"{prefix}: {error}")

        try:
            validate_bitstring(str(record.get("payload", "")))
            if payload_sha256(record["payload"]) != record.get("payload_sha256"):
                errors.append(f"{prefix}: payload_sha256 mismatch.")
        except Exception as exc:
            errors.append(f"{prefix}: invalid payload/hash: {exc}")

        source_model = str(record["source_model"])
        if source_model not in tokenizer_cache:
            tokenizer_cache[source_model] = load_tokenizer(source_model)
        tokenizer = tokenizer_cache[source_model]
        family = str(record["attack_family"])
        parent_length = int(parent["actual_token_length"])

        if family in {"token_blackbox", "token_whitebox"}:
            actual_length = len(tokenizer.encode(record["text"], add_special_tokens=False))
            if actual_length != parent_length:
                errors.append(
                    f"{prefix}: independent source-tokenizer length={actual_length}, "
                    f"parent={parent_length}."
                )
            if actual_length != int(record.get("stego_actual_token_length", -1)):
                errors.append(
                    f"{prefix}: independent length differs from stored stego length."
                )

        if family == "out_of_band":
            settings = attacks_config["out_of_band"]
            try:
                decoded = decode_unicode_payload(
                    record["text"],
                    zero_codepoint=str(settings["bit_zero_codepoint"]),
                    one_codepoint=str(settings["bit_one_codepoint"]),
                    prefix_codepoint=str(settings["prefix_codepoint"]),
                    suffix_codepoint=str(settings["suffix_codepoint"]),
                )
                if decoded.payload != record["payload"]:
                    errors.append(f"{prefix}: independent Unicode payload mismatch.")
                if decoded.stripped_text != parent["text"]:
                    errors.append(f"{prefix}: Unicode stripped text differs from clean parent.")
                stripped_length = len(
                    tokenizer.encode(decoded.stripped_text, add_special_tokens=False)
                )
                raw_length = len(tokenizer.encode(record["text"], add_special_tokens=False))
                if stripped_length != parent_length:
                    errors.append(
                        f"{prefix}: Unicode stripped text has {stripped_length} tokens, "
                        f"expected {parent_length}."
                    )
                if raw_length != int(record.get("raw_stego_source_token_length", -1)):
                    errors.append(f"{prefix}: raw Unicode token metadata mismatch.")
                if stripped_length != int(
                    record.get("visible_text_source_token_length_after_strip", -1)
                ):
                    errors.append(f"{prefix}: stripped Unicode token metadata mismatch.")
            except Exception as exc:
                errors.append(f"{prefix}: Unicode validation exception: {exc}")

        elif family == "token_blackbox":
            parameters = record.get("channel_parameters", {})
            active_slot_ids = parameters.get("active_slot_ids")
            expected_slot_ids = [str(value) for value in task["synonym_eligible_slot_ids"]]
            if active_slot_ids != expected_slot_ids:
                errors.append(
                    f"{prefix}: active synonym slots do not equal the frozen full-capacity slots."
                )
            decoded = decode_synonym_payload(
                record["text"],
                codebook,
                active_slot_ids=active_slot_ids,
            )
            if not decoded.success:
                errors.append(
                    f"{prefix}: independent synonym decode failed: {decoded.errors}"
                )
            elif decoded.payload != record["payload"]:
                errors.append(f"{prefix}: independent synonym payload mismatch.")

            if record["text"] == parent["text"]:
                errors.append(f"{prefix}: synonym stego text is identical to clean parent.")
            if not parameters.get("changed_slot_ids"):
                errors.append(f"{prefix}: synonym record has no changed_slot_ids.")

            try:
                discovered = discover_inplace_synonym_slots(
                    clean_text=parent["text"],
                    tokenizer=tokenizer,
                    codebook=codebook,
                    required_token_length=parent_length,
                    require_complete_ending=bool(
                        attacks_config["token_blackbox"]["require_complete_ending"]
                    ),
                    require_tokenizer_roundtrip=bool(
                        attacks_config["token_blackbox"]["require_tokenizer_roundtrip"]
                    ),
                    forbid_format_codepoints=bool(
                        attacks_config["token_blackbox"].get(
                            "forbid_format_codepoints", True
                        )
                    ),
                )
                discovered_ids = [slot.slot_id for slot in discovered]
                if discovered_ids != expected_slot_ids:
                    errors.append(
                        f"{prefix}: live synonym capacity differs from frozen task: "
                        f"{discovered_ids} != {expected_slot_ids}."
                    )
            except Exception as exc:
                errors.append(f"{prefix}: synonym capacity recheck failed: {exc}")

        elif family == "token_whitebox":
            rank_by_model[source_model].append(record)
        else:
            errors.append(f"{prefix}: unknown attack family {family!r}.")

    # Terminal failures must be valid eligible tasks and must represent the frozen
    # attack-search exhaustion, not environment or protocol errors.
    seen_failure_ids: set[str] = set()
    for failure in terminal_failures:
        task_id = str(failure.get("task_id", ""))
        if task_id in seen_failure_ids:
            errors.append(f"Duplicate terminal failure task_id: {task_id}")
        seen_failure_ids.add(task_id)
        task = task_by_id.get(task_id)
        if task is None:
            errors.append(f"Terminal failure references unknown task: {task_id}")
            continue
        if not bool(task["eligible"]):
            errors.append(f"Structural task incorrectly recorded as terminal failure: {task_id}")
        if failure.get("status") != "terminal_encoding_failure":
            errors.append(f"Terminal failure has wrong status: {task_id}")
        message = str(failure.get("error_message", ""))
        family = str(task["attack_family"])
        if family == "token_whitebox" and not message.startswith("Rank encoding failed after "):
            errors.append(f"Rank terminal failure lacks expected exhaustion message: {task_id}")
        if family == "token_blackbox" and not message.startswith(
            "Synonym full-capacity payload selection failed after "
        ):
            errors.append(f"Synonym terminal failure lacks expected exhaustion message: {task_id}")
        if family == "out_of_band":
            errors.append(f"Unicode task should not be terminalized as an attack search failure: {task_id}")

    if args.verify_rank_with_model:
        for source_model, records in rank_by_model.items():
            model = None
            tokenizer = None
            try:
                model, tokenizer, _, _ = load_model_and_tokenizer(source_model)
                for record in records:
                    parameters = record["channel_parameters"]
                    decoded = decode_rank_payload(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=record["prompt"],
                        stego_text=record["text"],
                        attack_seed=int(parameters["attack_seed"]),
                        encoding_nonce=int(parameters["encoding_nonce"]),
                        top_k=int(parameters["top_k"]),
                        final_candidate_search_k=int(
                            parameters["final_candidate_search_k"]
                        ),
                        require_complete_ending=True,
                    )
                    if not decoded.success:
                        errors.append(
                            f"{record['sample_id']}: model-based rank decode failed: "
                            f"{decoded.errors}"
                        )
                    elif decoded.payload != record["payload"]:
                        errors.append(
                            f"{record['sample_id']}: model-based rank payload mismatch."
                        )
            finally:
                model = None
                tokenizer = None
                collect_model_memory()

    if accounting["pending_tasks"] and not args.allow_pending:
        errors.append(
            f"Step 7 has {accounting['pending_tasks']} pending eligible tasks."
        )
    if accounting["terminal_encoding_failures"] and args.require_zero_terminal_failures:
        errors.append(
            "Step 7 contains terminal encoding failures but zero were required: "
            f"{accounting['terminal_encoding_failures']}."
        )

    if errors:
        print("Step 7 full attack validation failed.")
        for error in errors[:120]:
            print(f"- {error}")
        if len(errors) > 120:
            print(f"- ... and {len(errors) - 120} additional errors")
        raise SystemExit(1)

    print("Step 7 full attack validation passed.")
    print(f"Attack opportunities: {accounting['attack_opportunities']}")
    print(f"Structural zero-capacity tasks: {accounting['structural_ineligible_tasks']}")
    print(f"Eligible tasks: {accounting['eligible_tasks']}")
    print(f"Accepted attack records: {accounting['accepted_tasks']}")
    print(f"Terminal encoding failures: {accounting['terminal_encoding_failures']}")
    print(f"Pending tasks: {accounting['pending_tasks']}")
    print(f"Status: {accounting['status']}")
    print("Accepted counts by attack:")
    for attack_name, count in sorted(
        Counter(record["attack_name"] for record in attack_records).items()
    ):
        print(f"  {attack_name}: {count}")
    print("Accepted counts by source model:")
    for source_model, count in sorted(
        Counter(record["source_model"] for record in attack_records).items()
    ):
        print(f"  {source_model}: {count}")
    if rank_by_model and not args.verify_rank_with_model:
        print(
            "Rank records passed persisted metadata, lineage, token-length, and "
            "generation-time decode checks. Add --verify-rank-with-model for an "
            "independent model-based re-decode of all accepted rank records."
        )


if __name__ == "__main__":
    main()
