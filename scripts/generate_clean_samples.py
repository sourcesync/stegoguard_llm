from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from stegoguard_lm.clean_generation import (
    append_jsonl_record,
    filter_standard_prompts,
    generate_length_controlled_clean_text,
    load_model_and_tokenizer,
    make_clean_sample_ids,
    make_clean_sample_record,
    read_jsonl,
    write_clean_preview,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clean samples for StegoGuard-LM."
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mini", "full"],
        default="mini",
        help="mini = small sanity run; full = full clean pilot corpus",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of prompt records per model, useful for debugging.",
    )

    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Keep accepted records already in the output JSONL, skip their "
            "sample IDs, and append only missing records."
        ),
    )
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing sample and failure JSONL files before generation.",
    )

    return parser.parse_args()


def get_source_models(config: dict, mode: str) -> list[str]:
    if mode == "mini":
        return list(config["pilot"]["mini_pilot"]["source_models"])
    return [m["id"] for m in config["models"]["sources"]]


def get_domains(config: dict, mode: str) -> list[str]:
    if mode == "mini":
        return list(config["pilot"]["mini_pilot"]["domains"])
    return list(config["pilot"]["domains"])


def get_lengths(config: dict, mode: str) -> list[int]:
    if mode == "mini":
        return [int(x) for x in config["pilot"]["mini_pilot"]["lengths"]]
    return [int(x) for x in config["pilot"]["lengths"]]


def get_clean_replicates(config: dict, mode: str) -> int:
    if mode == "mini":
        return int(
            config["pilot"]["mini_pilot"].get(
                "clean_replicates_per_prompt",
                1,
            )
        )

    return int(
        config["pilot"].get(
            "clean_replicates_per_prompt",
            1,
        )
    )


def load_existing_records(
    *,
    output_path: Path,
    expected_tasks: dict[str, dict],
    manifest_version: str,
    temperature: float,
    top_p: float,
    length_policy: str,
    length_tolerance_by_target: dict[int, int],
) -> tuple[list[dict], set[str]]:
    """Load accepted records and verify they match the current run design."""
    if not output_path.exists():
        return [], set()

    records = read_jsonl(output_path)
    completed_sample_ids: set[str] = set()

    for line_number, record in enumerate(records, start=1):
        sample_id = str(record.get("sample_id", ""))

        if not sample_id:
            raise ValueError(
                f"Existing record {line_number} has no sample_id: {output_path}"
            )

        if sample_id in completed_sample_ids:
            raise ValueError(f"Duplicate sample_id in existing output: {sample_id}")

        if sample_id not in expected_tasks:
            raise ValueError(
                "Existing output contains a sample that does not belong to "
                f"the current configuration: {sample_id}"
            )

        task = expected_tasks[sample_id]

        if record.get("generation_status") != "accepted":
            raise ValueError(
                "Only accepted records may be resumed from the clean corpus. "
                f"Invalid record: {sample_id}"
            )

        checks = {
            "source_model": task["source_model"],
            "prompt_id": task["prompt_id"],
            "target_length": task["target_length"],
            "replicate_index": task["replicate_index"],
            "manifest_version": manifest_version,
            "length_policy": length_policy,
        }
        for field, expected_value in checks.items():
            if record.get(field) != expected_value:
                raise ValueError(
                    f"Resume mismatch for {sample_id}: field {field!r} "
                    f"is {record.get(field)!r}, expected {expected_value!r}."
                )

        if str(record.get("prompt")) != task["prompt"]:
            raise ValueError(
                f"Resume mismatch for {sample_id}: saved prompt text changed."
            )

        if float(record.get("temperature")) != temperature:
            raise ValueError(f"Resume mismatch for {sample_id}: temperature changed.")

        if float(record.get("top_p")) != top_p:
            raise ValueError(f"Resume mismatch for {sample_id}: top_p changed.")

        target_length = int(record["target_length"])
        expected_tolerance = length_tolerance_by_target[target_length]
        if int(record.get("length_tolerance")) != expected_tolerance:
            raise ValueError(
                f"Resume mismatch for {sample_id}: length tolerance changed."
            )

        completed_sample_ids.add(sample_id)

    return records, completed_sample_ids


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    rendered_prompts_path = Path(config["paths"]["rendered_prompts_jsonl"])
    clean_output_dir = Path(
        config["paths"].get("clean_output_dir", "outputs/pilot_v0/clean")
    )
    clean_output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "mini":
        output_path = Path(
            config["paths"].get(
                "clean_samples_mini_jsonl",
                clean_output_dir / "clean_samples_mini.jsonl",
            )
        )
        failure_path = Path(
            config["paths"].get(
                "clean_failures_mini_jsonl",
                clean_output_dir / "clean_failures_mini.jsonl",
            )
        )
        preview_path = clean_output_dir / "clean_preview_mini.txt"
        summary_path = clean_output_dir / "clean_summary_mini.json"
    else:
        output_path = Path(
            config["paths"].get(
                "clean_samples_full_jsonl",
                clean_output_dir / "clean_samples_full.jsonl",
            )
        )
        failure_path = Path(
            config["paths"].get(
                "clean_failures_full_jsonl",
                clean_output_dir / "clean_failures_full.jsonl",
            )
        )
        preview_path = clean_output_dir / "clean_preview_full.txt"
        summary_path = clean_output_dir / "clean_summary_full.json"

    source_models = get_source_models(config, args.mode)
    domains = get_domains(config, args.mode)
    lengths = get_lengths(config, args.mode)
    clean_replicates_per_prompt = get_clean_replicates(config, args.mode)

    temperature = float(config["generation"]["temperature"])
    top_p = float(config["generation"]["top_p"])
    max_regeneration_attempts = int(config["generation"]["max_regeneration_attempts"])
    extra_new_tokens = int(
        config["generation"].get("extra_new_tokens_for_length_control", 40)
    )
    base_seed = int(config.get("seed", 42))
    manifest_version = str(
        config.get("metadata", {}).get("manifest_version", "pilot_v0")
    )
    length_policy = str(
        config["generation"].get(
            "length_policy",
            "complete_sentence_near_nominal_target",
        )
    )
    length_tolerance_by_target = {
        int(target): int(tolerance)
        for target, tolerance in config["generation"]
        .get("length_tolerance_by_target", {64: 8, 256: 16})
        .items()
    }

    all_prompts = read_jsonl(rendered_prompts_path)
    selected_prompts = filter_standard_prompts(
        all_prompts,
        domains=domains,
        lengths=lengths,
    )

    if args.limit is not None:
        selected_prompts = selected_prompts[: args.limit]

    expected_tasks: dict[str, dict] = {}
    for source_model in source_models:
        for prompt_record in selected_prompts:
            for replicate_index in range(clean_replicates_per_prompt):
                sample_id, _ = make_clean_sample_ids(
                    prompt_record=prompt_record,
                    source_model=source_model,
                    replicate_index=replicate_index,
                )

                if sample_id in expected_tasks:
                    raise ValueError(
                        "Sample ID collision in the current design: "
                        f"{sample_id}. Use unique source-model aliases."
                    )

                expected_tasks[sample_id] = {
                    "source_model": source_model,
                    "prompt_id": prompt_record["prompt_id"],
                    "prompt": prompt_record["visible_prompt"],
                    "target_length": int(prompt_record["target_length"]),
                    "replicate_index": replicate_index,
                }

    expected_sample_ids = set(expected_tasks)
    expected_clean_records = len(expected_tasks)

    if args.overwrite:
        output_path.unlink(missing_ok=True)
        failure_path.unlink(missing_ok=True)
        preview_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    elif output_path.exists() and not args.resume:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --resume to continue "
            "or --overwrite to start over."
        )

    if args.resume:
        clean_records, completed_sample_ids = load_existing_records(
            output_path=output_path,
            expected_tasks=expected_tasks,
            manifest_version=manifest_version,
            temperature=temperature,
            top_p=top_p,
            length_policy=length_policy,
            length_tolerance_by_target=length_tolerance_by_target,
        )
    else:
        clean_records = []
        completed_sample_ids = set()

    records_at_start = len(clean_records)
    new_records_this_run = 0
    failures_this_run: list[dict] = []

    print("Clean generation configuration")
    print("=" * 40)
    print(f"mode: {args.mode}")
    print(f"resume: {args.resume}")
    print(f"overwrite: {args.overwrite}")
    print(f"source_models: {source_models}")
    print(f"domains: {domains}")
    print(f"lengths: {lengths}")
    print(f"selected standard prompts per model: {len(selected_prompts)}")
    print(f"clean_replicates_per_prompt: {clean_replicates_per_prompt}")
    print(f"temperature: {temperature}")
    print(f"top_p: {top_p}")
    print(f"max_regeneration_attempts: {max_regeneration_attempts}")
    print(f"extra_new_tokens: {extra_new_tokens}")
    print(f"manifest_version: {manifest_version}")
    print(f"length_policy: {length_policy}")
    print(f"length_tolerance_by_target: {length_tolerance_by_target}")
    print(f"expected_clean_records: {expected_clean_records}")
    print(f"records_already_complete: {records_at_start}")
    print("=" * 40)

    for model_index, source_model in enumerate(source_models):
        pending_for_model = 0
        for prompt_record in selected_prompts:
            for replicate_index in range(clean_replicates_per_prompt):
                sample_id, _ = make_clean_sample_ids(
                    prompt_record=prompt_record,
                    source_model=source_model,
                    replicate_index=replicate_index,
                )
                if sample_id not in completed_sample_ids:
                    pending_for_model += 1

        if pending_for_model == 0:
            print(
                f"Skipping source model because all records are complete: "
                f"{source_model}"
            )
            continue

        print(
            f"\nLoading source model: {source_model} "
            f"({pending_for_model} pending records)"
        )

        (
            model,
            tokenizer,
            model_revision,
            tokenizer_revision,
        ) = load_model_and_tokenizer(source_model)

        existing_model_revisions = {
            record.get("model_revision")
            for record in clean_records
            if record.get("source_model") == source_model
            and record.get("model_revision") is not None
        }
        if (
            existing_model_revisions
            and model_revision is not None
            and model_revision not in existing_model_revisions
        ):
            raise ValueError(
                f"Model revision changed while resuming {source_model}: "
                f"existing={sorted(existing_model_revisions)}, "
                f"current={model_revision}."
            )

        try:
            for prompt_index, prompt_record in enumerate(
                tqdm(
                    selected_prompts,
                    desc=f"Generating with {source_model}",
                )
            ):
                target_length = int(prompt_record["target_length"])
                if target_length not in length_tolerance_by_target:
                    raise ValueError(
                        f"No length tolerance configured for target {target_length}."
                    )
                length_tolerance = length_tolerance_by_target[target_length]

                for replicate_index in range(clean_replicates_per_prompt):
                    sample_id, _ = make_clean_sample_ids(
                        prompt_record=prompt_record,
                        source_model=source_model,
                        replicate_index=replicate_index,
                    )

                    if sample_id in completed_sample_ids:
                        continue

                    seed_offset = (
                        model_index
                        * len(selected_prompts)
                        * clean_replicates_per_prompt
                        + prompt_index * clean_replicates_per_prompt
                        + replicate_index
                    )

                    try:
                        result = generate_length_controlled_clean_text(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt_record["visible_prompt"],
                            target_length=target_length,
                            temperature=temperature,
                            top_p=top_p,
                            max_regeneration_attempts=max_regeneration_attempts,
                            extra_new_tokens=extra_new_tokens,
                            length_tolerance=length_tolerance,
                            seed=base_seed + seed_offset,
                        )

                        record = make_clean_sample_record(
                            prompt_record=prompt_record,
                            source_model=source_model,
                            model_revision=model_revision,
                            tokenizer_revision=tokenizer_revision,
                            manifest_version=manifest_version,
                            text=result["text"],
                            actual_token_length=result["actual_token_length"],
                            raw_token_length_before_adjustment=(
                                result["raw_token_length_before_adjustment"]
                            ),
                            length_adjustment=result["length_adjustment"],
                            length_tolerance=result["length_tolerance"],
                            length_policy=result["length_policy"],
                            ending_valid=result["ending_valid"],
                            generation_attempts=result["generation_attempts"],
                            generation_seed=result["generation_seed"],
                            temperature=temperature,
                            top_p=top_p,
                            replicate_index=replicate_index,
                        )

                        if record["sample_id"] != sample_id:
                            raise RuntimeError(
                                "Precomputed sample_id does not match record sample_id: "
                                f"{sample_id} != {record['sample_id']}"
                            )

                        append_jsonl_record(record, output_path)
                        clean_records.append(record)
                        completed_sample_ids.add(sample_id)
                        new_records_this_run += 1

                    except Exception as exc:
                        failure_record = {
                            "source_model": source_model,
                            "prompt_id": prompt_record.get("prompt_id"),
                            "topic_id": prompt_record.get("topic_id"),
                            "target_length": target_length,
                            "replicate_index": replicate_index,
                            "sample_id": sample_id,
                            "error": str(exc),
                        }
                        failures_this_run.append(failure_record)
                        append_jsonl_record(failure_record, failure_path)

        finally:
            print(f"Unloading source model: {source_model}")

            del model
            del tokenizer
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    write_clean_preview(clean_records, preview_path, limit=10)

    remaining_records = expected_clean_records - len(completed_sample_ids)
    summary = {
        "mode": args.mode,
        "resume": args.resume,
        "overwrite": args.overwrite,
        "num_clean_records": len(clean_records),
        "records_at_start": records_at_start,
        "new_records_this_run": new_records_this_run,
        "num_failures_this_run": len(failures_this_run),
        "remaining_records": remaining_records,
        "complete": remaining_records == 0,
        "source_models": source_models,
        "domains": domains,
        "lengths": lengths,
        "output_path": str(output_path),
        "failure_path": str(failure_path),
        "preview_path": str(preview_path),
        "failures_this_run": failures_this_run,
        "clean_replicates_per_prompt": clean_replicates_per_prompt,
        "length_policy": length_policy,
        "length_tolerance_by_target": length_tolerance_by_target,
        "expected_clean_records": expected_clean_records,
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nClean generation run complete.")
    print(f"Accepted records preserved at: {output_path}")
    print(f"Failure attempts logged at: {failure_path}")
    print(f"Wrote: {preview_path}")
    print(f"Wrote: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
