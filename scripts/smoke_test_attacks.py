from __future__ import annotations

import argparse
import gc
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

from stegoguard_lm.attacks.payloads import (
    derive_deterministic_int,
    derive_payload_bits,
)
from stegoguard_lm.attacks.rank_channel import encode_rank_payload
from stegoguard_lm.attacks.records import (
    build_attack_record,
    make_attack_sample_id,
)
from stegoguard_lm.attacks.synonym_channel import (
    discover_inplace_synonym_slots,
    encode_synonym_payload,
    load_synonym_codebook,
)
from stegoguard_lm.attacks.unicode_channel import encode_unicode_payload

CHANNEL_ALIASES = {
    "unicode": "out_of_band",
    "synonym": "token_blackbox",
    "rank": "token_whitebox",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable Step 6 attack smoke tests."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--channel",
        choices=["unicode", "synonym", "rank", "all"],
        default="all",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Clean parent sample ID for --matrix single.",
    )
    parser.add_argument(
        "--matrix",
        choices=["single", "core"],
        default="single",
    )

    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="Preserve accepted smoke records and generate only missing tasks.",
    )
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete previous smoke outputs before running.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    return config


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
    return records


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        # ensure_ascii makes invisible Unicode channel symbols auditable on disk.
        file.write(json.dumps(record, ensure_ascii=True) + "\n")
        file.flush()
        os.fsync(file.fileno())


def load_existing_attack_records(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()

    records = read_jsonl(path)
    seen: set[str] = set()
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"Existing attack record lacks sample_id: {record}")
        if sample_id in seen:
            raise ValueError(f"Duplicate existing attack sample_id: {sample_id}")
        if record.get("validation_status") != "accepted":
            raise ValueError(
                f"Existing record is not accepted and cannot be resumed: {sample_id}"
            )
        seen.add(sample_id)
    return records, seen


def load_tokenizer(model_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for attack smoke integration tests."
        ) from exc

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(model_id: str):
    # Lazy import keeps model-free unit tests and --help lightweight.
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


def requested_attack_families(channel: str) -> list[str]:
    if channel == "all":
        return ["out_of_band", "token_blackbox", "token_whitebox"]
    return [CHANNEL_ALIASES[channel]]


def choose_parent_ids(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> list[str]:
    smoke_config = config["attacks"].get("smoke", {})
    core_ids = list(smoke_config.get("core_sample_ids", []))
    if not core_ids:
        core_ids = [
            "clean_conv_001_64_qwen_000",
            "clean_tech_001_256_qwen_000",
            "clean_conv_001_64_mistral_000",
            "clean_tech_001_256_mistral_000",
        ]

    if args.matrix == "core":
        if args.sample_id is not None:
            raise ValueError("--sample-id cannot be combined with --matrix core.")
        return core_ids

    if args.sample_id is not None:
        return [args.sample_id]
    return [str(smoke_config.get("default_sample_id", core_ids[0]))]


def attack_name_for_family(config: dict[str, Any], family: str) -> str:
    return str(config["attacks"][family]["name"])


def write_preview(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record in records:
        text = str(record["text"])
        if record["attack_family"] == "out_of_band":
            visible = text.encode("unicode_escape").decode("ascii")
        else:
            visible = text
        lines.extend(
            [
                "=" * 88,
                f"sample_id: {record['sample_id']}",
                f"parent_sample_id: {record['parent_sample_id']}",
                f"attack: {record['attack_family']} / {record['attack_name']}",
                f"source_model: {record['source_model']}",
                f"payload: {record['payload']}",
                f"decoded_payload: {record['decoded_payload']}",
                f"length_invariant: {record['length_invariant']}",
                f"parent_actual_token_length: {record['parent_actual_token_length']}",
                f"stego_actual_token_length: {record.get('stego_actual_token_length')}",
                "text:",
                visible,
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    *,
    records: list[dict[str, Any]],
    requested_sample_ids: set[str],
    records_at_start: int,
    new_records: int,
    failures_this_run: list[dict[str, Any]],
    output_path: Path,
    failure_path: Path,
    summary_path: Path,
    preview_path: Path,
) -> dict[str, Any]:
    present_ids = {record["sample_id"] for record in records}
    remaining = sorted(requested_sample_ids - present_ids)
    summary = {
        "records_at_start": records_at_start,
        "new_records_this_run": new_records,
        "num_attack_records": len(records),
        "requested_tasks": len(requested_sample_ids),
        "requested_tasks_remaining": len(remaining),
        "requested_task_ids_remaining": remaining,
        "num_failures_this_run": len(failures_this_run),
        "failures_this_run": failures_this_run,
        "complete_for_requested_tasks": not remaining,
        "counts_by_attack": dict(
            sorted(Counter(record["attack_name"] for record in records).items())
        ),
        "counts_by_source_model": dict(
            sorted(Counter(record["source_model"] for record in records).items())
        ),
        "exact_decode_success_rate": (
            sum(bool(record.get("exact_match")) for record in records) / len(records)
            if records
            else None
        ),
        "output_path": str(output_path),
        "failure_path": str(failure_path),
        "preview_path": str(preview_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    paths = config["paths"]
    attacks_config = config["attacks"]
    manifest_version = str(config["metadata"]["manifest_version"])

    clean_path = Path(paths["clean_samples_full_jsonl"])
    output_path = Path(paths["attack_smoke_samples_jsonl"])
    failure_path = Path(paths["attack_smoke_failures_jsonl"])
    preview_path = Path(paths["attack_smoke_preview_txt"])
    summary_path = Path(paths["attack_smoke_summary_json"])

    if args.overwrite:
        for path in (output_path, failure_path, preview_path, summary_path):
            path.unlink(missing_ok=True)
    elif output_path.exists() and not args.resume:
        raise FileExistsError(
            f"Smoke output already exists: {output_path}. Use --resume or --overwrite."
        )

    clean_records = read_jsonl(clean_path)
    clean_by_id = {record["sample_id"]: record for record in clean_records}
    if len(clean_by_id) != len(clean_records):
        raise ValueError("Clean corpus contains duplicate sample_id values.")

    parent_ids = choose_parent_ids(args=args, config=config)
    missing_parents = [
        sample_id for sample_id in parent_ids if sample_id not in clean_by_id
    ]
    if missing_parents:
        raise KeyError(f"Requested clean parent IDs were not found: {missing_parents}")

    families = requested_attack_families(args.channel)
    tasks: list[tuple[dict[str, Any], str]] = []
    requested_output_ids: set[str] = set()
    for parent_id in parent_ids:
        parent = clean_by_id[parent_id]
        for family in families:
            if not bool(attacks_config[family].get("enabled", False)):
                raise ValueError(f"Requested attack family is disabled: {family}")
            attack_name = attack_name_for_family(config, family)
            expected_id = make_attack_sample_id(parent, attack_name)
            if expected_id in requested_output_ids:
                raise ValueError(f"Duplicate requested smoke task ID: {expected_id}")
            requested_output_ids.add(expected_id)
            tasks.append((parent, family))

    if args.resume:
        attack_records, completed_ids = load_existing_attack_records(output_path)
    else:
        attack_records, completed_ids = [], set()

    records_at_start = len(attack_records)
    new_records = 0
    failures_this_run: list[dict[str, Any]] = []

    codebook = None
    if "token_blackbox" in families:
        codebook = load_synonym_codebook(paths["synonym_codebook_yaml"])

    payload_config = attacks_config["payload"]
    base_seed = int(payload_config["base_seed"])
    generation_config = config["generation"]

    tasks_by_model: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for parent, family in tasks:
        attack_name = attack_name_for_family(config, family)
        expected_id = make_attack_sample_id(parent, attack_name)
        if expected_id not in completed_ids:
            tasks_by_model[parent["source_model"]].append((parent, family))

    if not tasks_by_model:
        print("All requested attack smoke records are already complete.")

    for source_model, model_tasks in tasks_by_model.items():
        # V4 synonym substitution operates directly on the frozen clean cover
        # and needs only the public source tokenizer. Only rank encoding needs
        # the causal LM weights/logits.
        needs_model = any(family == "token_whitebox" for _, family in model_tasks)

        model = None
        tokenizer = None
        if needs_model:
            model, tokenizer, _, _ = load_model_and_tokenizer(source_model)
        else:
            tokenizer = load_tokenizer(source_model)

        try:
            for parent, family in model_tasks:
                attack_settings = attacks_config[family]
                attack_name = str(attack_settings["name"])
                expected_id = make_attack_sample_id(parent, attack_name)
                parent_id = parent["sample_id"]
                required_length = int(parent["actual_token_length"])

                attack_seed = derive_deterministic_int(
                    namespace=attack_name,
                    parent_sample_id=parent_id,
                    base_seed=base_seed,
                )

                # Fixed-capacity channels derive their payload here. The V4
                # synonym channel has cover-dependent capacity, so its payload
                # is derived inside the channel branch after eligible slots are
                # discovered with the source tokenizer.
                payload = None
                if family == "out_of_band":
                    num_bits = int(attack_settings["payload_bits"])
                    payload = derive_payload_bits(
                        manifest_version=manifest_version,
                        parent_sample_id=parent_id,
                        attack_name=attack_name,
                        base_seed=base_seed,
                        num_bits=num_bits,
                    )
                elif family == "token_whitebox":
                    num_bits = required_length
                    payload = derive_payload_bits(
                        manifest_version=manifest_version,
                        parent_sample_id=parent_id,
                        attack_name=attack_name,
                        base_seed=base_seed,
                        num_bits=num_bits,
                    )

                try:
                    if family == "out_of_band":
                        result = encode_unicode_payload(
                            clean_text=parent["text"],
                            payload=payload,
                            tokenizer=tokenizer,
                            zero_codepoint=str(attack_settings["bit_zero_codepoint"]),
                            one_codepoint=str(attack_settings["bit_one_codepoint"]),
                            prefix_codepoint=str(attack_settings["prefix_codepoint"]),
                            suffix_codepoint=str(attack_settings["suffix_codepoint"]),
                            framing_version=str(attack_settings["framing_version"]),
                            embed_position=str(attack_settings["embed_position"]),
                        )
                    elif family == "token_blackbox":
                        assert tokenizer is not None
                        assert codebook is not None

                        eligible = discover_inplace_synonym_slots(
                            clean_text=parent["text"],
                            tokenizer=tokenizer,
                            codebook=codebook,
                            required_token_length=required_length,
                            require_complete_ending=bool(
                                attack_settings["require_complete_ending"]
                            ),
                            require_tokenizer_roundtrip=bool(
                                attack_settings["require_tokenizer_roundtrip"]
                            ),
                            forbid_format_codepoints=bool(
                                attack_settings.get("forbid_format_codepoints", True)
                            ),
                        )
                        max_payload_bits = min(
                            int(attack_settings.get("max_payload_bits", 8)),
                            len(eligible),
                        )
                        min_payload_bits = int(
                            attack_settings.get("min_payload_bits", 1)
                        )
                        if max_payload_bits < min_payload_bits:
                            raise RuntimeError(
                                "no_capacity: clean cover has "
                                f"{len(eligible)} bidirectionally length-stable "
                                "synonym slots."
                            )

                        result = None
                        last_synonym_error = "payload selection not attempted"
                        nonce_attempts = int(
                            attack_settings.get("payload_nonce_attempts", 16)
                        )
                        for payload_nonce in range(nonce_attempts):
                            requested_payload = derive_payload_bits(
                                manifest_version=manifest_version,
                                parent_sample_id=(
                                    f"{parent_id}|synonym_nonce={payload_nonce}"
                                ),
                                attack_name=attack_name,
                                base_seed=base_seed,
                                num_bits=max_payload_bits,
                            )
                            try:
                                candidate = encode_synonym_payload(
                                    clean_text=parent["text"],
                                    tokenizer=tokenizer,
                                    payload=requested_payload,
                                    codebook=codebook,
                                    required_token_length=required_length,
                                    max_payload_bits=max_payload_bits,
                                    min_payload_bits=min_payload_bits,
                                    require_text_change=bool(
                                        attack_settings.get("require_text_change", True)
                                    ),
                                    forbid_format_codepoints=bool(
                                        attack_settings.get(
                                            "forbid_format_codepoints", True
                                        )
                                    ),
                                    require_complete_ending=bool(
                                        attack_settings["require_complete_ending"]
                                    ),
                                    require_tokenizer_roundtrip=bool(
                                        attack_settings["require_tokenizer_roundtrip"]
                                    ),
                                )
                            except RuntimeError as exc:
                                last_synonym_error = str(exc)
                                continue

                            candidate.channel_parameters.update(
                                {
                                    "payload_nonce": payload_nonce,
                                    "payload_nonce_attempts": payload_nonce + 1,
                                    "requested_payload": requested_payload,
                                    "max_payload_bits_configured": int(
                                        attack_settings.get("max_payload_bits", 8)
                                    ),
                                }
                            )
                            result = candidate
                            break

                        if result is None:
                            raise RuntimeError(
                                "Synonym payload selection failed after "
                                f"{nonce_attempts} deterministic nonce attempts. "
                                f"Last error: {last_synonym_error}"
                            )
                    elif family == "token_whitebox":
                        assert model is not None and tokenizer is not None
                        result = encode_rank_payload(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=parent["prompt"],
                            payload=payload,
                            required_token_length=required_length,
                            attack_seed=attack_seed,
                            top_k=int(attack_settings["top_k"]),
                            max_encoding_attempts=int(
                                attack_settings["max_encoding_attempts"]
                            ),
                            final_candidate_search_k=int(
                                attack_settings["final_candidate_search_k"]
                            ),
                            require_complete_ending=bool(
                                attack_settings["require_complete_ending"]
                            ),
                            require_tokenizer_roundtrip=bool(
                                attack_settings["require_tokenizer_roundtrip"]
                            ),
                        )
                    else:
                        raise ValueError(f"Unsupported attack family: {family}")

                    record = build_attack_record(
                        parent_record=parent,
                        result=result,
                        manifest_version=manifest_version,
                        attack_seed=attack_seed,
                    )
                    if record["sample_id"] != expected_id:
                        raise RuntimeError(
                            f"Expected sample_id {expected_id}, got {record['sample_id']}."
                        )

                    append_jsonl(record, output_path)
                    attack_records.append(record)
                    completed_ids.add(record["sample_id"])
                    new_records += 1
                    print(f"Accepted: {record['sample_id']}")

                except Exception as exc:
                    failure = {
                        "sample_id": expected_id,
                        "parent_sample_id": parent_id,
                        "pair_id": parent["pair_id"],
                        "attack_family": family,
                        "attack_name": attack_name,
                        "source_model": source_model,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "attack_seed": attack_seed,
                        "manifest_version": manifest_version,
                    }
                    append_jsonl(failure, failure_path)
                    failures_this_run.append(failure)
                    print(f"Failed: {expected_id}: {exc}")
        finally:
            # Drop caller-held references before collecting memory.
            model = None
            tokenizer = None
            collect_model_memory()

    # Reload from disk so summary/preview reflect the exact persisted corpus.
    persisted_records = read_jsonl(output_path) if output_path.exists() else []
    write_preview(persisted_records, preview_path)
    summary = write_summary(
        records=persisted_records,
        requested_sample_ids=requested_output_ids,
        records_at_start=records_at_start,
        new_records=new_records,
        failures_this_run=failures_this_run,
        output_path=output_path,
        failure_path=failure_path,
        summary_path=summary_path,
        preview_path=preview_path,
    )

    print("Attack smoke run complete.")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
