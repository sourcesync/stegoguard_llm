from __future__ import annotations

import argparse
import gc
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

from stegoguard_lm.attacks.payloads import payload_sha256, validate_bitstring
from stegoguard_lm.attacks.rank_channel import decode_rank_payload
from stegoguard_lm.attacks.synonym_channel import (
    decode_synonym_payload,
    load_synonym_codebook,
)
from stegoguard_lm.attacks.unicode_channel import decode_unicode_payload
from stegoguard_lm.attacks.validation import validate_attack_record_against_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Step 6 attack smoke JSONL.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--samples",
        default="outputs/pilot_v0/attacks/smoke/attack_smoke_samples.jsonl",
    )
    parser.add_argument(
        "--clean-samples",
        default="outputs/pilot_v0/clean/clean_samples_full.jsonl",
    )
    parser.add_argument(
        "--expected-count",
        "--expected_count",
        dest="expected_count",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--verify-rank-with-model",
        action="store_true",
        help="Independently rerun text-based rank decoding with the source model.",
    )
    return parser.parse_args()


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


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    return config


def load_tokenizer(model_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for independent tokenization."
        ) from exc

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
    attacks_config = config["attacks"]

    attack_records = read_jsonl(args.samples)
    clean_records = read_jsonl(args.clean_samples)
    clean_by_id = {record["sample_id"]: record for record in clean_records}
    if len(clean_by_id) != len(clean_records):
        raise ValueError("Clean corpus contains duplicate sample_id values.")

    failures: list[str] = []
    seen_sample_ids: set[str] = set()
    seen_comparison_ids: set[str] = set()
    tokenizer_cache: dict[str, Any] = {}
    codebook = load_synonym_codebook(config["paths"]["synonym_codebook_yaml"])

    rank_records_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for line_number, record in enumerate(attack_records, start=1):
        sample_id = str(record.get("sample_id", ""))
        prefix = f"Line {line_number} ({sample_id or 'missing sample_id'})"

        if sample_id in seen_sample_ids:
            failures.append(f"{prefix}: duplicate sample_id.")
        seen_sample_ids.add(sample_id)

        comparison_id = str(record.get("comparison_id", ""))
        if comparison_id in seen_comparison_ids:
            failures.append(f"{prefix}: duplicate comparison_id.")
        seen_comparison_ids.add(comparison_id)

        parent_id = record.get("parent_sample_id")
        parent = clean_by_id.get(parent_id)
        if parent is None:
            failures.append(f"{prefix}: clean parent not found: {parent_id!r}.")
            continue

        for error in validate_attack_record_against_parent(record, parent):
            failures.append(f"{prefix}: {error}")

        try:
            validate_bitstring(str(record.get("payload", "")))
        except Exception as exc:
            failures.append(f"{prefix}: invalid payload: {exc}")

        try:
            if payload_sha256(record["payload"]) != record["payload_sha256"]:
                failures.append(f"{prefix}: payload_sha256 mismatch.")
        except Exception as exc:
            failures.append(f"{prefix}: could not verify payload hash: {exc}")

        source_model = record.get("source_model")
        if source_model not in tokenizer_cache:
            tokenizer_cache[source_model] = load_tokenizer(source_model)
        tokenizer = tokenizer_cache[source_model]

        family = record.get("attack_family")
        if family in {"token_blackbox", "token_whitebox"}:
            actual_length = len(
                tokenizer.encode(record["text"], add_special_tokens=False)
            )
            if actual_length != int(parent["actual_token_length"]):
                failures.append(
                    f"{prefix}: independent token count {actual_length} does not "
                    f"match parent length {parent['actual_token_length']}."
                )
            if actual_length != int(record["stego_actual_token_length"]):
                failures.append(
                    f"{prefix}: independent token count does not match stored "
                    "stego_actual_token_length."
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
                    failures.append(f"{prefix}: independent Unicode payload mismatch.")
                if decoded.stripped_text != parent["text"]:
                    failures.append(
                        f"{prefix}: stripped Unicode text differs from parent."
                    )

                raw_length = len(
                    tokenizer.encode(record["text"], add_special_tokens=False)
                )
                stripped_length = len(
                    tokenizer.encode(decoded.stripped_text, add_special_tokens=False)
                )
                if raw_length != int(record["raw_stego_source_token_length"]):
                    failures.append(
                        f"{prefix}: raw Unicode token-length metadata mismatch."
                    )
                if stripped_length != int(
                    record["visible_text_source_token_length_after_strip"]
                ):
                    failures.append(
                        f"{prefix}: stripped Unicode token-length metadata mismatch."
                    )
                if stripped_length != int(parent["actual_token_length"]):
                    failures.append(
                        f"{prefix}: stripped Unicode text does not preserve parent length."
                    )
            except Exception as exc:
                failures.append(f"{prefix}: Unicode validation failed: {exc}")

        elif family == "token_blackbox":
            parameters = record.get("channel_parameters", {})
            active_slot_ids = parameters.get("active_slot_ids")
            decoded = decode_synonym_payload(
                record["text"],
                codebook,
                active_slot_ids=active_slot_ids,
            )
            if not decoded.success:
                failures.append(
                    f"{prefix}: independent synonym decode failed: {decoded.errors}"
                )
            elif decoded.payload != record["payload"]:
                failures.append(f"{prefix}: independent synonym payload mismatch.")

            if active_slot_ids is not None:
                if len(active_slot_ids) != int(record.get("payload_bit_length", -1)):
                    failures.append(
                        f"{prefix}: active synonym slot count does not match "
                        "payload_bit_length."
                    )
                changed_slot_ids = parameters.get("changed_slot_ids", [])
                if bool(
                    attacks_config["token_blackbox"].get("require_text_change", True)
                ):
                    if record["text"] == parent["text"]:
                        failures.append(
                            f"{prefix}: V4 synonym stego text is identical to clean parent."
                        )
                    if not changed_slot_ids:
                        failures.append(
                            f"{prefix}: V4 synonym record has no changed_slot_ids."
                        )
                if record.get("strategy_id") != "synonym_inplace_substitution":
                    failures.append(
                        f"{prefix}: synonym strategy_id is not the V4 in-place strategy."
                    )

        elif family == "token_whitebox":
            rank_records_by_model[source_model].append(record)

    if args.verify_rank_with_model:
        for source_model, records in rank_records_by_model.items():
            model, tokenizer, _, _ = load_model_and_tokenizer(source_model)
            try:
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
                        failures.append(
                            f"{record['sample_id']}: model-based rank decode failed: "
                            f"{decoded.errors}"
                        )
                    elif decoded.payload != record["payload"]:
                        failures.append(
                            f"{record['sample_id']}: model-based rank payload mismatch."
                        )
            finally:
                model = None
                tokenizer = None
                collect_model_memory()

    if not attack_records:
        failures.append("Attack smoke JSONL contains no records.")
    if args.expected_count is not None and len(attack_records) != args.expected_count:
        failures.append(
            f"Expected {args.expected_count} records, found {len(attack_records)}."
        )

    if failures:
        print("Attack smoke validation failed.")
        for failure in failures[:100]:
            print(f"- {failure}")
        if len(failures) > 100:
            print(f"- ... and {len(failures) - 100} additional failures")
        raise SystemExit(1)

    print("Attack smoke validation passed.")
    print(f"Total attack records: {len(attack_records)}")
    print("Counts by attack:")
    for attack_name, count in sorted(
        Counter(record["attack_name"] for record in attack_records).items()
    ):
        print(f"  {attack_name}: {count}")
    print("Counts by source model:")
    for source_model, count in sorted(
        Counter(record["source_model"] for record in attack_records).items()
    ):
        print(f"  {source_model}: {count}")
    if rank_records_by_model and not args.verify_rank_with_model:
        print(
            "Rank records passed structural and token-length checks. "
            "Add --verify-rank-with-model for independent model-based decoding."
        )


if __name__ == "__main__":
    main()
