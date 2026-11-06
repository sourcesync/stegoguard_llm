from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from stegoguard_lm.attacks.synonym_channel import (
    discover_inplace_synonym_slots,
    load_synonym_codebook,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit V4 in-place synonym capacity under each source tokenizer."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--matrix",
        choices=["core", "full"],
        default="core",
    )
    parser.add_argument(
        "--output",
        default="outputs/pilot_v0/manifests/synonym_inplace_capacity.json",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping.")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    paths = config["paths"]
    settings = config["attacks"]["token_blackbox"]
    codebook = load_synonym_codebook(paths["synonym_codebook_yaml"])
    clean_records = read_jsonl(paths["clean_samples_full_jsonl"])

    if args.matrix == "core":
        wanted = set(config["attacks"]["smoke"]["core_sample_ids"])
        records = [r for r in clean_records if r["sample_id"] in wanted]
        missing = sorted(wanted - {r["sample_id"] for r in records})
        if missing:
            raise KeyError(f"Core clean samples missing: {missing}")
    else:
        records = clean_records

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_model[record["source_model"]].append(record)

    rows: list[dict[str, Any]] = []
    for model_id, model_records in by_model.items():
        tokenizer = load_tokenizer(model_id)
        for record in model_records:
            eligible = discover_inplace_synonym_slots(
                clean_text=record["text"],
                tokenizer=tokenizer,
                codebook=codebook,
                required_token_length=int(record["actual_token_length"]),
                require_complete_ending=bool(settings["require_complete_ending"]),
                require_tokenizer_roundtrip=bool(settings["require_tokenizer_roundtrip"]),
                forbid_format_codepoints=bool(
                    settings.get("forbid_format_codepoints", True)
                ),
            )
            capacity = min(
                len(eligible),
                int(settings.get("max_payload_bits", 8)),
            )
            row = {
                "sample_id": record["sample_id"],
                "source_model": model_id,
                "domain": record["domain"],
                "nominal_target_length": int(record["target_length"]),
                "actual_token_length": int(record["actual_token_length"]),
                "eligible_slot_ids": [slot.slot_id for slot in eligible],
                "eligible_terms": [slot.original_term for slot in eligible],
                "capacity_bits": capacity,
                "has_capacity": capacity >= int(settings.get("min_payload_bits", 1)),
            }
            rows.append(row)
            print(
                f"{row['sample_id']}: capacity={capacity}, "
                f"slots={row['eligible_slot_ids']}"
            )

    capacities = [int(row["capacity_bits"]) for row in rows]
    summary = {
        "matrix": args.matrix,
        "record_count": len(rows),
        "codebook_version": codebook.version,
        "strategy_id": settings.get("strategy_id"),
        "capacity_distribution": dict(sorted(Counter(capacities).items())),
        "records_with_capacity": sum(bool(row["has_capacity"]) for row in rows),
        "records_without_capacity": sum(not bool(row["has_capacity"]) for row in rows),
        "mean_capacity_bits": (sum(capacities) / len(capacities)) if capacities else 0.0,
        "rows": rows,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote: {output}")

    if args.matrix == "core" and summary["records_without_capacity"]:
        raise SystemExit(
            "Core V4 synonym capacity audit failed: at least one core cover has no "
            "bidirectionally length-stable lexical slot."
        )


if __name__ == "__main__":
    main()
