from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REQUIRED_FIELDS = {
    "sample_id",
    "pair_id",
    "lineage_id",
    "label",
    "experiment_stage",
    "round_id",
    "domain",
    "topic_id",
    "topic_seed",
    "target_length",
    "nominal_target_length",
    "actual_token_length",
    "length_delta",
    "length_tolerance",
    "length_policy",
    "source_model",
    "model_revision",
    "tokenizer_revision",
    "text",
    "prompt",
    "prompt_id",
    "prompt_type",
    "template_version",
    "replicate_index",
    "length_adjustment",
    "ending_valid",
    "manifest_version",
    "seed",
    "temperature",
    "top_p",
    "raw_token_length_before_adjustment",
    "generation_attempts",
    "length_valid",
    "generation_status",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate clean sample JSONL.")
    parser.add_argument(
        "--samples",
        type=str,
        default="outputs/pilot_v0/clean/clean_samples_mini.jsonl",
    )
    parser.add_argument(
        "--expected_count",
        type=int,
        default=None,
        help="Expected number of clean samples in the file.",
    )
    return parser.parse_args()


def has_complete_ending(text: str) -> bool:
    return bool(re.search(r"[.!?][\"')\]]*$", text.rstrip()))


def main():
    args = parse_args()
    path = Path(args.samples)

    if not path.exists():
        raise FileNotFoundError(f"Clean samples file not found: {path}")

    seen_sample_ids = set()
    seen_pair_ids = set()
    total = 0
    failures = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            total += 1
            record = json.loads(line)

            missing = REQUIRED_FIELDS - set(record.keys())
            if missing:
                failures.append(f"Line {line_number}: missing fields {sorted(missing)}")
                continue

            if record["sample_id"] in seen_sample_ids:
                failures.append(
                    f"Line {line_number}: duplicate sample_id " f"{record['sample_id']}"
                )
            seen_sample_ids.add(record["sample_id"])

            if record["pair_id"] in seen_pair_ids:
                failures.append(
                    f"Line {line_number}: duplicate pair_id {record['pair_id']}"
                )
            seen_pair_ids.add(record["pair_id"])

            if record["label"] != "clean":
                failures.append(f"Line {line_number}: label is not clean")

            if not str(record["text"]).strip():
                failures.append(f"Line {line_number}: empty text")

            if not str(record["prompt"]).strip():
                failures.append(f"Line {line_number}: empty prompt")

            if record["lineage_id"] != record["pair_id"]:
                failures.append(
                    f"Line {line_number}: lineage_id does not match pair_id"
                )

            if record["experiment_stage"] != "clean_baseline":
                failures.append(
                    f"Line {line_number}: experiment_stage is not clean_baseline"
                )

            if int(record["round_id"]) != 0:
                failures.append(f"Line {line_number}: clean baseline round_id is not 0")

            if int(record["generation_attempts"]) < 1:
                failures.append(
                    f"Line {line_number}: generation_attempts must be at least 1"
                )

            target = int(record["target_length"])
            nominal = int(record["nominal_target_length"])
            actual = int(record["actual_token_length"])
            tolerance = int(record["length_tolerance"])
            delta = actual - target

            if nominal != target:
                failures.append(
                    f"Line {line_number}: nominal_target_length does not "
                    f"match target_length"
                )

            if int(record["length_delta"]) != delta:
                failures.append(
                    f"Line {line_number}: length_delta does not match "
                    f"actual minus target"
                )

            if abs(delta) > tolerance:
                failures.append(
                    f"Line {line_number}: actual length is outside the "
                    f"permitted target band"
                )

            if record["length_valid"] is not True:
                failures.append(f"Line {line_number}: length_valid is not true")

            if record["length_policy"] != ("complete_sentence_near_nominal_target"):
                failures.append(
                    f"Line {line_number}: unexpected length_policy "
                    f"{record['length_policy']!r}"
                )

            if record["generation_status"] != "accepted":
                failures.append(
                    f"Line {line_number}: generation_status is not accepted"
                )

            if record["prompt_type"] != "standard":
                failures.append(f"Line {line_number}: prompt_type is not standard")

            if not str(record["template_version"]).strip():
                failures.append(f"Line {line_number}: empty template_version")

            if int(record["replicate_index"]) < 0:
                failures.append(f"Line {line_number}: replicate_index is negative")

            expected_suffix = f"_{int(record['replicate_index']):03d}"
            if not str(record["pair_id"]).endswith(expected_suffix):
                failures.append(
                    f"Line {line_number}: pair_id does not match " f"replicate_index"
                )

            if record["ending_valid"] is not True:
                failures.append(f"Line {line_number}: ending_valid is not true")

            if not has_complete_ending(str(record["text"])):
                failures.append(
                    f"Line {line_number}: text does not end with "
                    f"sentence-closing punctuation"
                )

    if total == 0:
        failures.append("Clean sample file contains no records")

    if args.expected_count is not None and total != args.expected_count:
        failures.append(f"Expected {args.expected_count} records, found {total}")

    if failures:
        print("Clean sample validation failed.")
        for failure in failures[:50]:
            print(f"- {failure}")
        raise SystemExit(1)

    print("Clean sample validation passed.")
    print(f"Total clean records: {total}")


if __name__ == "__main__":
    main()
