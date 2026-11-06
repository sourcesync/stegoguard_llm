from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

from stegoguard_lm.detection.datasets import validate_detection_dataset_records
from stegoguard_lm.detection.io_utils import read_json, read_jsonl, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen Step 8 detection dataset.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    paths = config["paths"]
    records = read_jsonl(paths["detection_dataset_jsonl"])
    manifest = read_json(paths["detection_dataset_manifest_json"])
    validate_detection_dataset_records(records)

    expected_total = int(config["static_detection"]["expected_detection_records"])
    if len(records) != expected_total:
        raise ValueError(f"Expected {expected_total} detection rows, found {len(records)}.")
    if sha256_file(paths["detection_dataset_jsonl"]) != manifest["dataset_sha256"]:
        raise ValueError("Detection dataset hash does not match its manifest.")

    label_counts = Counter(row["label"] for row in records)
    attack_counts = Counter(row["attack_name"] for row in records if row["label"] == "stego")
    common_clean = sum(
        row["label"] == "clean" and row["is_common_all_three"] for row in records
    )
    balanced_clean = sum(
        row["label"] == "clean" and row["is_balanced_common"] for row in records
    )
    if common_clean != int(manifest["common_parent_count_all_three"]):
        raise ValueError("Common-parent count mismatch.")
    if balanced_clean != int(manifest["balanced_common_parent_count"]):
        raise ValueError("Balanced common-parent count mismatch.")

    print("Step 8 detection dataset validation passed.")
    print(f"Total records: {len(records)}")
    print(f"Clean: {label_counts['clean']}")
    print(f"Stego: {label_counts['stego']}")
    print("Stego counts by attack:")
    for key, value in sorted(attack_counts.items()):
        print(f"  {key}: {value}")
    print(f"Common all-three parents: {common_clean}")
    print(f"Balanced common parents: {balanced_clean}")


if __name__ == "__main__":
    main()
