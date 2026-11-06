from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from stegoguard_lm.detection.io_utils import read_json, read_jsonl, sha256_file
from stegoguard_lm.detection.splits import validate_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 8 leakage-safe folds.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    paths = config["paths"]
    records = read_jsonl(paths["detection_dataset_jsonl"])
    splits = read_json(paths["detection_splits_json"])
    if splits.get("detection_dataset_sha256") != sha256_file(paths["detection_dataset_jsonl"]):
        raise ValueError("Split file was built for a different detection dataset hash.")
    validate_splits(records, splits)
    expected_topics = int(config["static_detection"]["expected_topic_count"])
    if int(splits["topic_count"]) != expected_topics:
        raise ValueError(f"Expected {expected_topics} topics, found {splits['topic_count']}.")
    print("Step 8 split validation passed.")
    print(f"Strategy: {splits['split_strategy']}")
    print(f"Folds: {splits['fold_count']}")
    print("No sample, parent-lineage, or topic leakage detected.")


if __name__ == "__main__":
    main()
