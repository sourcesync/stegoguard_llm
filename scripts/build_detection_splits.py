from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from stegoguard_lm.detection.io_utils import atomic_write_json, read_jsonl, sha256_file
from stegoguard_lm.detection.splits import build_leave_one_topic_out_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe Step 8 topic-disjoint folds.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    paths = config["paths"]
    output = Path(paths["detection_splits_json"])
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Split file already exists: {output}. Use --overwrite intentionally.")
    records = read_jsonl(paths["detection_dataset_jsonl"])
    splits = build_leave_one_topic_out_splits(records)
    splits["detection_dataset_sha256"] = sha256_file(paths["detection_dataset_jsonl"])
    atomic_write_json(splits, output)
    print("Step 8 leave-one-topic-out splits built.")
    print(f"Topics: {splits['topic_count']}")
    print(f"Folds: {splits['fold_count']}")
    for fold in splits["folds"]:
        print(
            f"  {fold['fold_id']}: test_topic={fold['test_topic_id']}, "
            f"test_records={len(fold['test_sample_ids'])}"
        )
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
