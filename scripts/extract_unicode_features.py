from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from stegoguard_lm.detection.io_utils import atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_file
from stegoguard_lm.detection.unicode_features import extract_unicode_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract deterministic Unicode hygiene features.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    paths = config["paths"]
    output = Path(paths["detection_unicode_features_jsonl"])
    summary_path = Path(paths["detection_unicode_summary_json"])
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Unicode feature output already exists: {output}")
    dataset = read_jsonl(paths["detection_dataset_jsonl"])
    rows = []
    suspicious = 0
    for record in dataset:
        features = extract_unicode_features(str(record["text"]))
        suspicious += int(features["has_suspicious_codepoint"])
        rows.append({"sample_id": record["sample_id"], **features})
    atomic_write_jsonl(rows, output)
    summary = {
        "protocol_version": "step8_static_v1",
        "record_count": len(rows),
        "suspicious_record_count": suspicious,
        "dataset_sha256": sha256_file(paths["detection_dataset_jsonl"]),
        "output_sha256": sha256_file(output),
    }
    atomic_write_json(summary, summary_path)
    print("Unicode feature extraction complete.")
    print(f"Records: {len(rows)}")
    print(f"Records with suspicious codepoints: {suspicious}")
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
