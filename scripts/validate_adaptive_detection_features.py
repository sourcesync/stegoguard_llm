from __future__ import annotations

import argparse
from pathlib import Path

from stegoguard_lm.adaptive.io import load_yaml, verify_step10_protocol
from stegoguard_lm.detection.io_utils import atomic_write_json, read_jsonl, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 10 candidate detector feature completeness.")
    parser.add_argument("--config", default="config_step10.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    verify_step10_protocol(config)
    paths = config["paths"]
    dataset = read_jsonl(paths["adaptive_detection_dataset_jsonl"])
    expected = {str(r["sample_id"]) for r in dataset}
    files = {
        "evaluator_a": paths["adaptive_detection_evaluator_a_features_jsonl"],
        "evaluator_b": paths["adaptive_detection_evaluator_b_features_jsonl"],
        "semantic": paths["adaptive_detection_semantic_features_jsonl"],
        "unicode": paths["adaptive_detection_unicode_features_jsonl"],
    }
    errors = []
    hashes = {}
    for label, path in files.items():
        p = Path(path)
        if not p.exists():
            errors.append(f"Missing {label} feature file: {path}")
            continue
        rows = read_jsonl(path)
        ids = [str(r["sample_id"]) for r in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"Duplicate {label} feature sample IDs.")
        if set(ids) != expected:
            errors.append(f"{label} feature IDs do not match adaptive detection dataset: missing={len(expected-set(ids))}, extra={len(set(ids)-expected)}")
        hashes[label] = sha256_file(path)
    if errors:
        print("Adaptive detection feature validation failed.")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    manifest = {
        "protocol_version": config["adaptive_evaluation"]["protocol_version"],
        "record_count": len(dataset),
        "complete": True,
        "dataset_sha256": sha256_file(paths["adaptive_detection_dataset_jsonl"]),
        "feature_hashes": hashes,
    }
    atomic_write_json(manifest, paths["adaptive_detection_feature_manifest_json"])
    print("Adaptive detection feature validation passed.")
    print(f"Records: {len(dataset)}")


if __name__ == "__main__":
    main()
