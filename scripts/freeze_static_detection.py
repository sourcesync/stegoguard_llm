from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from stegoguard_lm.detection.io_utils import atomic_write_json, read_json, sha256_file

CODE_PATHS = [
    "src/stegoguard_lm/detection/datasets.py",
    "src/stegoguard_lm/detection/splits.py",
    "src/stegoguard_lm/detection/unicode_features.py",
    "src/stegoguard_lm/detection/evaluator_scoring.py",
    "src/stegoguard_lm/detection/semantic.py",
    "src/stegoguard_lm/detection/features.py",
    "src/stegoguard_lm/detection/modeling.py",
    "src/stegoguard_lm/detection/metrics.py",
    "scripts/build_detection_dataset.py",
    "scripts/build_detection_splits.py",
    "scripts/extract_unicode_features.py",
    "scripts/extract_evaluator_features.py",
    "scripts/extract_semantic_features.py",
    "scripts/validate_detection_features.py",
    "scripts/run_static_detection.py",
    "scripts/summarize_static_detection.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze and hash the completed Step 8 experiment.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    root = Path(".")
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    paths = config["paths"]
    feature_manifest = read_json(paths["detection_feature_manifest_json"])
    if not feature_manifest.get("complete"):
        raise ValueError("Cannot freeze Step 8: feature extraction is incomplete.")
    summary = read_json(paths["detection_summary_json"])
    required_artifacts = [
        paths["detection_dataset_jsonl"],
        paths["detection_dataset_manifest_json"],
        paths["detection_splits_json"],
        paths["detection_unicode_features_jsonl"],
        paths["detection_evaluator_a_features_jsonl"],
        paths["detection_evaluator_b_features_jsonl"],
        paths["detection_semantic_features_jsonl"],
        paths["detection_feature_manifest_json"],
        paths["detection_predictions_jsonl"],
        paths["detection_metrics_json"],
        paths["detection_summary_json"],
    ]
    artifact_hashes = {path: sha256_file(path) for path in required_artifacts}
    code_hashes = {path: sha256_file(root / path) for path in CODE_PATHS}
    manifest = {
        "protocol_version": "step8_static_v1",
        "status": "frozen",
        "primary_population": summary["primary_population"],
        "artifact_hashes": artifact_hashes,
        "code_hashes": code_hashes,
        "config_sha256": sha256_file(args.config),
    }
    atomic_write_json(manifest, paths["detection_freeze_manifest_json"])
    lines = ["Step 8 final SHA-256 hashes", ""]
    lines.extend(f"{value}  {key}" for key, value in artifact_hashes.items())
    lines.extend(["", "Implementation code hashes", ""])
    lines.extend(f"{value}  {key}" for key, value in code_hashes.items())
    Path(paths["detection_final_hashes_txt"]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["detection_final_hashes_txt"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Step 8 static detection experiment frozen.")
    print(f"Wrote: {paths['detection_freeze_manifest_json']}")
    print(f"Wrote: {paths['detection_final_hashes_txt']}")


if __name__ == "__main__":
    main()
