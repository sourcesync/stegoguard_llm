from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from stegoguard_lm.detection.io_utils import read_json, read_jsonl, sha256_file
from stegoguard_lm.mitigation.tasks import (
    mitigation_protocol_sha256,
    step9_implementation_hashes,
    validate_mitigation_tasks,
    verify_step8_freeze,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen Step 9 mitigation manifest.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(args.config).resolve().parent
    paths = config["paths"]
    verify_step8_freeze(config, project_root=root)
    tasks = read_jsonl(paths["mitigation_task_manifest_jsonl"])
    dataset = read_jsonl(paths["detection_dataset_jsonl"])
    validation = validate_mitigation_tasks(tasks=tasks, config=config, detection_records=dataset)
    manifest = read_json(paths["mitigation_manifest_json"])
    errors: list[str] = []
    checks = {
        "protocol_version": config["mitigation_evaluation"]["protocol_version"],
        "task_count": validation["task_count"],
        "primary_task_count": validation["primary_task_count"],
        "mitigation_protocol_sha256": mitigation_protocol_sha256(config),
        "task_manifest_sha256": sha256_file(paths["mitigation_task_manifest_jsonl"]),
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            errors.append(f"Manifest {key} mismatch: observed={manifest.get(key)!r}, expected={expected!r}")
    observed_hashes = step9_implementation_hashes(root)
    if manifest.get("implementation_hashes") != observed_hashes:
        errors.append("Step 9 implementation hashes no longer match the frozen manifest.")
    for path_key, manifest_key in (
        ("detection_dataset_jsonl", "detection_dataset_sha256"),
        ("attack_full_samples_jsonl", "step7_stego_samples_sha256"),
        ("clean_samples_full_jsonl", "clean_samples_sha256"),
    ):
        observed = sha256_file(paths[path_key])
        if observed != manifest.get(manifest_key):
            errors.append(f"Frozen source hash mismatch for {paths[path_key]}")
    if errors:
        print("Step 9 mitigation manifest validation failed.")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Step 9 mitigation manifest validation passed.")
    print(f"Detection records: {validation['detection_records']}")
    print(f"Mitigation tasks: {validation['task_count']}")
    print(f"Primary matched tasks: {validation['primary_task_count']}")


if __name__ == "__main__":
    main()
