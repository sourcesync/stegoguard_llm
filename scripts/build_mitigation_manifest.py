from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from stegoguard_lm.detection.io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
)
from stegoguard_lm.mitigation.tasks import (
    build_mitigation_tasks,
    mitigation_protocol_sha256,
    mitigation_protocol_subset,
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
    parser = argparse.ArgumentParser(description="Build and freeze the Step 9 mitigation task universe.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(args.config).resolve().parent
    paths = config["paths"]
    task_path = Path(paths["mitigation_task_manifest_jsonl"])
    manifest_path = Path(paths["mitigation_manifest_json"])
    output_path = Path(paths["mitigation_outputs_jsonl"])
    failure_path = Path(paths["mitigation_failures_jsonl"])
    if (output_path.exists() or failure_path.exists()) and args.overwrite:
        raise FileExistsError(
            "Refusing to rebuild the Step 9 task universe after mitigation outputs/failures exist. "
            "Reset/version Step 9 intentionally instead."
        )
    if task_path.exists() and not args.overwrite:
        raise FileExistsError(f"Task manifest already exists: {task_path}. Use --overwrite before generation begins.")

    step8_manifest = verify_step8_freeze(config, project_root=root)
    dataset = read_jsonl(paths["detection_dataset_jsonl"])
    tasks = build_mitigation_tasks(config=config, detection_records=dataset)
    validation = validate_mitigation_tasks(tasks=tasks, config=config, detection_records=dataset)
    atomic_write_jsonl(tasks, task_path)

    manifest = {
        "protocol_version": config["mitigation_evaluation"]["protocol_version"],
        "status": "frozen_task_universe",
        **validation,
        "mitigation_protocol": mitigation_protocol_subset(config),
        "mitigation_protocol_sha256": mitigation_protocol_sha256(config),
        "frozen_step8_manifest_sha256": sha256_file(paths["detection_freeze_manifest_json"]),
        "frozen_step8_config_sha256": step8_manifest["config_sha256"],
        "detection_dataset_sha256": sha256_file(paths["detection_dataset_jsonl"]),
        "step7_stego_samples_sha256": sha256_file(paths["attack_full_samples_jsonl"]),
        "clean_samples_sha256": sha256_file(paths["clean_samples_full_jsonl"]),
        "task_manifest_sha256": sha256_file(task_path),
        "implementation_hashes": step9_implementation_hashes(root),
    }
    atomic_write_json(manifest, manifest_path)
    print("Step 9 mitigation task manifest built and frozen.")
    print(f"Detection records: {validation['detection_records']}")
    print(f"Mitigation tasks: {validation['task_count']}")
    print(f"Primary matched tasks: {validation['primary_task_count']}")
    print("Counts by condition:")
    for key, value in validation["counts_by_condition"].items():
        print(f"  {key}: {value}")
    print(f"Wrote: {task_path}")
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
