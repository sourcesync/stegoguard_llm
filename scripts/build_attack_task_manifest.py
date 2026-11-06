from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from stegoguard_lm.attacks.full_generation import (
    atomic_write_json,
    atomic_write_jsonl,
    build_attack_task_manifest,
    build_generation_manifest,
    read_json,
    read_jsonl,
    verify_frozen_clean_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and freeze the Step 7 full attack-opportunity manifest."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Step 7 manifest artifacts after re-validating inputs.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping.")
    return value


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    project_root = Path(args.config).resolve().parent
    paths = config["paths"]
    full = config["full_attack_generation"]

    clean_path = Path(paths["clean_samples_full_jsonl"])
    step5_manifest_path = Path(paths["step5_clean_corpus_manifest_json"])
    capacity_path = Path(paths["synonym_capacity_json"])
    codebook_path = Path(paths["synonym_codebook_yaml"])
    task_path = Path(paths["attack_task_manifest_jsonl"])
    structural_path = Path(paths["attack_structural_ineligible_jsonl"])
    generation_manifest_path = Path(paths["attack_generation_manifest_json"])

    # Rebuilding the frozen task universe after full generation has started would
    # invalidate persisted task IDs and status accounting. Refuse even with
    # --overwrite if generated Step 7 outputs already contain records.
    generated_paths = [
        Path(paths["attack_full_samples_jsonl"]),
        Path(paths["attack_full_failures_jsonl"]),
    ]
    started = [
        path for path in generated_paths if path.exists() and path.stat().st_size > 0
    ]
    if started:
        raise RuntimeError(
            "Step 7 generation has already started. The frozen task manifest cannot "
            "be rebuilt without first creating a new experiment/version. Existing "
            "generated artifacts: " + ", ".join(str(path) for path in started)
        )

    outputs = (task_path, structural_path, generation_manifest_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Step 7 manifest artifacts already exist. Use --overwrite only if you "
            "intend to rebuild the frozen task universe. Existing: "
            + ", ".join(str(path) for path in existing)
        )

    expected_clean = int(full["expected_clean_parents"])
    clean_hash = verify_frozen_clean_corpus(
        clean_path=clean_path,
        step5_manifest_path=step5_manifest_path,
        expected_record_count=expected_clean,
    )
    clean_records = read_jsonl(clean_path)
    capacity_audit = read_json(capacity_path)

    tasks, structural = build_attack_task_manifest(
        config=config,
        clean_records=clean_records,
        capacity_audit=capacity_audit,
    )

    # The manifest files are written atomically. The generation manifest is written
    # last because it hashes the two JSONL artifacts and becomes their frozen guard.
    atomic_write_jsonl(tasks, task_path)
    atomic_write_jsonl(structural, structural_path)

    eligible = sum(bool(task["eligible"]) for task in tasks)
    task_stats = {
        "clean_parents": expected_clean,
        "attack_opportunities": len(tasks),
        "eligible_tasks": eligible,
        "structural_ineligible_tasks": len(structural),
        "counts_by_family": {
            family: sum(task["attack_family"] == family for task in tasks)
            for family in ("out_of_band", "token_blackbox", "token_whitebox")
        },
    }
    generation_manifest = build_generation_manifest(
        config=config,
        task_manifest_path=task_path,
        structural_path=structural_path,
        clean_path=clean_path,
        capacity_audit_path=capacity_path,
        codebook_path=codebook_path,
        task_stats=task_stats,
        project_root=project_root,
    )
    atomic_write_json(generation_manifest, generation_manifest_path)

    print("Step 7 attack task manifest built and frozen.")
    print(f"Frozen clean corpus SHA256: {clean_hash}")
    print(f"Attack opportunities: {len(tasks)}")
    print(f"Eligible tasks: {eligible}")
    print(f"Structural zero-capacity tasks: {len(structural)}")
    print(f"Task manifest: {task_path}")
    print(f"Structural exclusions: {structural_path}")
    print(f"Generation manifest: {generation_manifest_path}")
    print(json.dumps(generation_manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
