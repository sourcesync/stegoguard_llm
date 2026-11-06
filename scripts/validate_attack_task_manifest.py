from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from stegoguard_lm.attacks.full_generation import (
    read_json,
    read_jsonl,
    validate_attack_task_manifest,
    validate_synonym_capacity_audit,
    verify_frozen_clean_corpus,
    verify_generation_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the frozen Step 7 task universe before model generation."
    )
    parser.add_argument("--config", default="config.yaml")
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

    verify_frozen_clean_corpus(
        clean_path=clean_path,
        step5_manifest_path=step5_manifest_path,
        expected_record_count=int(full["expected_clean_parents"]),
    )
    clean_records = read_jsonl(clean_path)
    capacity_audit = read_json(capacity_path)
    validate_synonym_capacity_audit(
        capacity_audit=capacity_audit,
        clean_records=clean_records,
        expected_record_count=int(full["expected_clean_parents"]),
    )

    tasks = read_jsonl(task_path)
    structural = read_jsonl(structural_path)
    structural_ids = {task["task_id"] for task in structural}
    task_structural_ids = {task["task_id"] for task in tasks if not bool(task["eligible"])}
    if structural_ids != task_structural_ids:
        raise ValueError(
            "Structural exclusion JSONL does not exactly match ineligible task rows."
        )

    stats = validate_attack_task_manifest(
        tasks=tasks,
        config=config,
        expected_structural_count=len(structural),
    )
    generation_manifest = read_json(generation_manifest_path)
    verify_generation_manifest(
        config=config,
        generation_manifest=generation_manifest,
        task_manifest_path=task_path,
        structural_path=structural_path,
        clean_path=clean_path,
        capacity_audit_path=capacity_path,
        codebook_path=codebook_path,
        project_root=project_root,
    )

    expected_structural = int(capacity_audit["records_without_capacity"])
    expected_eligible = int(full["expected_attack_opportunities"]) - expected_structural
    if stats["structural_ineligible_tasks"] != expected_structural:
        raise ValueError(
            f"Expected {expected_structural} structural tasks from capacity audit, "
            f"found {stats['structural_ineligible_tasks']}."
        )
    if stats["eligible_tasks"] != expected_eligible:
        raise ValueError(
            f"Expected {expected_eligible} eligible tasks, found {stats['eligible_tasks']}."
        )

    print("Step 7 attack task manifest validation passed.")
    print(f"Clean parents: {stats['clean_parents']}")
    print(f"Attack opportunities: {stats['attack_opportunities']}")
    print(f"Eligible tasks: {stats['eligible_tasks']}")
    print(f"Structural zero-capacity tasks: {stats['structural_ineligible_tasks']}")
    print("Counts by family:")
    for family, count in sorted(stats["counts_by_family"].items()):
        print(f"  {family}: {count}")


if __name__ == "__main__":
    main()
