from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from stegoguard_lm.detection.io_utils import read_jsonl
from stegoguard_lm.mitigation.tasks import account_mitigation_tasks


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate persisted Step 9 mitigation outputs.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    tasks = read_jsonl(paths["mitigation_task_manifest_jsonl"])
    dataset = read_jsonl(paths["detection_dataset_jsonl"])
    outputs = read_jsonl(paths["mitigation_outputs_jsonl"]) if Path(paths["mitigation_outputs_jsonl"]).exists() else []
    failures = read_jsonl(paths["mitigation_failures_jsonl"]) if Path(paths["mitigation_failures_jsonl"]).exists() else []
    task_by_id = {str(row["task_id"]): row for row in tasks}
    data_by_id = {str(row["sample_id"]): row for row in dataset}
    errors: list[str] = []
    seen: set[str] = set()
    for row in outputs:
        task_id = str(row.get("task_id", ""))
        if task_id in seen:
            errors.append(f"Duplicate mitigation output task_id: {task_id}")
        seen.add(task_id)
        task = task_by_id.get(task_id)
        if task is None:
            errors.append(f"Unknown task_id in mitigation outputs: {task_id}")
            continue
        original = data_by_id.get(str(task["original_sample_id"]))
        if original is None:
            errors.append(f"Missing original detection row for {task_id}")
            continue
        for field in ("condition", "original_sample_id", "clean_parent_id", "attack_name", "source_model"):
            if row.get(field) != task.get(field):
                errors.append(f"{task_id}: {field} mismatch")
        if row.get("original_text") != original.get("text"):
            errors.append(f"{task_id}: persisted original_text differs from frozen detection dataset")
        if not isinstance(row.get("mitigated_text"), str) or not row["mitigated_text"].strip():
            errors.append(f"{task_id}: empty mitigated_text")
        if row.get("generation_status") != "accepted":
            errors.append(f"{task_id}: generation_status is not accepted")
    accounting = account_mitigation_tasks(tasks=tasks, accepted_outputs=outputs, terminal_failures=failures)
    if accounting["pending_tasks"] and not args.allow_pending:
        errors.append(f"{accounting['pending_tasks']} mitigation tasks remain pending")
    if errors:
        print("Step 9 mitigation output validation failed.")
        for error in errors[:100]:
            print(f"- {error}")
        raise SystemExit(1)
    print("Step 9 mitigation output validation passed.")
    for key in ("task_count", "accepted_tasks", "terminal_failures", "pending_tasks", "status"):
        print(f"{key}: {accounting[key]}")


if __name__ == "__main__":
    main()
