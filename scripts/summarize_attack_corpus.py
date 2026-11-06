from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from stegoguard_lm.attacks.full_generation import (
    atomic_write_json,
    build_corpus_summary,
    read_jsonl,
    write_full_preview,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the Step 7 corpus summary and stratified preview."
    )
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping.")
    return value


def distribution_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]

    tasks = read_jsonl(paths["attack_task_manifest_jsonl"])
    accepted = read_jsonl(paths["attack_full_samples_jsonl"], allow_missing=True)
    failures = read_jsonl(paths["attack_full_failures_jsonl"], allow_missing=True)

    summary = build_corpus_summary(
        tasks=tasks,
        accepted_records=accepted,
        terminal_failures=failures,
    )

    payload_bits_by_attack: defaultdict[str, list[float]] = defaultdict(list)
    payload_rates_by_attack: defaultdict[str, list[float]] = defaultdict(list)
    accepted_parent_attacks: defaultdict[str, set[str]] = defaultdict(set)
    parent_cell: dict[str, tuple[str, str, int]] = {}

    for record in accepted:
        attack_name = str(record["attack_name"])
        payload_bits_by_attack[attack_name].append(float(record["payload_bit_length"]))
        payload_rates_by_attack[attack_name].append(float(record["payload_rate_bpt"]))
        parent_id = str(record["parent_sample_id"])
        accepted_parent_attacks[parent_id].add(attack_name)
        parent_cell[parent_id] = (
            str(record["source_model"]),
            str(record["domain"]),
            int(record["nominal_target_length"]),
        )

    all_attack_names = {
        str(task["attack_name"])
        for task in tasks
        if bool(task["eligible"])
    }
    common_parent_ids = {
        parent_id
        for parent_id, attacks in accepted_parent_attacks.items()
        if all_attack_names.issubset(attacks)
    }
    common_by_cell = Counter(parent_cell[parent_id] for parent_id in common_parent_ids)
    balanced_per_cell = min(common_by_cell.values()) if len(common_by_cell) == 8 else 0

    summary.update(
        {
            "payload_bit_length_by_attack": {
                key: distribution_summary(values)
                for key, values in sorted(payload_bits_by_attack.items())
            },
            "payload_rate_bpt_by_attack": {
                key: distribution_summary(values)
                for key, values in sorted(payload_rates_by_attack.items())
            },
            "accepted_common_parent_count_all_three_attacks": len(common_parent_ids),
            "accepted_common_parent_counts_by_model_domain_length": {
                " | ".join(map(str, key)): value
                for key, value in sorted(common_by_cell.items())
            },
            "balanced_common_parents_per_cell_available": balanced_per_cell,
            "balanced_common_parent_total_available": balanced_per_cell * 8,
            "total_static_inventory_clean_plus_accepted_stego": (
                int(config["full_attack_generation"]["expected_clean_parents"])
                + len(accepted)
            ),
            "summary_path": str(paths["attack_full_summary_json"]),
            "preview_path": str(paths["attack_full_preview_txt"]),
        }
    )

    atomic_write_json(summary, paths["attack_full_summary_json"])
    write_full_preview(accepted, paths["attack_full_preview_txt"])

    print("Step 7 corpus summary rebuilt.")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
