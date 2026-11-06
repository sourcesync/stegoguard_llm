from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from stegoguard_lm.adaptive.io import load_yaml, verify_freeze_manifest
from stegoguard_lm.detection.io_utils import atomic_write_json, read_json, read_jsonl, sha256_file


def load_step11_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    if "step11_system_analysis" not in config or "paths" not in config:
        raise ValueError("Invalid Step 11 config: missing step11_system_analysis or paths.")
    return config


def round_feedback_path(config: dict[str, Any], round_id: int) -> Path:
    template = str(config["paths"]["step10_round_dir_template"])
    return Path(template.format(round_id=int(round_id))) / "feedback_results.jsonl"


def verify_step10_frozen(config: dict[str, Any]) -> dict[str, Any]:
    step10_config = load_yaml(config["paths"]["step10_config"])
    manifest_path = Path(config["paths"]["step10_freeze_manifest"])
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Step 10 final freeze manifest not found: {manifest_path}. "
            "Complete and verify the Step 10 final freeze before Step 11."
        )
    verify_freeze_manifest(".", manifest_path)
    manifest = read_json(manifest_path)
    expected = config["step11_system_analysis"]["input_protocol_version"]
    if manifest.get("protocol_version") != expected:
        raise ValueError(
            f"Step 10 protocol mismatch: observed={manifest.get('protocol_version')} expected={expected}"
        )
    if manifest.get("status") != "frozen":
        raise ValueError("Step 10 final freeze status is not frozen.")
    if step10_config.get("adaptive_evaluation", {}).get("protocol_version") != expected:
        raise ValueError("config_step10.yaml no longer matches the frozen Step 10 protocol.")
    return manifest


def load_step10_feedback_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for round_id in config["step11_system_analysis"]["round_ids"]:
        path = round_feedback_path(config, int(round_id))
        if not path.exists():
            raise FileNotFoundError(f"Missing Step 10 feedback file: {path}")
        rows.extend(read_jsonl(path))
    return rows


def write_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    rows = list(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_json(value: Any, path: str | Path) -> None:
    atomic_write_json(value, path)


def hash_files(paths: Iterable[str | Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Cannot hash missing file: {path}")
        result[str(path).replace("\\", "/")] = sha256_file(path)
    return result


def verify_hash_mapping(mapping: dict[str, str]) -> list[str]:
    failures: list[str] = []
    for rel, expected in mapping.items():
        path = Path(rel)
        if not path.exists():
            failures.append(f"missing: {rel}")
            continue
        observed = sha256_file(path)
        if observed.casefold() != str(expected).casefold():
            failures.append(f"hash mismatch: {rel}")
    return failures
