from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from stegoguard_lm.adaptive.io import load_yaml, verify_freeze_manifest
from stegoguard_lm.detection.io_utils import atomic_write_json, read_json, read_jsonl, sha256_file


def load_step12_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    if "step12_robustness" not in config or "paths" not in config:
        raise ValueError("Invalid Step 12 config: missing step12_robustness or paths.")
    return config


def verify_frozen_inputs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Verify every frozen stage that Step 12 reads directly."""
    paths = config["paths"]
    manifests = {
        "step8": Path(paths["step8_freeze_manifest"]),
        "step9": Path(paths["step9_freeze_manifest"]),
        "step10": Path(paths["step10_freeze_manifest"]),
        "step11": Path(paths["step11_freeze_manifest"]),
    }
    result: dict[str, dict[str, Any]] = {}
    expected_protocols = config["step12_robustness"]["input_protocol_versions"]
    for name, path in manifests.items():
        if not path.exists():
            raise FileNotFoundError(f"Required frozen input manifest not found: {path}")
        verify_freeze_manifest(".", path)
        manifest = read_json(path)
        expected = expected_protocols[name]
        if manifest.get("protocol_version") != expected:
            raise ValueError(
                f"{name} protocol mismatch: observed={manifest.get('protocol_version')} expected={expected}"
            )
        if manifest.get("status") != "frozen":
            raise ValueError(f"{name} is not frozen.")
        result[name] = manifest

    step11_validation = read_json(paths["step11_validation_summary"])
    if step11_validation.get("status") != "validated":
        raise ValueError("Step 11 validation summary is not validated.")
    return result


def load_step10_feedback_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    template = str(config["paths"]["step10_round_dir_template"])
    for round_id in config["step12_robustness"]["round_ids"]:
        path = Path(template.format(round_id=int(round_id))) / "feedback_results.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen Step 10 feedback file: {path}")
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
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


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
