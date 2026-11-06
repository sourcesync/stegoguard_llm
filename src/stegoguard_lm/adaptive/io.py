from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from stegoguard_lm.detection.io_utils import read_json, sha256_file


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def verify_freeze_manifest(project_root: str | Path, manifest_path: str | Path) -> None:
    root = Path(project_root)
    manifest = read_json(root / manifest_path)
    if manifest.get("status") != "frozen":
        raise ValueError(f"Freeze manifest is not frozen: {manifest_path}")

    artifact_hashes = manifest.get("artifact_hashes", {})
    code_hashes = manifest.get("code_hashes", manifest.get("implementation_hashes", {}))
    for rel, expected in {**artifact_hashes, **code_hashes}.items():
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(f"Frozen artifact missing: {rel}")
        observed = sha256_file(path)
        if observed.casefold() != str(expected).casefold():
            raise ValueError(
                f"Frozen artifact hash mismatch for {rel}: observed={observed}, expected={expected}"
            )


def derive_int(*parts: object, modulo: int = 2**31 - 1) -> int:
    if modulo < 2:
        raise ValueError("modulo must be >= 2")
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % modulo


def round_dir(config: dict[str, Any], round_id: int) -> Path:
    return Path(config["paths"]["adaptive_output_dir"]) / f"round_{int(round_id)}"


def round_paths(config: dict[str, Any], round_id: int) -> dict[str, Path]:
    base = round_dir(config, round_id)
    return {
        "dir": base,
        "tasks": base / "round_tasks.jsonl",
        "outputs": base / "defense_outputs.jsonl",
        "failures": base / "defense_failures.jsonl",
        "payload": base / "payload_results.jsonl",
        "utility": base / "utility_results.jsonl",
        "feedback": base / "feedback_results.jsonl",
        "validation": base / "validation_summary.json",
    }


def verify_step10_protocol(config: dict[str, Any]) -> None:
    manifest = read_json(config["paths"]["step10_protocol_manifest_json"])
    if manifest.get("status") != "frozen_protocol":
        raise ValueError("Step 10 protocol is not frozen.")
    if manifest.get("protocol_version") != config["adaptive_evaluation"]["protocol_version"]:
        raise ValueError("Step 10 protocol version mismatch between config and freeze manifest.")
    for rel, expected in manifest.get("artifact_hashes", {}).items():
        observed = sha256_file(rel)
        if observed.casefold() != str(expected).casefold():
            raise ValueError(f"Step 10 frozen protocol artifact changed: {rel}")
