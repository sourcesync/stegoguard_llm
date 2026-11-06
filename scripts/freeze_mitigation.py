from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from stegoguard_lm.detection.io_utils import atomic_write_json, read_json, sha256_file
from stegoguard_lm.mitigation.tasks import step9_implementation_hashes


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze completed Step 9 mitigation results.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(args.config).resolve().parent
    paths = config["paths"]
    summary = read_json(paths["mitigation_summary_json"])
    required = [
        paths["mitigation_task_manifest_jsonl"],
        paths["mitigation_manifest_json"],
        paths["mitigation_outputs_jsonl"],
        paths["mitigation_payload_results_jsonl"],
        paths["mitigation_utility_jsonl"],
        paths["mitigation_gated_results_jsonl"],
        paths["mitigation_summary_json"],
    ]
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze Step 9; missing artifacts: {missing}")
    artifact_hashes = {path: sha256_file(path) for path in required}
    manifest = {
        "protocol_version": config["mitigation_evaluation"]["protocol_version"],
        "status": "frozen",
        "primary_population": summary["primary_population"],
        "artifact_hashes": artifact_hashes,
        "implementation_hashes": step9_implementation_hashes(root),
        "step8_freeze_manifest_sha256": sha256_file(paths["detection_freeze_manifest_json"]),
    }
    atomic_write_json(manifest, paths["mitigation_freeze_manifest_json"])
    lines = ["Step 9 final SHA-256 hashes", ""]
    lines.extend(f"{value}  {key}" for key, value in artifact_hashes.items())
    Path(paths["mitigation_final_hashes_txt"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Step 9 mitigation experiment frozen.")
    print(f"Wrote: {paths['mitigation_freeze_manifest_json']}")
    print(f"Wrote: {paths['mitigation_final_hashes_txt']}")


if __name__ == "__main__":
    main()
