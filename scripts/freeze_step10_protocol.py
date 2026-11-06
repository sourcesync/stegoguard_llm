from __future__ import annotations

import argparse
from pathlib import Path

from stegoguard_lm.adaptive.io import load_yaml, verify_freeze_manifest
from stegoguard_lm.detection.io_utils import atomic_write_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Step 10 protocol before adaptive data generation.")
    parser.add_argument("--config", default="config_step10.yaml")
    parser.add_argument("--protocol", default="STEP10_PROTOCOL.md")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    paths = config["paths"]
    output = Path(paths["step10_protocol_manifest_json"])
    hashes_txt = Path(paths["step10_protocol_hashes_txt"])
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Step 10 protocol is already frozen: {output}")
    verify_freeze_manifest(".", "outputs/pilot_v0/manifests/step8_static_detection_manifest.json")
    verify_freeze_manifest(".", "outputs/pilot_v0/manifests/step9_mitigation_freeze_manifest.json")
    artifacts = [
        Path(args.config),
        Path(args.protocol),
        Path(paths["adaptive_candidate_manifest_jsonl"]),
        Path(paths["adaptive_trajectory_manifest_jsonl"]),
    ]
    hashes = {str(path).replace('\\','/'): sha256_file(path) for path in artifacts}
    manifest = {
        "protocol_version": config["adaptive_evaluation"]["protocol_version"],
        "status": "frozen_protocol",
        "round_ids": config["adaptive_evaluation"]["round_ids"],
        "expected_parent_count": config["adaptive_evaluation"]["expected_parent_count"],
        "attacker_conditions": config["adaptive_evaluation"]["attacker_conditions"],
        "defender_policies": config["adaptive_evaluation"]["defender_policies"],
        "artifact_hashes": hashes,
        "step8_freeze_manifest_sha256": sha256_file("outputs/pilot_v0/manifests/step8_static_detection_manifest.json"),
        "step9_freeze_manifest_sha256": sha256_file("outputs/pilot_v0/manifests/step9_mitigation_freeze_manifest.json"),
    }
    atomic_write_json(manifest, output)
    hashes_txt.parent.mkdir(parents=True, exist_ok=True)
    hashes_txt.write_text(
        "Step 10 protocol SHA-256 hashes\n\n" + "\n".join(f"{digest}  {path}" for path, digest in hashes.items()) + "\n",
        encoding="utf-8",
    )
    print("STEP 10 PROTOCOL FREEZE PASSED")
    print(f"Protocol: {manifest['protocol_version']}")
    print(f"Status: {manifest['status']}")
    print(f"Frozen protocol artifacts: {len(hashes)}")


if __name__ == "__main__":
    main()
