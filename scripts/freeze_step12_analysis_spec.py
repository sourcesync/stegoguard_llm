from __future__ import annotations

import argparse
from pathlib import Path

from stegoguard_lm.robustness.io import hash_files, load_step12_config, verify_frozen_inputs, write_json


SPEC_ARTIFACTS = [
    "config_step12.yaml",
    "STEP12_PROTOCOL.md",
    "STEP12_IMPLEMENTATION.md",
    "src/stegoguard_lm/robustness/__init__.py",
    "src/stegoguard_lm/robustness/io.py",
    "src/stegoguard_lm/robustness/statistics.py",
    "src/stegoguard_lm/robustness/analysis.py",
    "src/stegoguard_lm/robustness/reporting.py",
    "scripts/verify_step12_inputs.py",
    "scripts/freeze_step12_analysis_spec.py",
    "scripts/verify_step12_analysis_spec.py",
    "scripts/smoke_test_step12.py",
    "scripts/run_step12_robustness_analysis.py",
    "scripts/validate_step12_results.py",
    "scripts/freeze_step12_analysis.py",
    "scripts/verify_step12_freeze.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the Step 12 robustness-analysis specification.")
    parser.add_argument("--config", default="config_step12.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_step12_config(args.config)
    manifests = verify_frozen_inputs(config)
    output = Path(config["paths"]["step12_analysis_spec_manifest"])
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Step 12 analysis-spec freeze already exists: {output}")
    hashes = hash_files(SPEC_ARTIFACTS)
    manifest = {
        "protocol_version": config["step12_robustness"]["protocol_version"],
        "status": "frozen_analysis_spec",
        "input_protocol_versions": {k: v["protocol_version"] for k, v in manifests.items()},
        "artifact_hashes": hashes,
    }
    write_json(manifest, output)
    hash_path = Path(config["paths"]["step12_analysis_spec_hashes"])
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(
        "Step 12 robustness analysis-spec SHA-256 hashes\n\n"
        + "\n".join(f"{digest}  {path}" for path, digest in hashes.items()) + "\n",
        encoding="utf-8",
    )
    print("STEP 12 ANALYSIS-SPEC FREEZE PASSED")
    print(f"Protocol: {manifest['protocol_version']}")
    print(f"Status: {manifest['status']}")
    print(f"Frozen spec/code artifacts: {len(hashes)}")


if __name__ == "__main__":
    main()
