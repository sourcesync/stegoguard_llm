from __future__ import annotations

import argparse
from pathlib import Path

from stegoguard_lm.analysis.io import hash_files, load_step11_config, verify_step10_frozen, write_json


SPEC_ARTIFACTS = [
    "config_step11.yaml",
    "STEP11_PROTOCOL.md",
    "STEP11_IMPLEMENTATION.md",
    "src/stegoguard_lm/analysis/__init__.py",
    "src/stegoguard_lm/analysis/io.py",
    "src/stegoguard_lm/analysis/statistics.py",
    "src/stegoguard_lm/analysis/system.py",
    "src/stegoguard_lm/analysis/reporting.py",
    "scripts/verify_step11_inputs.py",
    "scripts/freeze_step11_analysis_spec.py",
    "scripts/verify_step11_analysis_spec.py",
    "scripts/smoke_test_step11.py",
    "scripts/run_step11_system_analysis.py",
    "scripts/validate_step11_results.py",
    "scripts/freeze_step11_analysis.py",
    "scripts/verify_step11_freeze.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the Step 11 post-outcome analysis specification.")
    parser.add_argument("--config", default="config_step11.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_step11_config(args.config)
    step10_manifest = verify_step10_frozen(config)
    output = Path(config["paths"]["step11_analysis_spec_manifest"])
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Step 11 analysis-spec freeze already exists: {output}")
    hashes = hash_files(SPEC_ARTIFACTS)
    manifest = {
        "protocol_version": config["step11_system_analysis"]["protocol_version"],
        "status": "frozen_analysis_spec",
        "input_protocol_version": step10_manifest["protocol_version"],
        "artifact_hashes": hashes,
    }
    write_json(manifest, output)
    hash_path = Path(config["paths"]["step11_analysis_spec_hashes"])
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(
        "Step 11 analysis-spec SHA-256 hashes\n\n" + "\n".join(f"{digest}  {path}" for path, digest in hashes.items()) + "\n",
        encoding="utf-8",
    )
    print("STEP 11 ANALYSIS-SPEC FREEZE PASSED")
    print(f"Protocol: {manifest['protocol_version']}")
    print(f"Status: {manifest['status']}")
    print(f"Frozen spec/code artifacts: {len(hashes)}")


if __name__ == "__main__":
    main()
