from __future__ import annotations

import argparse
from pathlib import Path

from stegoguard_lm.robustness.io import load_step12_config, verify_hash_mapping
from stegoguard_lm.detection.io_utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen Step 12 robustness-analysis specification.")
    parser.add_argument("--config", default="config_step12.yaml")
    args = parser.parse_args()
    config = load_step12_config(args.config)
    path = Path(config["paths"]["step12_analysis_spec_manifest"])
    if not path.exists():
        raise SystemExit(f"Step 12 analysis-spec verification failed: manifest not found: {path}")
    manifest = read_json(path)
    if manifest.get("status") != "frozen_analysis_spec":
        raise SystemExit("Step 12 analysis-spec verification failed: status mismatch.")
    if manifest.get("protocol_version") != config["step12_robustness"]["protocol_version"]:
        raise SystemExit("Step 12 analysis-spec verification failed: protocol mismatch.")
    failures = verify_hash_mapping(manifest.get("artifact_hashes", {}))
    if failures:
        print("STEP 12 ANALYSIS-SPEC VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("STEP 12 ANALYSIS-SPEC VERIFICATION PASSED")
    print(f"Protocol: {manifest['protocol_version']}")
    print(f"Status: {manifest['status']}")
    print(f"Frozen spec/code artifacts: {len(manifest.get('artifact_hashes', {}))}")


if __name__ == "__main__":
    main()
