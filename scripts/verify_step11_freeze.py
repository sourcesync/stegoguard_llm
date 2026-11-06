from __future__ import annotations

import argparse
from pathlib import Path

from stegoguard_lm.analysis.io import load_step11_config, verify_hash_mapping
from stegoguard_lm.detection.io_utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the final frozen Step 11 analysis.")
    parser.add_argument("--config", default="config_step11.yaml")
    args = parser.parse_args()
    config = load_step11_config(args.config)
    path = Path(config["paths"]["step11_final_freeze_manifest"])
    if not path.exists():
        raise SystemExit(f"Step 11 freeze verification failed: manifest not found: {path}")
    manifest = read_json(path)
    if manifest.get("status") != "frozen":
        raise SystemExit("Step 11 freeze verification failed: status is not frozen.")
    if manifest.get("protocol_version") != config["step11_system_analysis"]["protocol_version"]:
        raise SystemExit("Step 11 freeze verification failed: protocol mismatch.")
    checks = dict(manifest.get("artifact_hashes", {}))
    checks.update(manifest.get("implementation_hashes", {}))
    failures = verify_hash_mapping(checks)
    if failures:
        print("STEP 11 FREEZE VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("STEP 11 FREEZE VERIFICATION PASSED")
    print(f"Protocol: {manifest['protocol_version']}")
    print(f"Status: {manifest['status']}")
    print(f"Frozen artifacts: {len(manifest.get('artifact_hashes', {}))}")
    print(f"Frozen implementation files: {len(manifest.get('implementation_hashes', {}))}")


if __name__ == "__main__":
    main()
