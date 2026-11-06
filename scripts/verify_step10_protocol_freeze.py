from __future__ import annotations

import argparse
from pathlib import Path

from stegoguard_lm.adaptive.io import load_yaml
from stegoguard_lm.detection.io_utils import read_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen Step 10 protocol before generation.")
    parser.add_argument("--config", default="config_step10.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    manifest = read_json(config["paths"]["step10_protocol_manifest_json"])
    if manifest.get("status") != "frozen_protocol":
        raise SystemExit("Step 10 protocol freeze verification failed: status is not frozen_protocol.")
    for rel, expected in manifest["artifact_hashes"].items():
        path = Path(rel)
        observed = sha256_file(path)
        if observed != expected:
            raise SystemExit(f"Step 10 protocol freeze verification failed: {rel} hash mismatch.")
    print("STEP 10 PROTOCOL FREEZE VERIFICATION PASSED")
    print(f"Protocol: {manifest['protocol_version']}")
    print(f"Frozen protocol artifacts: {len(manifest['artifact_hashes'])}")


if __name__ == "__main__":
    main()
