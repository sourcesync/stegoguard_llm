from __future__ import annotations

import argparse
from pathlib import Path

from stegoguard_lm.adaptive.io import load_yaml
from stegoguard_lm.detection.io_utils import read_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the final frozen Step 10 adaptive experiment.")
    parser.add_argument("--config", default="config_step10.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    path = Path(config["paths"]["step10_final_freeze_manifest_json"])
    if not path.exists():
        raise SystemExit(f"Step 10 freeze verification failed: manifest not found: {path}")
    manifest = read_json(path)
    if manifest.get("status") != "frozen":
        raise SystemExit("Step 10 freeze verification failed: status is not frozen.")
    expected_protocol = config["adaptive_evaluation"]["protocol_version"]
    if manifest.get("protocol_version") != expected_protocol:
        raise SystemExit("Step 10 freeze verification failed: protocol version mismatch.")

    checks = {}
    checks.update(manifest.get("artifact_hashes", {}))
    checks.update(manifest.get("implementation_hashes", {}))
    failures = []
    for rel, expected in checks.items():
        file_path = Path(rel)
        if not file_path.exists():
            failures.append(f"missing: {rel}")
            continue
        observed = sha256_file(file_path)
        if observed.casefold() != str(expected).casefold():
            failures.append(f"hash mismatch: {rel}")
    if failures:
        print("STEP 10 FREEZE VERIFICATION FAILED")
        for failure in failures[:100]:
            print(f"- {failure}")
        raise SystemExit(1)

    print("STEP 10 FREEZE VERIFICATION PASSED")
    print(f"Protocol: {manifest['protocol_version']}")
    print(f"Status: {manifest['status']}")
    print(f"Frozen artifacts: {len(manifest.get('artifact_hashes', {}))}")
    print(f"Frozen implementation files: {len(manifest.get('implementation_hashes', {}))}")


if __name__ == "__main__":
    main()
