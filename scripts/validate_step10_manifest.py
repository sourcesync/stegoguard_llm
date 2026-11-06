from __future__ import annotations

import argparse

from stegoguard_lm.adaptive.io import load_yaml
from stegoguard_lm.adaptive.manifests import validate_manifests
from stegoguard_lm.detection.io_utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 10 candidate and trajectory manifests.")
    parser.add_argument("--config", default="config_step10.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    paths = config["paths"]
    candidates = read_jsonl(paths["adaptive_candidate_manifest_jsonl"])
    trajectories = read_jsonl(paths["adaptive_trajectory_manifest_jsonl"])
    errors = validate_manifests(config, candidates, trajectories)
    if errors:
        print("Step 10 manifest validation failed.")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Step 10 manifest validation passed.")
    print(f"Candidate specifications: {len(candidates)}")
    print(f"Trajectories: {len(trajectories)}")
    print(f"Parents: {len({r['clean_parent_id'] for r in trajectories})}")


if __name__ == "__main__":
    main()
