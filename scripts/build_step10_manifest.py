from __future__ import annotations

import argparse
from pathlib import Path

from stegoguard_lm.adaptive.io import load_yaml
from stegoguard_lm.adaptive.manifests import build_manifests
from stegoguard_lm.detection.io_utils import atomic_write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen Step 10 candidate and trajectory manifests.")
    parser.add_argument("--config", default="config_step10.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    paths = config["paths"]
    candidate_path = Path(paths["adaptive_candidate_manifest_jsonl"])
    trajectory_path = Path(paths["adaptive_trajectory_manifest_jsonl"])
    if (candidate_path.exists() or trajectory_path.exists()) and not args.overwrite:
        raise FileExistsError("Step 10 manifests already exist. Use --overwrite only before protocol freeze.")
    candidates, trajectories = build_manifests(config)
    atomic_write_jsonl(candidates, candidate_path)
    atomic_write_jsonl(trajectories, trajectory_path)
    print("Step 10 manifests built.")
    print(f"Candidate specifications: {len(candidates)}")
    print(f"Trajectories: {len(trajectories)}")
    print(f"Candidate manifest: {candidate_path}")
    print(f"Trajectory manifest: {trajectory_path}")


if __name__ == "__main__":
    main()
