from __future__ import annotations

import argparse

from stegoguard_lm.analysis.io import load_step11_config, load_step10_feedback_rows, verify_step10_frozen
from stegoguard_lm.analysis.system import compute_round_metrics, compute_trajectory_metrics, split_roles
from stegoguard_lm.detection.io_utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a model-free Step 11 system-analysis smoke test.")
    parser.add_argument("--config", default="config_step11.yaml")
    args = parser.parse_args()
    config = load_step11_config(args.config)
    verify_step10_frozen(config)
    spec = read_json(config["paths"]["step11_analysis_spec_manifest"])
    if spec.get("status") != "frozen_analysis_spec":
        raise ValueError("Freeze and verify the Step 11 analysis specification before smoke testing.")
    rows = load_step10_feedback_rows(config)
    stego, _ = split_roles(rows)
    round_metrics = compute_round_metrics(stego, config)
    trajectory_metrics = compute_trajectory_metrics(stego, config)
    parents = len({str(r["clean_parent_id"]) for r in stego})
    if len(round_metrics) != 24:
        raise ValueError(f"Expected 24 round-condition rows, got {len(round_metrics)}")
    if len(trajectory_metrics) != 288:
        raise ValueError(f"Expected 288 trajectory rows, got {len(trajectory_metrics)}")
    print("STEP 11 MODEL-FREE SMOKE TEST PASSED")
    print(f"Parents: {parents}")
    print(f"Round-condition rows: {len(round_metrics)}")
    print(f"Trajectory rows: {len(trajectory_metrics)}")


if __name__ == "__main__":
    main()
