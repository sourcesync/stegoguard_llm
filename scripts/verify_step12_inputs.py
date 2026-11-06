from __future__ import annotations

import argparse

from stegoguard_lm.robustness.io import load_step10_feedback_rows, load_step12_config, verify_frozen_inputs
from stegoguard_lm.robustness.analysis import split_roles


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen inputs for Step 12 robustness analysis.")
    parser.add_argument("--config", default="config_step12.yaml")
    args = parser.parse_args()
    config = load_step12_config(args.config)
    manifests = verify_frozen_inputs(config)
    rows = load_step10_feedback_rows(config)
    stego, clean = split_roles(rows)
    parents = len({str(r["clean_parent_id"]) for r in stego})
    trajectories = len({str(r["trajectory_id"]) for r in stego})
    cfg = config["step12_robustness"]
    checks = {
        "parents": (parents, int(cfg["expected_parent_count"])),
        "trajectories": (trajectories, int(cfg["expected_trajectory_count"])),
        "stego_rows": (len(stego), int(cfg["expected_stego_rows"])),
        "clean_rows": (len(clean), int(cfg["expected_clean_rows"])),
    }
    failures = [f"{k}: observed={o} expected={e}" for k, (o, e) in checks.items() if o != e]
    if failures:
        print("STEP 12 INPUT VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("STEP 12 INPUT VERIFICATION PASSED")
    for name in ["step8", "step9", "step10", "step11"]:
        print(f"{name.upper()} protocol: {manifests[name]['protocol_version']} (frozen)")
    print(f"Parents: {parents}")
    print(f"Trajectories: {trajectories}")
    print(f"Stego rows: {len(stego)}")
    print(f"Clean rows: {len(clean)}")


if __name__ == "__main__":
    main()
