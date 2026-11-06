from __future__ import annotations

import argparse

from stegoguard_lm.analysis.io import load_step11_config, load_step10_feedback_rows, verify_step10_frozen
from stegoguard_lm.analysis.system import split_roles


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen Step 10 inputs for Step 11 system analysis.")
    parser.add_argument("--config", default="config_step11.yaml")
    args = parser.parse_args()
    config = load_step11_config(args.config)
    manifest = verify_step10_frozen(config)
    rows = load_step10_feedback_rows(config)
    stego, clean = split_roles(rows)
    cfg = config["step11_system_analysis"]
    parents = len({str(r["clean_parent_id"]) for r in stego})
    trajectories = len({str(r["trajectory_id"]) for r in stego})
    expected = {
        "parents": int(cfg["expected_parent_count"]),
        "trajectories": int(cfg["expected_trajectory_count"]),
        "stego": int(cfg["expected_stego_rows"]),
        "clean": int(cfg["expected_clean_rows"]),
    }
    observed = {"parents": parents, "trajectories": trajectories, "stego": len(stego), "clean": len(clean)}
    failures = [f"{key}: observed={observed[key]} expected={expected[key]}" for key in expected if observed[key] != expected[key]]
    if failures:
        print("STEP 11 INPUT VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("STEP 11 INPUT VERIFICATION PASSED")
    print(f"Step 10 status: {manifest['status']}")
    print(f"Step 10 protocol: {manifest['protocol_version']}")
    print(f"Parents: {parents}")
    print(f"Trajectories: {trajectories}")
    print(f"Stego rows: {len(stego)}")
    print(f"Clean rows: {len(clean)}")


if __name__ == "__main__":
    main()
