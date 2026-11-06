from __future__ import annotations

import argparse

from stegoguard_lm.robustness.analysis import (
    compute_influence,
    compute_randomization_tests,
    split_roles,
)
from stegoguard_lm.robustness.io import load_step10_feedback_rows, load_step12_config, verify_frozen_inputs
from stegoguard_lm.robustness.statistics import benjamini_hochberg, paired_sign_flip_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a model-free Step 12 smoke test.")
    parser.add_argument("--config", default="config_step12.yaml")
    args = parser.parse_args()
    config = load_step12_config(args.config)
    verify_frozen_inputs(config)
    rows = load_step10_feedback_rows(config)
    stego, _ = split_roles(rows)
    # Low-cost algorithm checks.
    test = paired_sign_flip_test([1.0, 1.0, -1.0, 1.0], replicates=1000, seed=7)
    if not (0.0 <= float(test["p_value"]) <= 1.0):
        raise AssertionError("Invalid paired sign-flip p-value.")
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    if any(x is None or not (0.0 <= float(x) <= 1.0) for x in q):
        raise AssertionError("Invalid BH-FDR result.")
    # Use a reduced-randomization config copy to exercise the real contrast logic quickly.
    smoke = dict(config)
    smoke["step12_robustness"] = dict(config["step12_robustness"])
    smoke["step12_robustness"]["randomization_replicates"] = 200
    randomization = compute_randomization_tests(stego, smoke)
    influence, loo_parent, loo_cell = compute_influence(stego, smoke)
    if len(randomization) != 54 or len(influence) != 9 or len(loo_parent) != 432 or len(loo_cell) != 72:
        raise AssertionError("Unexpected Step 12 smoke-test row counts.")
    print("STEP 12 MODEL-FREE SMOKE TEST PASSED")
    print(f"Randomization rows: {len(randomization)}")
    print(f"R3 contrasts: {len(influence)}")
    print(f"Leave-one-parent rows: {len(loo_parent)}")
    print(f"Leave-one-cell rows: {len(loo_cell)}")


if __name__ == "__main__":
    main()
