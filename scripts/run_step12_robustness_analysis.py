from __future__ import annotations

import argparse

from stegoguard_lm.detection.io_utils import read_json
from stegoguard_lm.robustness.analysis import build_step12_analysis
from stegoguard_lm.robustness.io import load_step10_feedback_rows, load_step12_config, verify_frozen_inputs
from stegoguard_lm.robustness.reporting import write_result_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Step 12 robustness and sensitivity analysis.")
    parser.add_argument("--config", default="config_step12.yaml")
    args = parser.parse_args()
    config = load_step12_config(args.config)
    verify_frozen_inputs(config)
    spec = read_json(config["paths"]["step12_analysis_spec_manifest"])
    if spec.get("status") != "frozen_analysis_spec":
        raise ValueError("Step 12 analysis specification must be frozen before analysis.")
    rows = load_step10_feedback_rows(config)
    step8_metrics = read_json(config["paths"]["step8_metrics"])
    step8_summary = read_json(config["paths"]["step8_summary"])
    step9_summary = read_json(config["paths"]["step9_summary"])
    step11_summary = read_json(config["paths"]["step11_summary"])
    summary, tables, claims = build_step12_analysis(
        rows, step8_metrics, step8_summary, step9_summary, step11_summary, config
    )
    figures = write_result_package(summary, tables, claims, config)
    print("STEP 12 ROBUSTNESS ANALYSIS COMPLETE")
    print(f"Parents: {summary['parents']}")
    print(f"Trajectories: {summary['trajectories']}")
    print(f"Randomization test rows: {summary['randomization_test_rows']}")
    print(f"R3 endpoint contrasts: {summary['r3_contrast_count']}")
    print(f"Leave-one-parent rows: {summary['loo_parent_rows']}")
    print(f"Leave-one-cell rows: {summary['loo_cell_rows']}")
    print(f"Figures: {len(figures)}")
    print(f"Wrote: {config['paths']['step12_summary_json']}")


if __name__ == "__main__":
    main()
