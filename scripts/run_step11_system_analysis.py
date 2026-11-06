from __future__ import annotations

import argparse

from stegoguard_lm.analysis.io import load_step11_config, load_step10_feedback_rows, verify_step10_frozen
from stegoguard_lm.analysis.reporting import write_result_package
from stegoguard_lm.analysis.system import build_system_analysis
from stegoguard_lm.detection.io_utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Step 11 system-level adaptive analysis.")
    parser.add_argument("--config", default="config_step11.yaml")
    args = parser.parse_args()
    config = load_step11_config(args.config)
    verify_step10_frozen(config)
    spec = read_json(config["paths"]["step11_analysis_spec_manifest"])
    if spec.get("status") != "frozen_analysis_spec":
        raise ValueError("Step 11 analysis specification must be frozen before analysis.")
    rows = load_step10_feedback_rows(config)
    summary, tables, hypotheses = build_system_analysis(rows, config)
    figures = write_result_package(summary, tables, hypotheses, config)
    print("STEP 11 SYSTEM-LEVEL ANALYSIS COMPLETE")
    print(f"Parents: {summary['parents']}")
    print(f"Trajectories: {summary['trajectories']}")
    print(f"Primary pairwise comparison rows: {summary['pairwise_comparison_rows']}")
    print(f"R3 stratified rows: {summary['r3_stratified_rows']}")
    print(f"Figures: {len(figures)}")
    print(f"Wrote: {config['paths']['step11_system_analysis_json']}")


if __name__ == "__main__":
    main()
