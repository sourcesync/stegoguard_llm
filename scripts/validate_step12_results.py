from __future__ import annotations

import argparse
import csv
from pathlib import Path

from stegoguard_lm.detection.io_utils import read_json
from stegoguard_lm.robustness.io import load_step12_config, verify_frozen_inputs, write_json


def csv_count(path: str) -> int:
    target = Path(path)
    if not target.exists():
        return -1
    with target.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Step 12 robustness result package.")
    parser.add_argument("--config", default="config_step12.yaml")
    args = parser.parse_args()
    config = load_step12_config(args.config)
    verify_frozen_inputs(config)
    p = config["paths"]
    summary_path = Path(p["step12_summary_json"])
    if not summary_path.exists():
        raise SystemExit(f"Step 12 result validation failed: missing {summary_path}")
    summary = read_json(summary_path)
    cfg = config["step12_robustness"]
    failures: list[str] = []
    expected_summary = {
        "parents": int(cfg["expected_parent_count"]),
        "trajectories": int(cfg["expected_trajectory_count"]),
        "stego_rows": int(cfg["expected_stego_rows"]),
        "clean_rows": int(cfg["expected_clean_rows"]),
        "randomization_test_rows": 54,
        "r3_contrast_count": 9,
        "loo_parent_rows": 432,
        "loo_cell_rows": 72,
        "alternative_leakage_rows": 18,
        "adaptive_window_rows": 9,
        "history_conditioned_rows": 18,
        "heterogeneity_rows": 54,
        "clean_cell_rows": 64,
        "static_ablation_rows": 8,
        "cross_stage_reference_rows": 15,
    }
    for key, expected in expected_summary.items():
        if int(summary.get(key, -1)) != expected:
            failures.append(f"{key}: observed={summary.get(key)} expected={expected}")
    csv_expect = {
        "step12_randomization_tests_csv": 54,
        "step12_r3_influence_summary_csv": 9,
        "step12_loo_parent_csv": 432,
        "step12_loo_cell_csv": 72,
        "step12_alternative_leakage_csv": 18,
        "step12_adaptive_window_csv": 9,
        "step12_history_conditioned_csv": 18,
        "step12_heterogeneity_csv": 54,
        "step12_clean_cell_metrics_csv": 64,
        "step12_static_ablation_reference_csv": 8,
        "step12_cross_stage_reference_csv": 15,
    }
    for key, expected in csv_expect.items():
        observed = csv_count(p[key])
        if observed != expected:
            failures.append(f"{key}: observed={observed} expected={expected}")
    for key in ["step12_claim_stability_json", "step12_interpretation_md", "step12_qa_review_txt"]:
        if not Path(p[key]).exists():
            failures.append(f"missing: {p[key]}")
    figure_dir = Path(p["step12_figures_dir"])
    figures = sorted(figure_dir.glob("step12_*.png")) if figure_dir.exists() else []
    if len(figures) != 5:
        failures.append(f"figure_count_on_disk: observed={len(figures)} expected=5")
    result = {
        "protocol_version": cfg["protocol_version"],
        "status": "validated" if not failures else "failed",
        "parents": summary.get("parents"),
        "trajectories": summary.get("trajectories"),
        "randomization_test_rows": summary.get("randomization_test_rows"),
        "r3_contrast_count": summary.get("r3_contrast_count"),
        "figure_count": len(figures),
        "failures": failures,
    }
    write_json(result, p["step12_validation_summary_json"])
    if failures:
        print("STEP 12 RESULT VALIDATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("STEP 12 RESULT VALIDATION PASSED")
    print(result)


if __name__ == "__main__":
    main()
