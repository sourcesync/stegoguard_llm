from __future__ import annotations

import argparse
import csv
from pathlib import Path

from stegoguard_lm.analysis.io import load_step11_config, verify_step10_frozen, write_json
from stegoguard_lm.detection.io_utils import read_json


def csv_count(path: str) -> int:
    target = Path(path)
    if not target.exists():
        return -1
    with target.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Step 11 system-level analysis result package.")
    parser.add_argument("--config", default="config_step11.yaml")
    args = parser.parse_args()
    config = load_step11_config(args.config)
    verify_step10_frozen(config)
    summary_path = Path(config["paths"]["step11_system_analysis_json"])
    if not summary_path.exists():
        raise SystemExit(f"Step 11 result validation failed: missing {summary_path}")
    summary = read_json(summary_path)
    cfg = config["step11_system_analysis"]
    failures: list[str] = []
    expected = {
        "parents": int(cfg["expected_parent_count"]),
        "trajectories": int(cfg["expected_trajectory_count"]),
        "stego_rows": int(cfg["expected_stego_rows"]),
        "clean_rows": int(cfg["expected_clean_rows"]),
        "pairwise_comparison_rows": 27,
        "r3_stratified_rows": 48,
        "figure_count": 5,
    }
    for key, value in expected.items():
        if int(summary.get(key, -1)) != value:
            failures.append(f"{key}: observed={summary.get(key)} expected={value}")
    csv_expect = {
        "step11_round_metrics_csv": 24,
        "step11_pairwise_comparisons_csv": 27,
        "step11_trajectory_metrics_csv": 288,
        "step11_transition_metrics_csv": 18,
        "step11_attack_family_metrics_csv": 48,
        "step11_rank_residual_capacity_csv": 24,
        "step11_recovery_metrics_csv": 6,
        "step11_resilience_profiles_csv": 6,
        "step11_history_conditioned_metrics_csv": 16,
        "step11_stratified_metrics_csv": 48,
        "step11_clean_metrics_csv": 8,
    }
    for key, expected_count in csv_expect.items():
        observed = csv_count(config["paths"][key])
        if observed != expected_count:
            failures.append(f"{key}: observed={observed} expected={expected_count}")
    for key in ["step11_hypothesis_assessment_json", "step11_system_interpretation_md", "step11_qa_review_txt"]:
        if not Path(config["paths"][key]).exists():
            failures.append(f"missing: {config['paths'][key]}")
    figure_dir = Path(config["paths"]["step11_figures_dir"])
    figures = sorted(figure_dir.glob("step11_*.png")) if figure_dir.exists() else []
    if len(figures) != 5:
        failures.append(f"figure_count_on_disk: observed={len(figures)} expected=5")
    result = {
        "protocol_version": cfg["protocol_version"],
        "status": "validated" if not failures else "failed",
        "parents": summary.get("parents"),
        "trajectories": summary.get("trajectories"),
        "stego_rows": summary.get("stego_rows"),
        "clean_rows": summary.get("clean_rows"),
        "pairwise_comparison_rows": summary.get("pairwise_comparison_rows"),
        "figure_count": len(figures),
        "failures": failures,
    }
    write_json(result, config["paths"]["step11_validation_summary_json"])
    if failures:
        print("STEP 11 RESULT VALIDATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("STEP 11 RESULT VALIDATION PASSED")
    print(result)


if __name__ == "__main__":
    main()
