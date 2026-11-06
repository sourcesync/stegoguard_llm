from __future__ import annotations

import argparse
from pathlib import Path

from stegoguard_lm.detection.io_utils import read_json
from stegoguard_lm.robustness.io import hash_files, load_step12_config, verify_frozen_inputs, write_json


IMPLEMENTATION_FILES = [
    "src/stegoguard_lm/robustness/__init__.py",
    "src/stegoguard_lm/robustness/io.py",
    "src/stegoguard_lm/robustness/statistics.py",
    "src/stegoguard_lm/robustness/analysis.py",
    "src/stegoguard_lm/robustness/reporting.py",
    "scripts/verify_step12_inputs.py",
    "scripts/freeze_step12_analysis_spec.py",
    "scripts/verify_step12_analysis_spec.py",
    "scripts/smoke_test_step12.py",
    "scripts/run_step12_robustness_analysis.py",
    "scripts/validate_step12_results.py",
    "scripts/freeze_step12_analysis.py",
    "scripts/verify_step12_freeze.py",
]


def artifact_files(config: dict) -> list[str]:
    p = config["paths"]
    figures = [
        f"{p['step12_figures_dir']}/step12_r3_effect_estimates.png",
        f"{p['step12_figures_dir']}/step12_leave_one_cell_stability.png",
        f"{p['step12_figures_dir']}/step12_alternative_leakage_metrics.png",
        f"{p['step12_figures_dir']}/step12_clean_burden_cells.png",
        f"{p['step12_figures_dir']}/step12_static_ablation_reference.png",
    ]
    return [
        "config_step12.yaml",
        "STEP12_PROTOCOL.md",
        "STEP12_IMPLEMENTATION.md",
        p["step12_analysis_spec_manifest"],
        p["step12_analysis_spec_hashes"],
        p["step12_summary_json"],
        p["step12_randomization_tests_csv"],
        p["step12_r3_influence_summary_csv"],
        p["step12_loo_parent_csv"],
        p["step12_loo_cell_csv"],
        p["step12_alternative_leakage_csv"],
        p["step12_adaptive_window_csv"],
        p["step12_history_conditioned_csv"],
        p["step12_heterogeneity_csv"],
        p["step12_clean_cell_metrics_csv"],
        p["step12_static_ablation_reference_csv"],
        p["step12_cross_stage_reference_csv"],
        p["step12_claim_stability_json"],
        p["step12_interpretation_md"],
        p["step12_qa_review_txt"],
        *figures,
        p["step12_validation_summary_json"],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze validated Step 12 robustness artifacts.")
    parser.add_argument("--config", default="config_step12.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_step12_config(args.config)
    manifests = verify_frozen_inputs(config)
    validation = read_json(config["paths"]["step12_validation_summary_json"])
    if validation.get("status") != "validated":
        raise ValueError("Step 12 results must validate before final freeze.")
    output = Path(config["paths"]["step12_final_freeze_manifest"])
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Step 12 final freeze already exists: {output}")
    artifacts = hash_files(artifact_files(config))
    implementation = hash_files(IMPLEMENTATION_FILES)
    manifest = {
        "protocol_version": config["step12_robustness"]["protocol_version"],
        "status": "frozen",
        "input_protocol_versions": {k: v["protocol_version"] for k, v in manifests.items()},
        "artifact_hashes": artifacts,
        "implementation_hashes": implementation,
    }
    write_json(manifest, output)
    hashes = Path(config["paths"]["step12_final_hashes"])
    hashes.parent.mkdir(parents=True, exist_ok=True)
    hashes.write_text(
        "Step 12 final SHA-256 hashes\n\n"
        + "\n".join(f"{digest}  {path}" for path, digest in artifacts.items())
        + "\n\nImplementation code hashes\n\n"
        + "\n".join(f"{digest}  {path}" for path, digest in implementation.items())
        + "\n",
        encoding="utf-8",
    )
    print("STEP 12 FREEZE PASSED")
    print(f"Protocol: {manifest['protocol_version']}")
    print("Status: frozen")
    print(f"Frozen artifacts: {len(artifacts)}")
    print(f"Frozen implementation files: {len(implementation)}")


if __name__ == "__main__":
    main()
