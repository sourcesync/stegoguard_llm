from __future__ import annotations

import argparse
from pathlib import Path

from stegoguard_lm.analysis.io import hash_files, load_step11_config, verify_step10_frozen, write_json
from stegoguard_lm.detection.io_utils import read_json


IMPLEMENTATION_FILES = [
    "src/stegoguard_lm/analysis/__init__.py",
    "src/stegoguard_lm/analysis/io.py",
    "src/stegoguard_lm/analysis/statistics.py",
    "src/stegoguard_lm/analysis/system.py",
    "src/stegoguard_lm/analysis/reporting.py",
    "scripts/verify_step11_inputs.py",
    "scripts/freeze_step11_analysis_spec.py",
    "scripts/verify_step11_analysis_spec.py",
    "scripts/smoke_test_step11.py",
    "scripts/run_step11_system_analysis.py",
    "scripts/validate_step11_results.py",
    "scripts/freeze_step11_analysis.py",
    "scripts/verify_step11_freeze.py",
]


def artifact_files(config: dict) -> list[str]:
    p = config["paths"]
    figures = [
        f"{p['step11_figures_dir']}/step11_throughput_by_condition.png",
        f"{p['step11_figures_dir']}/step11_exact_survival_by_condition.png",
        f"{p['step11_figures_dir']}/step11_migration_synonym_share.png",
        f"{p['step11_figures_dir']}/step11_rank_residual_capacity_r3.png",
        f"{p['step11_figures_dir']}/step11_clean_semantic_utility.png",
    ]
    return [
        "config_step11.yaml",
        "STEP11_PROTOCOL.md",
        "STEP11_IMPLEMENTATION.md",
        p["step11_analysis_spec_manifest"],
        p["step11_analysis_spec_hashes"],
        p["step11_system_analysis_json"],
        p["step11_round_metrics_csv"],
        p["step11_pairwise_comparisons_csv"],
        p["step11_trajectory_metrics_csv"],
        p["step11_transition_metrics_csv"],
        p["step11_attack_family_metrics_csv"],
        p["step11_rank_residual_capacity_csv"],
        p["step11_recovery_metrics_csv"],
        p["step11_resilience_profiles_csv"],
        p["step11_history_conditioned_metrics_csv"],
        p["step11_stratified_metrics_csv"],
        p["step11_clean_metrics_csv"],
        p["step11_hypothesis_assessment_json"],
        p["step11_system_interpretation_md"],
        p["step11_qa_review_txt"],
        *figures,
        p["step11_validation_summary_json"],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze validated Step 11 system-level analysis artifacts.")
    parser.add_argument("--config", default="config_step11.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_step11_config(args.config)
    step10_manifest = verify_step10_frozen(config)
    validation = read_json(config["paths"]["step11_validation_summary_json"])
    if validation.get("status") != "validated":
        raise ValueError("Step 11 results must validate before final freeze.")
    output = Path(config["paths"]["step11_final_freeze_manifest"])
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Step 11 final freeze already exists: {output}")
    artifacts = hash_files(artifact_files(config))
    implementation = hash_files(IMPLEMENTATION_FILES)
    manifest = {
        "protocol_version": config["step11_system_analysis"]["protocol_version"],
        "status": "frozen",
        "input_step10_protocol_version": step10_manifest["protocol_version"],
        "artifact_hashes": artifacts,
        "implementation_hashes": implementation,
    }
    write_json(manifest, output)
    hashes = Path(config["paths"]["step11_final_hashes"])
    hashes.parent.mkdir(parents=True, exist_ok=True)
    hashes.write_text(
        "Step 11 final SHA-256 hashes\n\n"
        + "\n".join(f"{digest}  {path}" for path, digest in artifacts.items())
        + "\n\nImplementation code hashes\n\n"
        + "\n".join(f"{digest}  {path}" for path, digest in implementation.items())
        + "\n",
        encoding="utf-8",
    )
    print("STEP 11 FREEZE PASSED")
    print(f"Protocol: {manifest['protocol_version']}")
    print("Status: frozen")
    print(f"Frozen artifacts: {len(artifacts)}")
    print(f"Frozen implementation files: {len(implementation)}")


if __name__ == "__main__":
    main()
