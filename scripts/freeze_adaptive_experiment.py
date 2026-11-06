from __future__ import annotations

import argparse
from pathlib import Path

from stegoguard_lm.adaptive.io import load_yaml, round_paths, verify_freeze_manifest, verify_step10_protocol
from stegoguard_lm.detection.io_utils import atomic_write_json, read_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze completed Step 10 adaptive experiment artifacts and implementation.")
    parser.add_argument("--config", default="config_step10.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    verify_step10_protocol(config)
    verify_freeze_manifest(".", "outputs/pilot_v0/manifests/step8_static_detection_manifest.json")
    verify_freeze_manifest(".", "outputs/pilot_v0/manifests/step9_mitigation_freeze_manifest.json")
    cfg = config["adaptive_evaluation"]
    summary = read_json(config["paths"]["adaptive_summary_json"])
    if summary.get("status") != "complete":
        raise ValueError("Step 10 summary is not complete.")
    for round_id in cfg["round_ids"]:
        validation = read_json(round_paths(config, int(round_id))["validation"])
        if validation.get("status") != "validated":
            raise ValueError(f"Round {round_id} is not validated.")

    artifacts = [
        "config_step10.yaml",
        "STEP10_PROTOCOL.md",
        config["paths"]["adaptive_candidate_manifest_jsonl"],
        config["paths"]["adaptive_trajectory_manifest_jsonl"],
        config["paths"]["adaptive_candidate_samples_jsonl"],
        config["paths"]["adaptive_candidate_failures_jsonl"],
        config["paths"]["adaptive_detection_dataset_jsonl"],
        config["paths"]["adaptive_detection_evaluator_a_features_jsonl"],
        config["paths"]["adaptive_detection_evaluator_b_features_jsonl"],
        config["paths"]["adaptive_detection_semantic_features_jsonl"],
        config["paths"]["adaptive_detection_unicode_features_jsonl"],
        config["paths"]["adaptive_detection_feature_manifest_json"],
        config["paths"]["adaptive_detector_scores_jsonl"],
        config["paths"]["adaptive_summary_json"],
        config["paths"]["adaptive_qa_review_txt"],
        config["paths"]["step10_protocol_manifest_json"],
    ]
    for round_id in cfg["round_ids"]:
        rp = round_paths(config, int(round_id))
        artifacts.extend(str(rp[key]) for key in ("tasks", "outputs", "payload", "utility", "feedback", "validation"))
        if rp["failures"].exists():
            artifacts.append(str(rp["failures"]))
    implementation = sorted(
        [str(p).replace('\\','/') for p in Path("src/stegoguard_lm/adaptive").glob("*.py")]
        + [
            "scripts/build_step10_manifest.py",
            "scripts/validate_step10_manifest.py",
            "scripts/freeze_step10_protocol.py",
            "scripts/verify_step10_protocol_freeze.py",
            "scripts/generate_adaptive_candidates.py",
            "scripts/validate_adaptive_candidates.py",
            "scripts/build_adaptive_detection_dataset.py",
            "scripts/extract_adaptive_evaluator_features.py",
            "scripts/extract_adaptive_semantic_features.py",
            "scripts/extract_adaptive_unicode_features.py",
            "scripts/validate_adaptive_detection_features.py",
            "scripts/score_adaptive_candidates.py",
            "scripts/build_adaptive_round.py",
            "scripts/generate_adaptive_defense_outputs.py",
            "scripts/evaluate_adaptive_payloads.py",
            "scripts/extract_adaptive_utility.py",
            "scripts/finalize_adaptive_feedback.py",
            "scripts/validate_adaptive_round.py",
            "scripts/summarize_adaptive_results.py",
            "scripts/freeze_adaptive_experiment.py",
            "scripts/verify_step10_freeze.py",
            "scripts/smoke_test_adaptive.py",
        ]
    )
    artifact_hashes = {str(path).replace('\\','/'): sha256_file(path) for path in artifacts if Path(path).exists()}
    implementation_hashes = {path: sha256_file(path) for path in implementation}
    manifest = {
        "protocol_version": cfg["protocol_version"],
        "status": "frozen",
        "parent_count": cfg["expected_parent_count"],
        "round_ids": cfg["round_ids"],
        "artifact_hashes": artifact_hashes,
        "implementation_hashes": implementation_hashes,
        "step9_freeze_manifest_sha256": sha256_file("outputs/pilot_v0/manifests/step9_mitigation_freeze_manifest.json"),
    }
    output = Path(config["paths"]["step10_final_freeze_manifest_json"])
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Step 10 final freeze already exists: {output}")
    atomic_write_json(manifest, output)
    hashes = Path(config["paths"]["step10_final_hashes_txt"])
    hashes.parent.mkdir(parents=True, exist_ok=True)
    hashes.write_text(
        "Step 10 final SHA-256 hashes\n\n"
        + "\n".join(f"{digest}  {path}" for path, digest in artifact_hashes.items())
        + "\n\nImplementation code hashes\n\n"
        + "\n".join(f"{digest}  {path}" for path, digest in implementation_hashes.items())
        + "\n",
        encoding="utf-8",
    )
    print("STEP 10 FREEZE PASSED")
    print(f"Protocol: {cfg['protocol_version']}")
    print("Status: frozen")
    print(f"Frozen artifacts: {len(artifact_hashes)}")


if __name__ == "__main__":
    main()
