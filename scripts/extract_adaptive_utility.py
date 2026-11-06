from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

from stegoguard_lm.adaptive.io import load_yaml, round_paths, verify_step10_protocol
from stegoguard_lm.detection.evaluator_scoring import load_evaluator
from stegoguard_lm.detection.io_utils import append_jsonl_record, read_jsonl
from stegoguard_lm.mitigation.utility import evaluate_utility_after_mitigation


def baseline_utility(original_dist: dict[str, Any], original_sem: dict[str, Any]) -> dict[str, Any]:
    tokens = int(original_dist["response_token_count"])
    return {
        "semantic_similarity_original_to_mitigated": 1.0,
        "visible_instruction_retention_before": float(original_sem["visible_instruction_retention"]),
        "visible_instruction_retention_after": float(original_sem["visible_instruction_retention"]),
        "visible_instruction_retention_delta": 0.0,
        "mean_logp_before": float(original_dist["mean_logp"]),
        "mean_logp_after": float(original_dist["mean_logp"]),
        "mean_logp_delta": 0.0,
        "ppl_before": float(original_dist["ppl"]),
        "ppl_after": float(original_dist["ppl"]),
        "ppl_delta": 0.0,
        "response_token_count_before": tokens,
        "response_token_count_after": tokens,
        "absolute_token_change": 0,
        "length_ratio": 1.0,
        "logp_var_after": float(original_dist["logp_var"]),
        "sentence_logp_var_after": float(original_dist["sentence_logp_var"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Step 10 post-defense utility for one round.")
    parser.add_argument("--config", default="config_step10.yaml")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--role", choices=["all", "stego", "clean"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    verify_step10_protocol(config)
    rp = round_paths(config, args.round)
    outputs = read_jsonl(rp["outputs"])
    existing = read_jsonl(rp["utility"]) if rp["utility"].exists() else []
    done = {str(r["task_id"]) for r in existing}
    if done and not args.resume:
        raise FileExistsError("Adaptive utility results exist. Use --resume.")
    candidates = {str(r["candidate_id"]): r for r in read_jsonl(config["paths"]["adaptive_candidate_samples_jsonl"])}
    static_dataset = {str(r["sample_id"]): r for r in read_jsonl(config["paths"]["detection_dataset_jsonl"])}
    sem = {str(r["sample_id"]): r for r in read_jsonl(config["paths"]["adaptive_detection_semantic_features_jsonl"])}
    dist = {str(r["sample_id"]): r for r in read_jsonl(config["paths"]["adaptive_detection_evaluator_a_features_jsonl"])}

    selected = []
    for row in outputs:
        if str(row["task_id"]) in done:
            continue
        if args.role == "stego" and row["record_role"] != "stego_trajectory":
            continue
        if args.role == "clean" and row["record_role"] != "clean_control":
            continue
        selected.append(row)
    if args.limit is not None:
        selected = selected[: max(0, int(args.limit))]
    model_rows = [r for r in selected if r["intervention_route"] != "pass_through"]
    evaluator = None
    evaluator_role = str(config["mitigation_evaluation"]["utility"]["evaluator_role"])
    evaluator_id = next(str(m["id"]) for m in config["models"]["evaluators"] if m["role"] == evaluator_role)
    if model_rows:
        evaluator = load_evaluator(evaluator_id)
    embedding_cache: dict[str, Any] = {}
    try:
        for index, row in enumerate(selected, start=1):
            sample_id = str(row["candidate_id"])
            original = candidates[sample_id] if row["record_role"] == "stego_trajectory" else static_dataset[sample_id]
            if row["intervention_route"] == "pass_through":
                utility = baseline_utility(dist[sample_id], sem[sample_id])
            else:
                assert evaluator is not None
                utility, embedding = evaluate_utility_after_mitigation(
                    model=evaluator.model,
                    tokenizer=evaluator.tokenizer,
                    visible_prompt=str(original["prompt"]),
                    original_text=str(original["text"]),
                    mitigated_text=str(row["final_text"]),
                    original_semantic_row=sem[sample_id],
                    original_distribution_row=dist[sample_id],
                    original_embedding=embedding_cache.get(sample_id),
                )
                embedding_cache[sample_id] = embedding
            append_jsonl_record(
                {
                    "task_id": str(row["task_id"]),
                    "record_role": str(row["record_role"]),
                    "trajectory_id": str(row["trajectory_id"]),
                    "round_id": int(row["round_id"]),
                    "candidate_id": sample_id,
                    "attack_name": str(row["attack_name"]),
                    "source_model": str(row["source_model"]),
                    **utility,
                    "evaluator_role": evaluator_role,
                    "evaluator_id": evaluator_id,
                    "protocol_version": config["adaptive_evaluation"]["protocol_version"],
                },
                rp["utility"],
            )
            print(f"[{index}/{len(selected)}] Utility evaluated: {row['task_id']}")
    finally:
        if evaluator is not None:
            del evaluator
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    print(f"Step 10 round {args.round} utility extraction complete.")


if __name__ == "__main__":
    main()
