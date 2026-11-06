from __future__ import annotations

import argparse
import gc
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from stegoguard_lm.adaptive.io import load_yaml, round_paths, verify_step10_protocol
from stegoguard_lm.adaptive.strategy import transformed_codebook
from stegoguard_lm.attacks.synonym_channel import load_synonym_codebook
from stegoguard_lm.detection.io_utils import append_jsonl_record, read_jsonl
from stegoguard_lm.mitigation.payload_eval import (
    evaluate_rank_payload_after_mitigation,
    evaluate_synonym_payload_after_mitigation,
)


def collect_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def baseline(candidate: dict[str, Any]) -> dict[str, Any]:
    bits = int(candidate["payload_bit_length"])
    return {
        "decoder_success": True,
        "decoder_error": None,
        "decoded_payload": str(candidate["payload"]),
        "exact_payload_recovery": True,
        "expected_bit_count": bits,
        "observed_bit_count": bits,
        "compared_bit_count": bits,
        "correct_bit_count": bits,
        "effective_bit_accuracy": 1.0,
        "effective_bit_error_rate": 0.0,
        "conditional_observed_bit_accuracy": 1.0,
        "residual_capacity_bits": bits,
        "residual_capacity_fraction": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate post-defense payload recovery for one Step 10 round.")
    parser.add_argument("--config", default="config_step10.yaml")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--channel", choices=["all", "rank", "synonym"], default="all")
    parser.add_argument("--source-model", choices=["all", "qwen", "mistral"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from stegoguard_lm.clean_generation import load_model_and_tokenizer
    config = load_yaml(args.config)
    verify_step10_protocol(config)
    rp = round_paths(config, args.round)
    outputs = [r for r in read_jsonl(rp["outputs"]) if r["record_role"] == "stego_trajectory"]
    candidates = {str(r["candidate_id"]): r for r in read_jsonl(config["paths"]["adaptive_candidate_samples_jsonl"])}
    existing = read_jsonl(rp["payload"]) if rp["payload"].exists() else []
    done = {str(r["task_id"]) for r in existing}
    if done and not args.resume:
        raise FileExistsError("Adaptive payload results already exist. Use --resume.")

    selected = []
    for row in outputs:
        if str(row["task_id"]) in done:
            continue
        if args.channel == "rank" and row["attack_name"] != "rank_encoding":
            continue
        if args.channel == "synonym" and row["attack_name"] != "synonym_slot_codebook":
            continue
        model = str(row["source_model"]).casefold()
        if args.source_model == "qwen" and "qwen" not in model:
            continue
        if args.source_model == "mistral" and "mistral" not in model:
            continue
        selected.append(row)
    if args.limit is not None:
        selected = selected[: max(0, int(args.limit))]

    codebook = load_synonym_codebook(config["paths"]["synonym_codebook_yaml"])
    rank_pending: dict[str, list[dict[str, Any]]] = defaultdict(list)
    processed = 0
    for row in selected:
        candidate = candidates[str(row["candidate_id"])]
        if row["intervention_route"] == "pass_through":
            metrics = baseline(candidate)
        elif row["attack_name"] == "synonym_slot_codebook":
            mapped = transformed_codebook(
                codebook,
                active_slot_ids=list(candidate["channel_parameters"]["active_slot_ids"]),
                mapping_mask=int(candidate.get("synonym_mapping_mask") or 0),
            )
            metrics = evaluate_synonym_payload_after_mitigation(
                original_attack=candidate,
                mitigated_text=str(row["final_text"]),
                codebook=mapped,
            )
        elif row["attack_name"] == "rank_encoding":
            rank_pending[str(row["source_model"])].append(row)
            continue
        else:
            raise ValueError(f"Unexpected adaptive attack: {row['attack_name']}")
        append_jsonl_record(
            {
                "task_id": str(row["task_id"]),
                "trajectory_id": str(row["trajectory_id"]),
                "round_id": int(row["round_id"]),
                "candidate_id": str(row["candidate_id"]),
                "attack_name": str(row["attack_name"]),
                "strategy_index": int(row["strategy_index"]),
                "source_model": str(row["source_model"]),
                **metrics,
                "protocol_version": config["adaptive_evaluation"]["protocol_version"],
            },
            rp["payload"],
        )
        processed += 1
        print(f"Payload evaluated: {row['task_id']}")

    rank_cfg = config["attacks"]["token_whitebox"]
    for model_id, rows in rank_pending.items():
        if all(r["intervention_route"] == "pass_through" for r in rows):
            bundle = None
        else:
            print(f"Loading source model for adaptive rank decoding: {model_id}")
            bundle = load_model_and_tokenizer(model_id)
        try:
            for row in rows:
                candidate = candidates[str(row["candidate_id"])]
                if row["intervention_route"] == "pass_through":
                    metrics = baseline(candidate)
                else:
                    assert bundle is not None
                    model, tokenizer, _, _ = bundle
                    metrics = evaluate_rank_payload_after_mitigation(
                        original_attack=candidate,
                        mitigated_text=str(row["final_text"]),
                        model=model,
                        tokenizer=tokenizer,
                        attack_config=rank_cfg,
                    )
                append_jsonl_record(
                    {
                        "task_id": str(row["task_id"]),
                        "trajectory_id": str(row["trajectory_id"]),
                        "round_id": int(row["round_id"]),
                        "candidate_id": str(row["candidate_id"]),
                        "attack_name": str(row["attack_name"]),
                        "strategy_index": int(row["strategy_index"]),
                        "source_model": str(row["source_model"]),
                        **metrics,
                        "protocol_version": config["adaptive_evaluation"]["protocol_version"],
                    },
                    rp["payload"],
                )
                processed += 1
                print(f"Payload evaluated: {row['task_id']}")
        finally:
            if bundle is not None:
                del bundle
            collect_memory()
    print(f"Step 10 round {args.round} payload evaluation complete. New rows: {processed}")


if __name__ == "__main__":
    main()
