from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import torch

from stegoguard_lm.adaptive.candidates import copy_static_candidate, generate_candidate
from stegoguard_lm.adaptive.io import load_yaml, verify_step10_protocol
from stegoguard_lm.attacks.synonym_channel import load_synonym_codebook
from stegoguard_lm.detection.io_utils import append_jsonl_record, read_jsonl


def collect_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the frozen Step 10 adaptive candidate catalog.")
    parser.add_argument("--config", default="config_step10.yaml")
    parser.add_argument("--channel", choices=["all", "rank", "synonym"], default="all")
    parser.add_argument("--source-model", choices=["all", "qwen", "mistral"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from stegoguard_lm.clean_generation import load_model_and_tokenizer

    config = load_yaml(args.config)
    verify_step10_protocol(config)
    paths = config["paths"]
    specs = read_jsonl(paths["adaptive_candidate_manifest_jsonl"])
    clean = {str(r["sample_id"]): r for r in read_jsonl(paths["clean_samples_full_jsonl"])}
    static_attacks = {str(r["sample_id"]): r for r in read_jsonl(paths["attack_full_samples_jsonl"])}
    output_path = Path(paths["adaptive_candidate_samples_jsonl"])
    failure_path = Path(paths["adaptive_candidate_failures_jsonl"])
    accepted = read_jsonl(output_path) if output_path.exists() else []
    failures = read_jsonl(failure_path) if failure_path.exists() else []
    accepted_ids = {str(r["candidate_id"]) for r in accepted}
    failure_ids = {str(r["candidate_id"]) for r in failures}
    if (accepted_ids or failure_ids) and not args.resume:
        raise FileExistsError("Adaptive candidate outputs already exist. Use --resume.")

    def matches(spec: dict[str, Any]) -> bool:
        if str(spec["candidate_id"]) in accepted_ids or str(spec["candidate_id"]) in failure_ids:
            return False
        if args.channel == "rank" and spec["attack_name"] != "rank_encoding":
            return False
        if args.channel == "synonym" and spec["attack_name"] != "synonym_slot_codebook":
            return False
        model = str(spec["source_model"]).casefold()
        if args.source_model == "qwen" and "qwen" not in model:
            return False
        if args.source_model == "mistral" and "mistral" not in model:
            return False
        return True

    selected = [spec for spec in specs if matches(spec)]
    if args.limit is not None:
        selected = selected[: max(0, int(args.limit))]
    if not selected:
        print("No adaptive candidate tasks are pending for the requested filters.")
        return

    codebook = load_synonym_codebook(paths["synonym_codebook_yaml"])
    model_bundle: tuple[Any, Any, Any, Any] | None = None
    current_model_id: str | None = None
    tokenizer_cache: dict[str, Any] = {}

    try:
        for index, spec in enumerate(selected, start=1):
            candidate_id = str(spec["candidate_id"])
            parent = clean[str(spec["clean_parent_id"])]
            try:
                if int(spec["strategy_index"]) == 0:
                    static = static_attacks[str(spec["static_step7_sample_id"])]
                    record = copy_static_candidate(
                        spec=spec,
                        static_attack=static,
                        protocol_version=str(config["adaptive_evaluation"]["protocol_version"]),
                    )
                else:
                    source_model = str(spec["source_model"])
                    if spec["attack_name"] == "rank_encoding":
                        if current_model_id != source_model:
                            if model_bundle is not None:
                                del model_bundle
                                collect_memory()
                            model_bundle = load_model_and_tokenizer(source_model)
                            current_model_id = source_model
                        model, tokenizer, _, _ = model_bundle
                    else:
                        tokenizer = tokenizer_cache.get(source_model)
                        if tokenizer is None:
                            from transformers import AutoTokenizer
                            print(f"Loading tokenizer only: {source_model}")
                            tokenizer = AutoTokenizer.from_pretrained(source_model, use_fast=True)
                            if tokenizer.pad_token_id is None:
                                tokenizer.pad_token = tokenizer.eos_token
                            tokenizer_cache[source_model] = tokenizer
                        model = None
                    record = generate_candidate(
                        config=config,
                        spec=spec,
                        parent=parent,
                        tokenizer=tokenizer,
                        model=model,
                        codebook=codebook,
                    )
                append_jsonl_record(record, output_path)
                accepted_ids.add(candidate_id)
                print(f"[{index}/{len(selected)}] Accepted: {candidate_id}")
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                infrastructure_markers = (
                    "out of memory", "cuda", "cublas", "cudnn", "meta device",
                    "cannot allocate memory", "connection", "timeout",
                )
                if any(marker in message.casefold() for marker in infrastructure_markers):
                    print(f"Infrastructure interruption: {candidate_id}: {message}")
                    raise
                failure = {
                    "candidate_id": candidate_id,
                    "clean_parent_id": str(spec["clean_parent_id"]),
                    "attack_name": str(spec["attack_name"]),
                    "strategy_index": int(spec["strategy_index"]),
                    "source_model": str(spec["source_model"]),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "status": "terminal_candidate_generation_failure",
                    "protocol_version": config["adaptive_evaluation"]["protocol_version"],
                }
                append_jsonl_record(failure, failure_path)
                failure_ids.add(candidate_id)
                print(f"Terminal candidate failure: {candidate_id}: {message}")
    finally:
        if model_bundle is not None:
            del model_bundle
        collect_memory()

    print("Step 10 adaptive candidate generation run complete.")
    print(f"Accepted total: {len(read_jsonl(output_path)) if output_path.exists() else 0}")
    print(f"Terminal failures total: {len(read_jsonl(failure_path)) if failure_path.exists() else 0}")


if __name__ == "__main__":
    main()
