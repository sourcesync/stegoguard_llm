from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import torch

from stegoguard_lm.adaptive.io import load_yaml, round_paths, verify_step10_protocol
from stegoguard_lm.detection.evaluator_scoring import load_evaluator
from stegoguard_lm.detection.io_utils import append_jsonl_record, read_jsonl
from stegoguard_lm.mitigation.normalization import sanitize_unicode_text
from stegoguard_lm.mitigation.rewriting import rewrite_response


def collect_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen Step 10 defender to one adaptive round.")
    parser.add_argument("--config", default="config_step10.yaml")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--role", choices=["all", "stego", "clean"], default="all")
    parser.add_argument("--policy", choices=["all", "deterministic_rewrite", "randomized_rewrite"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    verify_step10_protocol(config)
    rp = round_paths(config, args.round)
    tasks = read_jsonl(rp["tasks"])
    outputs = read_jsonl(rp["outputs"]) if rp["outputs"].exists() else []
    failures = read_jsonl(rp["failures"]) if rp["failures"].exists() else []
    done = {str(r["task_id"]) for r in outputs} | {str(r["task_id"]) for r in failures}
    if done and not args.resume:
        raise FileExistsError("Round defense outputs already exist. Use --resume.")
    candidates = {str(r["candidate_id"]): r for r in read_jsonl(config["paths"]["adaptive_candidate_samples_jsonl"])}
    static_dataset = {str(r["sample_id"]): r for r in read_jsonl(config["paths"]["detection_dataset_jsonl"])}
    selected = []
    for task in tasks:
        if str(task["task_id"]) in done:
            continue
        if args.role == "stego" and task["record_role"] != "stego_trajectory":
            continue
        if args.role == "clean" and task["record_role"] != "clean_control":
            continue
        if args.policy != "all" and task["defender_policy"] != args.policy:
            continue
        selected.append(task)
    if args.limit is not None:
        selected = selected[: max(0, int(args.limit))]

    rewriter = None
    evaluator_role = str(config["mitigation_evaluation"]["rewriter_role"])
    evaluator_id = next(str(m["id"]) for m in config["models"]["evaluators"] if m["role"] == evaluator_role)
    rewrite_cfg = config["mitigation_evaluation"]["rewriter_generation"]
    unicode_cfg = config["mitigation_evaluation"]["unicode_sanitization"]
    reserved_cfg = config["attacks"]["out_of_band"]
    reserved = [reserved_cfg["bit_zero_codepoint"], reserved_cfg["bit_one_codepoint"], reserved_cfg["prefix_codepoint"], reserved_cfg["suffix_codepoint"]]
    try:
        for index, task in enumerate(selected, start=1):
            task_id = str(task["task_id"])
            original = candidates[str(task["candidate_id"])] if task["record_role"] == "stego_trajectory" else static_dataset[str(task["candidate_id"])]
            original_text = str(original["text"])
            try:
                route = str(task["intervention_route"])
                metadata: dict[str, Any]
                if route == "pass_through":
                    final_text = original_text
                    metadata = {"changed": False, "mode": "pass_through"}
                elif route == "unicode_sanitization":
                    result = sanitize_unicode_text(
                        original_text,
                        normalization_form=str(unicode_cfg["normalization_form"]),
                        reserved_codepoints=reserved,
                        remove_all_format_codepoints=bool(unicode_cfg["remove_all_format_codepoints"]),
                    )
                    final_text = result.text
                    metadata = {
                        "changed": result.changed,
                        "mode": "unicode_sanitization",
                        "removed_codepoint_count": result.removed_codepoint_count,
                        "removed_format_codepoint_count": result.removed_format_codepoint_count,
                    }
                elif route == "linguistic_rewrite":
                    if rewriter is None:
                        rewriter = load_evaluator(evaluator_id)
                    mode = str(task["defender_policy"])
                    seed = task["rewrite_seed"] if mode == "randomized_rewrite" else None
                    try:
                        rewrite = rewrite_response(
                            model=rewriter.model,
                            tokenizer=rewriter.tokenizer,
                            visible_prompt=str(original["prompt"]),
                            response=original_text,
                            mode=mode,
                            seed=seed,
                            max_length_ratio=float(rewrite_cfg["max_length_ratio"]),
                            max_extra_tokens=int(rewrite_cfg["max_extra_tokens"]),
                            hard_cap=int(rewrite_cfg["hard_cap_new_tokens"]),
                            randomized_temperature=float(rewrite_cfg["randomized_temperature"]),
                            randomized_top_p=float(rewrite_cfg["randomized_top_p"]),
                        )
                        final_text = rewrite.text
                        metadata = {
                            "changed": rewrite.changed,
                            "mode": rewrite.mode,
                            "seed": rewrite.seed,
                            "input_token_count_rewriter": rewrite.input_token_count,
                            "output_token_count_rewriter": rewrite.output_token_count,
                            "max_new_tokens": rewrite.max_new_tokens,
                            "do_sample": rewrite.do_sample,
                            "temperature": rewrite.temperature,
                            "top_p": rewrite.top_p,
                            "empty_output_fallback": False,
                        }
                    except RuntimeError as rewrite_exc:
                        if not str(rewrite_exc).startswith("rewrite_empty_output:"):
                            raise
                        final_text = original_text
                        metadata = {
                            "changed": False,
                            "mode": mode,
                            "seed": seed,
                            "empty_output_fallback": True,
                            "fallback_reason": str(rewrite_exc),
                        }
                else:
                    raise ValueError(f"Unsupported Step 10 intervention route: {route}")
                record = {
                    **task,
                    "original_text": original_text,
                    "final_text": final_text,
                    "text_changed": final_text != original_text,
                    "method_metadata": metadata,
                    "applied_rewrite_seed": metadata.get("seed"),
                    "generation_status": "accepted",
                }
                append_jsonl_record(record, rp["outputs"])
                print(f"[{index}/{len(selected)}] Defender output accepted: {task_id}")
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                infrastructure = ("out of memory", "cuda", "cublas", "cudnn", "cannot allocate memory", "connection", "timeout")
                if any(marker in message.casefold() for marker in infrastructure):
                    print(f"Infrastructure interruption: {task_id}: {message}")
                    raise
                failure = {"task_id": task_id, "round_id": args.round, "error_type": type(exc).__name__, "error_message": str(exc), "status": "terminal_defense_generation_failure", "protocol_version": config["adaptive_evaluation"]["protocol_version"]}
                append_jsonl_record(failure, rp["failures"])
                print(f"Terminal defense failure: {task_id}: {message}")
    finally:
        if rewriter is not None:
            del rewriter
        collect_memory()
    print(f"Step 10 round {args.round} defense generation complete.")


if __name__ == "__main__":
    main()
