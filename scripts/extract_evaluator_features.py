from __future__ import annotations

import argparse
import gc
from pathlib import Path

import yaml

from stegoguard_lm.detection.evaluator_scoring import load_evaluator, score_response_distribution
from stegoguard_lm.detection.io_utils import append_jsonl_record, atomic_write_json, read_jsonl, sha256_file


def _output_for_role(paths: dict, role: str) -> tuple[Path, Path]:
    if role == "evaluator_a":
        return Path(paths["detection_evaluator_a_features_jsonl"]), Path(paths["detection_evaluator_a_summary_json"])
    if role == "evaluator_b":
        return Path(paths["detection_evaluator_b_features_jsonl"]), Path(paths["detection_evaluator_b_summary_json"])
    raise ValueError(f"Unsupported evaluator role: {role}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract response-only evaluator distributional features.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--evaluator", choices=["evaluator_a", "evaluator_b"], required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    paths = config["paths"]
    output, summary_path = _output_for_role(paths, args.evaluator)
    dataset = read_jsonl(paths["detection_dataset_jsonl"])
    model_cfg = next(
        item for item in config["models"]["evaluators"] if item["role"] == args.evaluator
    )
    model_id = str(model_cfg["id"])

    if args.overwrite:
        output.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    elif output.exists() and not args.resume:
        raise FileExistsError(f"Feature output exists: {output}. Use --resume or --overwrite.")

    existing = read_jsonl(output) if output.exists() else []
    existing_ids = {str(row["sample_id"]) for row in existing}
    if len(existing_ids) != len(existing):
        raise ValueError(f"Duplicate sample IDs already exist in {output}.")
    for row in existing:
        if row.get("evaluator_role") != args.evaluator or row.get("evaluator_id") != model_id:
            raise ValueError("Existing evaluator feature file belongs to a different evaluator configuration.")

    pending = [row for row in dataset if row["sample_id"] not in existing_ids]
    if args.limit is not None:
        pending = pending[: max(0, int(args.limit))]
    if not pending:
        print("No evaluator feature records are pending for the requested run.")
        return

    bundle = load_evaluator(model_id)
    new_count = 0
    try:
        for index, record in enumerate(pending, start=1):
            features = score_response_distribution(
                model=bundle.model,
                tokenizer=bundle.tokenizer,
                prompt=str(record["prompt"]),
                response=str(record["text"]),
            )
            output_record = {
                "sample_id": record["sample_id"],
                "evaluator_role": args.evaluator,
                "evaluator_id": model_id,
                "model_revision": bundle.model_revision,
                "tokenizer_revision": bundle.tokenizer_revision,
                **features,
            }
            append_jsonl_record(output_record, output)
            new_count += 1
            print(
                f"[{index}/{len(pending)}] {args.evaluator}: {record['sample_id']} "
                f"tokens={features['response_token_count']}"
            )
    finally:
        del bundle
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    all_rows = read_jsonl(output)
    summary = {
        "protocol_version": "step8_static_v1",
        "evaluator_role": args.evaluator,
        "evaluator_id": model_id,
        "record_count": len(all_rows),
        "new_records_this_run": new_count,
        "expected_record_count": len(dataset),
        "complete": len(all_rows) == len(dataset),
        "dataset_sha256": sha256_file(paths["detection_dataset_jsonl"]),
        "output_sha256": sha256_file(output),
    }
    atomic_write_json(summary, summary_path)
    print("Evaluator feature extraction run complete.")
    print(summary)


if __name__ == "__main__":
    main()
