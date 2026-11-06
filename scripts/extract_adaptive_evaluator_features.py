from __future__ import annotations

import argparse
import gc
from pathlib import Path

from stegoguard_lm.adaptive.io import load_yaml, verify_step10_protocol
from stegoguard_lm.detection.evaluator_scoring import load_evaluator, score_response_distribution
from stegoguard_lm.detection.io_utils import append_jsonl_record, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Step 10 evaluator features, reusing frozen Step 8 clean-control features.")
    parser.add_argument("--config", default="config_step10.yaml")
    parser.add_argument("--evaluator", choices=["evaluator_a", "evaluator_b"], required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    verify_step10_protocol(config)
    paths = config["paths"]
    dataset = read_jsonl(paths["adaptive_detection_dataset_jsonl"])
    if args.evaluator == "evaluator_a":
        output = Path(paths["adaptive_detection_evaluator_a_features_jsonl"])
        static_path = paths["detection_evaluator_a_features_jsonl"]
    else:
        output = Path(paths["adaptive_detection_evaluator_b_features_jsonl"])
        static_path = paths["detection_evaluator_b_features_jsonl"]
    existing = read_jsonl(output) if output.exists() else []
    existing_ids = {str(r["sample_id"]) for r in existing}
    if existing_ids and not args.resume:
        raise FileExistsError(f"Adaptive evaluator features exist: {output}. Use --resume.")
    static = {str(r["sample_id"]): r for r in read_jsonl(static_path)}
    model_id = next(str(m["id"]) for m in config["models"]["evaluators"] if m["role"] == args.evaluator)

    pending = [r for r in dataset if str(r["sample_id"]) not in existing_ids]
    if args.limit is not None:
        pending = pending[: max(0, int(args.limit))]
    model_pending = []
    for row in pending:
        sample_id = str(row["sample_id"])
        if row["adaptive_dataset_role"] == "clean_control" and sample_id in static:
            copied = dict(static[sample_id])
            copied["reused_from_step8"] = True
            append_jsonl_record(copied, output)
            existing_ids.add(sample_id)
            print(f"Reused Step 8 {args.evaluator}: {sample_id}")
        else:
            model_pending.append(row)
    if not model_pending:
        print("No new model-scored evaluator features are pending.")
        return
    bundle = load_evaluator(model_id)
    try:
        for index, row in enumerate(model_pending, start=1):
            features = score_response_distribution(
                model=bundle.model,
                tokenizer=bundle.tokenizer,
                prompt=str(row["prompt"]),
                response=str(row["text"]),
            )
            append_jsonl_record(
                {
                    "sample_id": str(row["sample_id"]),
                    "evaluator_role": args.evaluator,
                    "evaluator_id": model_id,
                    "model_revision": bundle.model_revision,
                    "tokenizer_revision": bundle.tokenizer_revision,
                    "reused_from_step8": False,
                    **features,
                },
                output,
            )
            print(f"[{index}/{len(model_pending)}] {args.evaluator}: {row['sample_id']}")
    finally:
        del bundle
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
