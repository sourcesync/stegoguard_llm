from __future__ import annotations

import argparse
import gc
from pathlib import Path

from stegoguard_lm.adaptive.io import load_yaml, verify_step10_protocol
from stegoguard_lm.detection.evaluator_scoring import load_evaluator
from stegoguard_lm.detection.io_utils import append_jsonl_record, read_jsonl
from stegoguard_lm.detection.semantic import paraphrase_embedding_similarity, visible_instruction_retention


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Step 10 semantic features, reusing frozen Step 8 clean controls.")
    parser.add_argument("--config", default="config_step10.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    verify_step10_protocol(config)
    paths = config["paths"]
    output = Path(paths["adaptive_detection_semantic_features_jsonl"])
    existing = read_jsonl(output) if output.exists() else []
    existing_ids = {str(r["sample_id"]) for r in existing}
    if existing_ids and not args.resume:
        raise FileExistsError(f"Adaptive semantic features exist: {output}. Use --resume.")
    dataset = read_jsonl(paths["adaptive_detection_dataset_jsonl"])
    static = {str(r["sample_id"]): r for r in read_jsonl(paths["detection_semantic_features_jsonl"])}
    semantic_role = str(config["static_detection"]["semantic_evaluator_role"])
    model_id = next(str(m["id"]) for m in config["models"]["evaluators"] if m["role"] == semantic_role)
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
            print(f"Reused Step 8 semantic: {sample_id}")
        else:
            model_pending.append(row)
    if not model_pending:
        print("No new model-scored semantic features are pending.")
        return
    bundle = load_evaluator(model_id)
    try:
        for index, row in enumerate(model_pending, start=1):
            retention = visible_instruction_retention(
                model=bundle.model,
                tokenizer=bundle.tokenizer,
                visible_prompt=str(row["prompt"]),
                response=str(row["text"]),
            )
            similarity, paraphrase = paraphrase_embedding_similarity(
                model=bundle.model,
                tokenizer=bundle.tokenizer,
                response=str(row["text"]),
                max_extra_tokens=int(config["static_detection"]["semantic_paraphrase_max_extra_tokens"]),
            )
            append_jsonl_record(
                {
                    "sample_id": str(row["sample_id"]),
                    "semantic_evaluator_role": semantic_role,
                    "semantic_evaluator_id": model_id,
                    "model_revision": bundle.model_revision,
                    "tokenizer_revision": bundle.tokenizer_revision,
                    "visible_instruction_retention": float(retention),
                    "paraphrase_embedding_similarity": float(similarity),
                    "paraphrase": paraphrase,
                    "reused_from_step8": False,
                },
                output,
            )
            print(f"[{index}/{len(model_pending)}] semantic: {row['sample_id']}")
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
