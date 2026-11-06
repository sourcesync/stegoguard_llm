from __future__ import annotations

import argparse
import gc
from pathlib import Path

import yaml

from stegoguard_lm.detection.evaluator_scoring import load_evaluator
from stegoguard_lm.detection.io_utils import append_jsonl_record, atomic_write_json, read_jsonl, sha256_file
from stegoguard_lm.detection.semantic import paraphrase_embedding_similarity, visible_instruction_retention


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Step 8 semantic consistency features.")
    parser.add_argument("--config", default="config.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    paths = config["paths"]
    output = Path(paths["detection_semantic_features_jsonl"])
    summary_path = Path(paths["detection_semantic_summary_json"])
    static_cfg = config["static_detection"]
    semantic_role = str(static_cfg["semantic_evaluator_role"])
    model_cfg = next(item for item in config["models"]["evaluators"] if item["role"] == semantic_role)
    model_id = str(model_cfg["id"])
    dataset = read_jsonl(paths["detection_dataset_jsonl"])

    if args.overwrite:
        output.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    elif output.exists() and not args.resume:
        raise FileExistsError(f"Semantic output exists: {output}. Use --resume or --overwrite.")

    existing = read_jsonl(output) if output.exists() else []
    existing_ids = {str(row["sample_id"]) for row in existing}
    if len(existing_ids) != len(existing):
        raise ValueError("Duplicate semantic feature records detected.")
    pending = [row for row in dataset if row["sample_id"] not in existing_ids]
    if args.limit is not None:
        pending = pending[: max(0, int(args.limit))]
    if not pending:
        print("No semantic feature records are pending for the requested run.")
        return

    bundle = load_evaluator(model_id)
    new_count = 0
    try:
        for index, record in enumerate(pending, start=1):
            retention = visible_instruction_retention(
                model=bundle.model,
                tokenizer=bundle.tokenizer,
                visible_prompt=str(record["prompt"]),
                response=str(record["text"]),
            )
            similarity, paraphrase = paraphrase_embedding_similarity(
                model=bundle.model,
                tokenizer=bundle.tokenizer,
                response=str(record["text"]),
                max_extra_tokens=int(static_cfg["semantic_paraphrase_max_extra_tokens"]),
            )
            result = {
                "sample_id": record["sample_id"],
                "semantic_evaluator_role": semantic_role,
                "semantic_evaluator_id": model_id,
                "model_revision": bundle.model_revision,
                "tokenizer_revision": bundle.tokenizer_revision,
                "visible_instruction_retention": float(retention),
                "paraphrase_embedding_similarity": float(similarity),
                "paraphrase": paraphrase,
            }
            append_jsonl_record(result, output)
            new_count += 1
            print(
                f"[{index}/{len(pending)}] semantic: {record['sample_id']} "
                f"retention={retention:.4f} similarity={similarity:.4f}"
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

    rows = read_jsonl(output)
    summary = {
        "protocol_version": "step8_static_v1",
        "semantic_evaluator_role": semantic_role,
        "semantic_evaluator_id": model_id,
        "record_count": len(rows),
        "new_records_this_run": new_count,
        "expected_record_count": len(dataset),
        "complete": len(rows) == len(dataset),
        "dataset_sha256": sha256_file(paths["detection_dataset_jsonl"]),
        "output_sha256": sha256_file(output),
    }
    atomic_write_json(summary, summary_path)
    print("Semantic feature extraction run complete.")
    print(summary)


if __name__ == "__main__":
    main()
