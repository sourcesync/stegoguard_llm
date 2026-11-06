from __future__ import annotations

import argparse
from pathlib import Path

from stegoguard_lm.adaptive.io import load_yaml, verify_step10_protocol
from stegoguard_lm.detection.io_utils import atomic_write_jsonl, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Step 10 detector dataset from balanced clean controls and accepted candidates.")
    parser.add_argument("--config", default="config_step10.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    verify_step10_protocol(config)
    paths = config["paths"]
    output = Path(paths["adaptive_detection_dataset_jsonl"])
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Adaptive detection dataset exists: {output}")
    static_dataset = read_jsonl(paths["detection_dataset_jsonl"])
    clean = [r for r in static_dataset if r["label"] == "clean" and bool(r["is_balanced_common"])]
    candidates = read_jsonl(paths["adaptive_candidate_samples_jsonl"])
    rows = []
    for row in clean:
        rows.append({**row, "adaptive_dataset_role": "clean_control", "protocol_version": config["adaptive_evaluation"]["protocol_version"]})
    for row in candidates:
        rows.append(
            {
                "sample_id": str(row["candidate_id"]),
                "clean_parent_id": str(row["parent_sample_id"]),
                "pair_id": str(row["pair_id"]),
                "lineage_id": str(row["lineage_id"]),
                "label": "stego",
                "y": 1,
                "attack_family": str(row["attack_family"]),
                "attack_name": str(row["attack_name"]),
                "source_model": str(row["source_model"]),
                "domain": str(row["domain"]),
                "topic_id": str(row["topic_id"]),
                "topic_seed": row.get("topic_seed"),
                "nominal_target_length": int(row["nominal_target_length"]),
                "parent_actual_token_length": int(row["parent_actual_token_length"]),
                "prompt_id": row.get("prompt_id"),
                "prompt": str(row["prompt"]),
                "text": str(row["text"]),
                "payload_bit_length": int(row["payload_bit_length"]),
                "payload_rate_bpt": float(row["payload_rate_bpt"]),
                "strategy_index": int(row["strategy_index"]),
                "strategy_id": str(row["strategy_id"]),
                "adaptive_dataset_role": "candidate",
                "protocol_version": config["adaptive_evaluation"]["protocol_version"],
            }
        )
    ids = [str(r["sample_id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample_id in Step 10 detector dataset.")
    atomic_write_jsonl(rows, output)
    print("Step 10 adaptive detection dataset built.")
    print(f"Clean controls: {len(clean)}")
    print(f"Accepted candidates: {len(candidates)}")
    print(f"Total: {len(rows)}")
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
