from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml

from stegoguard_lm.detection.io_utils import atomic_write_json, read_jsonl, sha256_file


def _index(rows, label):
    out = {}
    for row in rows:
        key = str(row.get("sample_id", ""))
        if not key:
            raise ValueError(f"{label} row lacks sample_id")
        if key in out:
            raise ValueError(f"Duplicate {label} sample_id: {key}")
        out[key] = row
    return out


def _finite(value, label):
    if not math.isfinite(float(value)):
        raise ValueError(f"Non-finite feature value: {label}={value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate complete Step 8 feature extraction.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    paths = config["paths"]
    dataset = read_jsonl(paths["detection_dataset_jsonl"])
    expected_ids = {str(row["sample_id"]) for row in dataset}
    sources = {
        "evaluator_a": paths["detection_evaluator_a_features_jsonl"],
        "evaluator_b": paths["detection_evaluator_b_features_jsonl"],
        "semantic": paths["detection_semantic_features_jsonl"],
        "unicode": paths["detection_unicode_features_jsonl"],
    }
    indices = {}
    for label, path in sources.items():
        rows = read_jsonl(path)
        indices[label] = _index(rows, label)
        missing = expected_ids - set(indices[label])
        extra = set(indices[label]) - expected_ids
        if extra:
            raise ValueError(f"{label} contains unknown sample IDs: {sorted(extra)[:5]}")
        if missing and not args.allow_partial:
            raise ValueError(f"{label} is missing {len(missing)} records; first={sorted(missing)[:5]}")

    complete_ids = set.intersection(*(set(index) for index in indices.values()))
    for sample_id in complete_ids:
        for evaluator in ("evaluator_a", "evaluator_b"):
            row = indices[evaluator][sample_id]
            token_logprobs = row.get("token_logprobs")
            if not isinstance(token_logprobs, list) or not token_logprobs:
                raise ValueError(f"Missing token_logprobs for {evaluator}/{sample_id}")
            if len(token_logprobs) != int(row["response_token_count"]):
                raise ValueError(f"Token-logprob length mismatch for {evaluator}/{sample_id}")
            for field in ("mean_logp", "ppl", "logp_var", "sentence_logp_var"):
                _finite(row[field], f"{evaluator}/{sample_id}/{field}")
            if float(row["ppl"]) <= 0:
                raise ValueError(f"Non-positive perplexity for {evaluator}/{sample_id}")
        semantic = indices["semantic"][sample_id]
        _finite(semantic["visible_instruction_retention"], f"semantic/{sample_id}/retention")
        _finite(semantic["paraphrase_embedding_similarity"], f"semantic/{sample_id}/similarity")
        if not 0.0 <= float(semantic["visible_instruction_retention"]) <= 1.0:
            raise ValueError(f"Retention outside [0,1] for {sample_id}")
        if not -1.000001 <= float(semantic["paraphrase_embedding_similarity"]) <= 1.000001:
            raise ValueError(f"Cosine similarity outside [-1,1] for {sample_id}")
        unicode = indices["unicode"][sample_id]
        for field in ("nonprinting_count", "zero_width_count", "unicode_tag_count"):
            if int(unicode[field]) < 0:
                raise ValueError(f"Negative Unicode count for {sample_id}/{field}")
        if int(unicode["has_suspicious_codepoint"]) not in {0, 1}:
            raise ValueError(f"Invalid Unicode flag for {sample_id}")

    manifest = {
        "protocol_version": "step8_static_v1",
        "expected_record_count": len(dataset),
        "complete_record_count": len(complete_ids),
        "complete": len(complete_ids) == len(dataset),
        "dataset_sha256": sha256_file(paths["detection_dataset_jsonl"]),
        "feature_hashes": {
            label: sha256_file(path) for label, path in sources.items()
        },
    }
    atomic_write_json(manifest, paths["detection_feature_manifest_json"])
    print("Step 8 feature validation passed.")
    print(f"Expected records: {len(dataset)}")
    print(f"Complete records across all feature families: {len(complete_ids)}")
    print(f"Complete: {manifest['complete']}")
    print(f"Wrote: {paths['detection_feature_manifest_json']}")


if __name__ == "__main__":
    main()
