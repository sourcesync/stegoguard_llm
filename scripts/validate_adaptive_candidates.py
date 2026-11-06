from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from stegoguard_lm.adaptive.io import load_yaml, verify_step10_protocol
from stegoguard_lm.detection.io_utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 10 adaptive candidate accounting and model-free invariants.")
    parser.add_argument("--config", default="config_step10.yaml")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    verify_step10_protocol(config)
    paths = config["paths"]
    specs = read_jsonl(paths["adaptive_candidate_manifest_jsonl"])
    accepted = read_jsonl(paths["adaptive_candidate_samples_jsonl"]) if Path(paths["adaptive_candidate_samples_jsonl"]).exists() else []
    failures = read_jsonl(paths["adaptive_candidate_failures_jsonl"]) if Path(paths["adaptive_candidate_failures_jsonl"]).exists() else []
    spec_ids = {str(r["candidate_id"]) for r in specs}
    accepted_ids = [str(r["candidate_id"]) for r in accepted]
    failure_ids = [str(r["candidate_id"]) for r in failures]
    errors: list[str] = []
    if len(set(accepted_ids)) != len(accepted_ids):
        errors.append("Duplicate accepted candidate IDs.")
    if len(set(failure_ids)) != len(failure_ids):
        errors.append("Duplicate terminal candidate failure IDs.")
    overlap = set(accepted_ids) & set(failure_ids)
    if overlap:
        errors.append(f"Candidates are both accepted and failed: {sorted(overlap)[:5]}")
    unknown = (set(accepted_ids) | set(failure_ids)) - spec_ids
    if unknown:
        errors.append(f"Outputs contain unknown candidate IDs: {sorted(unknown)[:5]}")
    for row in accepted:
        if not bool(row.get("decode_success")) or not bool(row.get("exact_match")):
            errors.append(f"Accepted candidate lacks exact pre-defense decode: {row['candidate_id']}")
        if str(row.get("payload")) != str(row.get("decoded_payload")):
            errors.append(f"Payload mismatch: {row['candidate_id']}")
        if int(row.get("payload_bit_length", -1)) != len(str(row.get("payload", ""))):
            errors.append(f"Payload length metadata mismatch: {row['candidate_id']}")
    if args.require_complete:
        missing = spec_ids - set(accepted_ids) - set(failure_ids)
        if missing:
            errors.append(f"Pending candidate tasks remain: {len(missing)}")
        # Strategy 0 is a required anchor for every parent/family.
        anchors = {(r["clean_parent_id"], r["attack_name"]) for r in specs if int(r["strategy_index"]) == 0}
        accepted_anchors = {(r["parent_sample_id"], r["attack_name"]) for r in accepted if int(r["strategy_index"]) == 0}
        missing_anchors = anchors - accepted_anchors
        if missing_anchors:
            errors.append(f"Missing required static strategy-0 anchors: {sorted(missing_anchors)[:5]}")
    if errors:
        print("Adaptive candidate validation failed.")
        for error in errors[:100]:
            print(f"- {error}")
        raise SystemExit(1)
    counts = Counter(str(r["attack_name"]) for r in accepted)
    print("Adaptive candidate validation passed.")
    print(f"Candidate specifications: {len(specs)}")
    print(f"Accepted: {len(accepted)}")
    print(f"Terminal failures: {len(failures)}")
    print(f"Accepted by attack: {dict(counts)}")


if __name__ == "__main__":
    main()
