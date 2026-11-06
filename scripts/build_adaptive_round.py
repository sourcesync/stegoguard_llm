from __future__ import annotations

import argparse
from pathlib import Path

from stegoguard_lm.adaptive.io import load_yaml, round_paths, verify_step10_protocol
from stegoguard_lm.adaptive.rounds import build_round_tasks
from stegoguard_lm.detection.io_utils import atomic_write_jsonl, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one Step 10 adaptive round from only prior finalized feedback.")
    parser.add_argument("--config", default="config_step10.yaml")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    verify_step10_protocol(config)
    rp = round_paths(config, args.round)
    if rp["tasks"].exists() and not args.overwrite:
        raise FileExistsError(f"Round task manifest exists: {rp['tasks']}")
    prior_feedback = []
    for r in range(args.round):
        path = round_paths(config, r)["feedback"]
        if not path.exists():
            raise FileNotFoundError(f"Round {r} feedback must be finalized before building round {args.round}: {path}")
        prior_feedback.extend(read_jsonl(path))
    static_dataset = read_jsonl(config["paths"]["detection_dataset_jsonl"])
    clean = [row for row in static_dataset if row["label"] == "clean" and bool(row["is_balanced_common"])]
    tasks = build_round_tasks(
        config=config,
        round_id=args.round,
        trajectories=read_jsonl(config["paths"]["adaptive_trajectory_manifest_jsonl"]),
        candidates=read_jsonl(config["paths"]["adaptive_candidate_samples_jsonl"]),
        detector_scores=read_jsonl(config["paths"]["adaptive_detector_scores_jsonl"]),
        prior_all_feedback=prior_feedback,
        clean_rows=clean,
        static_unicode_features=read_jsonl(config["paths"]["detection_unicode_features_jsonl"]),
    )
    atomic_write_jsonl(tasks, rp["tasks"])
    stego = sum(r["record_role"] == "stego_trajectory" for r in tasks)
    clean_n = sum(r["record_role"] == "clean_control" for r in tasks)
    print(f"Step 10 round {args.round} task manifest built.")
    print(f"Stego trajectories: {stego}")
    print(f"Clean controls: {clean_n}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Wrote: {rp['tasks']}")


if __name__ == "__main__":
    main()
