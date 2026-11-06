from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from stegoguard_lm.detection.datasets import build_detection_dataset
from stegoguard_lm.detection.io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_sha256,
    read_json,
    read_jsonl,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen Step 8 detection dataset.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    paths = config["paths"]
    output_path = Path(paths["detection_dataset_jsonl"])
    manifest_path = Path(paths["detection_dataset_manifest_json"])
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Detection dataset already exists: {output_path}. Use --overwrite intentionally."
        )

    clean_path = Path(paths["clean_samples_full_jsonl"])
    stego_path = Path(paths["attack_full_samples_jsonl"])
    step7_summary_path = Path(paths["attack_full_summary_json"])
    clean_records = read_jsonl(clean_path)
    stego_records = read_jsonl(stego_path)
    step7_summary = read_json(step7_summary_path)

    static_config = config["static_detection"]
    expected_clean = int(static_config["expected_clean_records"])
    expected_stego = int(static_config["expected_accepted_stego_records"])
    if len(clean_records) != expected_clean:
        raise ValueError(f"Expected {expected_clean} clean records, found {len(clean_records)}.")
    if len(stego_records) != expected_stego:
        raise ValueError(f"Expected {expected_stego} accepted stego records, found {len(stego_records)}.")
    if int(step7_summary.get("accepted_tasks", -1)) != expected_stego:
        raise ValueError("Step 7 summary accepted_tasks disagrees with the frozen stego corpus.")
    if int(step7_summary.get("pending_tasks", -1)) != 0:
        raise ValueError("Step 7 still has pending attack tasks; Step 8 cannot begin.")

    records, manifest = build_detection_dataset(
        clean_records=clean_records,
        stego_records=stego_records,
        seed=int(config["seed"]),
        balanced_per_cell=int(static_config["balanced_common_parents_per_cell"]),
    )
    atomic_write_jsonl(records, output_path)
    manifest.update(
        {
            "dataset_path": str(output_path),
            "dataset_sha256": sha256_file(output_path),
            "clean_corpus_sha256": sha256_file(clean_path),
            "stego_corpus_sha256": sha256_file(stego_path),
            "step7_summary_sha256": sha256_file(step7_summary_path),
            "static_detection_config_sha256": canonical_sha256(static_config),
        }
    )
    atomic_write_json(manifest, manifest_path)

    print("Step 8 detection dataset built and frozen.")
    print(f"Records: {manifest['record_count']}")
    print(f"Clean: {manifest['clean_record_count']}")
    print(f"Stego: {manifest['stego_record_count']}")
    print(f"Common all-three parents: {manifest['common_parent_count_all_three']}")
    print(f"Balanced common parents: {manifest['balanced_common_parent_count']}")
    print(f"Wrote: {output_path}")
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
