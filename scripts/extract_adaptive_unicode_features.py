from __future__ import annotations

import argparse
from pathlib import Path

from stegoguard_lm.adaptive.io import load_yaml, verify_step10_protocol
from stegoguard_lm.detection.io_utils import atomic_write_jsonl, read_jsonl
from stegoguard_lm.detection.unicode_features import extract_unicode_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract deterministic Unicode features for Step 10 candidates and clean controls.")
    parser.add_argument("--config", default="config_step10.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    verify_step10_protocol(config)
    paths = config["paths"]
    output = Path(paths["adaptive_detection_unicode_features_jsonl"])
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Adaptive Unicode feature file exists: {output}")
    dataset = read_jsonl(paths["adaptive_detection_dataset_jsonl"])
    rows = [{"sample_id": str(r["sample_id"]), **extract_unicode_features(str(r["text"]))} for r in dataset]
    atomic_write_jsonl(rows, output)
    print("Adaptive Unicode feature extraction complete.")
    print(f"Records: {len(rows)}")
    print(f"Suspicious: {sum(int(r['has_suspicious_codepoint']) for r in rows)}")


if __name__ == "__main__":
    main()
