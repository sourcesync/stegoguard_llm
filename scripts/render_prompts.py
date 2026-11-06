from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from stegoguard_lm.prompt_rendering import (
    load_conflict_templates,
    load_topics,
    render_all_prompts,
    summarize_records,
    write_jsonl,
    write_preview,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render StegoGuard-LM pilot prompts.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    topics_csv = Path(config["paths"]["topics_csv"])
    conflict_templates_csv = Path(config["paths"]["conflict_templates_csv"])
    output_dir = Path(config["paths"]["output_dir"])

    lengths = config["pilot"]["lengths"]

    prompt_config = config.get("prompt_rendering", {})
    output_subdir = prompt_config.get("output_subdir", "prompts")
    allowed_conflict_ids = prompt_config.get("conflict_templates_for_pilot", None)

    topics = load_topics(topics_csv)
    conflict_templates = load_conflict_templates(conflict_templates_csv)

    records = render_all_prompts(
        topics=topics,
        conflict_templates=conflict_templates,
        lengths=lengths,
        allowed_conflict_ids=allowed_conflict_ids,
    )

    prompt_output_dir = output_dir / output_subdir
    jsonl_path = prompt_output_dir / "rendered_prompts.jsonl"
    preview_path = prompt_output_dir / "prompt_preview.txt"
    summary_path = prompt_output_dir / "prompt_summary.json"

    write_jsonl(records, jsonl_path)
    write_preview(records, preview_path, limit=20)

    summary = summarize_records(records)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Prompt rendering complete.")
    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {preview_path}")
    print(f"Wrote: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
