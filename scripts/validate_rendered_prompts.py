from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_FIELDS = {
    "prompt_id",
    "prompt_type",
    "domain",
    "topic_id",
    "topic_seed",
    "template_version",
    "target_length",
    "visible_prompt",
    "hidden_payload",
    "conflict_id",
    "decode_rule",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate rendered prompt JSONL.")
    parser.add_argument(
        "--prompts",
        type=str,
        default="outputs/pilot_v0/prompts/rendered_prompts.jsonl",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.prompts)

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    seen_ids = set()
    total = 0
    standard = 0
    conflict = 0

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            total += 1
            record = json.loads(line)

            missing = REQUIRED_FIELDS - set(record.keys())
            if missing:
                raise ValueError(f"Line {line_number} missing fields: {missing}")

            prompt_id = record["prompt_id"]
            if prompt_id in seen_ids:
                raise ValueError(f"Duplicate prompt_id found: {prompt_id}")
            seen_ids.add(prompt_id)

            if not record["visible_prompt"].strip():
                raise ValueError(f"Empty visible_prompt at line {line_number}")

            if record["prompt_type"] == "standard":
                standard += 1
                if record["hidden_payload"] is not None:
                    raise ValueError(f"Standard prompt has hidden payload: {prompt_id}")
                if record["conflict_id"] is not None:
                    raise ValueError(f"Standard prompt has conflict_id: {prompt_id}")

            elif record["prompt_type"] == "conflict":
                conflict += 1
                if not record["hidden_payload"]:
                    raise ValueError(
                        f"Conflict prompt missing hidden payload: {prompt_id}"
                    )
                if not record["conflict_id"]:
                    raise ValueError(
                        f"Conflict prompt missing conflict_id: {prompt_id}"
                    )
                if not record["decode_rule"]:
                    raise ValueError(
                        f"Conflict prompt missing decode_rule: {prompt_id}"
                    )

                hidden = record["hidden_payload"].lower()
                visible = record["visible_prompt"].lower()
                if hidden in visible:
                    raise ValueError(
                        f"Hidden payload appears inside visible prompt: {prompt_id}"
                    )

            else:
                raise ValueError(
                    f"Unknown prompt_type at line {line_number}: {record['prompt_type']}"
                )

    print("Rendered prompt validation passed.")
    print(f"Total records: {total}")
    print(f"Standard prompts: {standard}")
    print(f"Conflict prompts: {conflict}")


if __name__ == "__main__":
    main()
