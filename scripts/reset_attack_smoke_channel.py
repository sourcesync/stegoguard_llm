from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


CHANNEL_TO_ATTACK = {
    "unicode": "unicode_hidden_channel",
    "synonym": "synonym_slot_codebook",
    "rank": "rank_encoding",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove accepted smoke records for one attack channel only."
    )
    parser.add_argument(
        "--samples",
        default="outputs/pilot_v0/attacks/smoke/attack_smoke_samples.jsonl",
    )
    parser.add_argument(
        "--channel",
        choices=sorted(CHANNEL_TO_ATTACK),
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.samples)
    if not path.exists():
        raise FileNotFoundError(f"Smoke sample file not found: {path}")

    attack_name = CHANNEL_TO_ATTACK[args.channel]
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kept = [record for record in records if record.get("attack_name") != attack_name]
    removed = [record for record in records if record.get("attack_name") == attack_name]

    backup = path.with_suffix(path.suffix + f".before_{args.channel}_reset.bak")
    shutil.copy2(path, backup)

    with path.open("w", encoding="utf-8") as file:
        for record in kept:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Backup: {backup}")
    print(f"Removed {len(removed)} {args.channel} records.")
    print(f"Preserved {len(kept)} other attack records.")
    for record in removed:
        print(f"  removed: {record.get('sample_id')}")


if __name__ == "__main__":
    main()
