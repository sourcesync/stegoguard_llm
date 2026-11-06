from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from stegoguard_lm.attacks.synonym_channel import (
    load_synonym_codebook,
    validate_synonym_codebook,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the Step 6 synonym codebook.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--skip-tokenizer-audit",
        action="store_true",
        help="Run only structural checks without loading source tokenizers.",
    )
    parser.add_argument(
        "--report",
        default="outputs/pilot_v0/manifests/synonym_codebook_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as file:
        config: dict[str, Any] = yaml.safe_load(file)

    codebook = load_synonym_codebook(config["paths"]["synonym_codebook_yaml"])
    errors = validate_synonym_codebook(codebook)
    report: dict[str, Any] = {
        "codebook_version": codebook.version,
        "slot_count": len(codebook.slots),
        "structural_errors": errors,
        "tokenizer_audit": {},
    }

    if len(codebook.slots) != int(
        config["attacks"]["token_blackbox"]["slots_per_sample"]
    ):
        errors.append("Codebook slot count does not match config slots_per_sample.")

    if not args.skip_tokenizer_audit:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required unless --skip-tokenizer-audit is used."
            ) from exc

        for source in config["models"]["sources"]:
            model_id = source["id"]
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            model_rows = []
            for slot in codebook.slots:
                zero_ids = tokenizer.encode(" " + slot.zero, add_special_tokens=False)
                one_ids = tokenizer.encode(" " + slot.one, add_special_tokens=False)
                equal_length = len(zero_ids) == len(one_ids)
                if not equal_length:
                    errors.append(
                        f"{model_id} {slot.slot_id}: unequal leading-space token counts "
                        f"({len(zero_ids)} vs {len(one_ids)})."
                    )
                model_rows.append(
                    {
                        "slot_id": slot.slot_id,
                        "zero": slot.zero,
                        "one": slot.one,
                        "zero_token_ids": zero_ids,
                        "one_token_ids": one_ids,
                        "zero_token_count": len(zero_ids),
                        "one_token_count": len(one_ids),
                        "equal_token_count": equal_length,
                    }
                )
            report["tokenizer_audit"][model_id] = model_rows

    report["errors"] = errors
    report["passed"] = not errors
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if errors:
        print("Synonym codebook audit failed.")
        for error in errors:
            print(f"- {error}")
        print(f"Report: {report_path}")
        raise SystemExit(1)

    print("Synonym codebook audit passed.")
    print(f"Slots: {len(codebook.slots)}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
