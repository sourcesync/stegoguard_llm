"""Check repository files against the frozen SHA-256 manifests.

The freeze manifests in outputs/pilot_v0/manifests/ record a hash for every
config, protocol document, script, source module, and output that each
pipeline step depended on. This script re-hashes whatever is present in the
working tree and reports:

  match     file present and byte-identical to the frozen version
  MISMATCH  file present but changed since it was frozen
  missing   file not in the working tree (expected for the large JSON/JSONL
            outputs, which are distributed separately; see data/README.md)

Exit status is 1 if any present file mismatches, 0 otherwise.

Usage:
    python scripts/verify_code_hashes.py
    python scripts/verify_code_hashes.py --show-missing
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

LINE = re.compile(r"^([0-9a-f]{64})\s+(\S.*)$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--show-missing", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifests = sorted((root / "outputs/pilot_v0/manifests").glob("*hashes*.txt"))
    if not manifests:
        print("No freeze manifests found under outputs/pilot_v0/manifests/.")
        return 1

    results: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for manifest in manifests:
        for raw in manifest.read_text(encoding="utf-8").splitlines():
            m = LINE.match(raw.strip())
            if not m:
                continue
            expected, rel = m.groups()
            rel = rel.replace("\\", "/")
            # Some early manifests recorded absolute Windows paths; keep the
            # repository-relative tail.
            if re.match(r"^[A-Za-z]:/", rel) and "/stegoguard_lm/" in rel:
                rel = rel.split("/stegoguard_lm/", 1)[1]
            path = root / rel
            if not path.is_file():
                results["missing"].append((manifest.name, rel))
            elif sha256(path) == expected:
                results["match"].append((manifest.name, rel))
            else:
                results["MISMATCH"].append((manifest.name, rel))

    for key in ("match", "MISMATCH", "missing"):
        print(f"{key:9s} {len(results[key])}")
    for name, rel in results["MISMATCH"]:
        print(f"  MISMATCH  {rel}  ({name})")
    if args.show_missing:
        for name, rel in results["missing"]:
            print(f"  missing   {rel}  ({name})")
    return 1 if results["MISMATCH"] else 0


if __name__ == "__main__":
    sys.exit(main())
