"""Build the self-contained results explorer page.

Reads outputs/explorer/explorer_bundle.json (made by export_explorer_bundle.py)
and embeds it in explorer/template.html. The result is a single HTML file that
opens in any browser with no server. Run from the repository root:

    python scripts/export_explorer_bundle.py
    python scripts/build_explorer_page.py
    python scripts/build_explorer_page.py --out path/to/index.html

Added after the experiments; not covered by the freeze manifests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PLACEHOLDER = "/*DATA*/null"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", default="outputs/explorer/explorer_bundle.json")
    ap.add_argument("--template", default="explorer/template.html")
    ap.add_argument("--out", default="outputs/explorer/index.html")
    args = ap.parse_args()

    template = Path(args.template).read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != 1:
        raise ValueError(f"{args.template}: expected exactly one {PLACEHOLDER!r}")
    data = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
    blob = (json.dumps(data, separators=(",", ":"), ensure_ascii=False)
            .replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    # The template is a page fragment; add a charset so it also opens straight from disk.
    page = '<meta charset="utf-8">\n' + template.replace(PLACEHOLDER, blob)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
