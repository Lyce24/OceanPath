#!/usr/bin/env python
"""
Copy the generated result tables into README.md.

Each block in the README between `<!-- results:NAME -->` and `<!-- /results:NAME -->`
is replaced by the contents of the matching file, so the README always shows what the
collectors last wrote:

    supervised        results/supervised/tables.md        (collect_supervised_results.py)
    label-efficiency  results/label_efficiency/tables.md  (collect_label_efficiency.py)
    scaling           results/scaling/scaling.md          (benchmark_scaling.py)
    serving           results/serving_latency/latency.md  (benchmark_serving.py)

Usage:
    python scripts/update_readme_results.py [--check]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BLOCKS = {
    "supervised": REPO / "results" / "supervised" / "tables.md",
    "label-efficiency": REPO / "results" / "label_efficiency" / "tables.md",
    "scaling": REPO / "results" / "scaling" / "scaling.md",
    "serving": REPO / "results" / "serving_latency" / "latency.md",
}


def render(readme: str) -> str:
    for name, source in BLOCKS.items():
        pattern = re.compile(
            rf"(<!-- results:{name} -->\n)(.*?)(<!-- /results:{name} -->)", re.DOTALL
        )
        if not pattern.search(readme) or not source.is_file():
            continue
        body = source.read_text().strip() + "\n"
        readme = pattern.sub(lambda m, body=body: m.group(1) + body + m.group(3), readme)
    return readme


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the README is stale")
    args = parser.parse_args()
    path = REPO / "README.md"
    current = path.read_text()
    updated = render(current)
    if args.check:
        sys.exit(0 if updated == current else 1)
    path.write_text(updated)


if __name__ == "__main__":
    main()
