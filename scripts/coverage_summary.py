"""Report test line coverage of each oceanpath subpackage, as a Markdown table.

    uv run --no-sync pytest -m "not slow and not gpu and not integration" --cov
    uv run --no-sync python scripts/coverage_summary.py

Reads the ``.coverage`` data a ``pytest --cov`` run leaves behind ([tool.coverage] in
pyproject.toml). Modules no test imports count as uncovered, so the Mamba-2 models (CUDA)
and TRIDENT feature extraction (whole-slide images) read low on a CPU run.

CI appends the table to the job summary. It reports and never fails on the numbers.
"""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from pathlib import Path

import coverage


def package_totals() -> dict[str, list[int]]:
    """[statements, covered lines] per subpackage of oceanpath."""
    cov = coverage.Coverage()
    cov.load()
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder) / "coverage.json"
        cov.json_report(outfile=str(out))
        files = json.loads(out.read_text())["files"]
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for name, entry in files.items():
        parts = Path(name).parts
        inner = parts[parts.index("oceanpath") + 1 :]
        package = inner[0] if len(inner) > 1 else "(top level)"
        totals[package][0] += entry["summary"]["num_statements"]
        totals[package][1] += entry["summary"]["covered_lines"]
    return dict(sorted(totals.items()))


def report() -> str:
    totals = package_totals()
    lines = ["| Package | Statements | Covered |", "| --- | ---: | ---: |"]
    for package, (statements, covered) in totals.items():
        lines.append(f"| `{package}` | {statements:,} | {_share(covered, statements)} |")
    statements = sum(t[0] for t in totals.values())
    covered = sum(t[1] for t in totals.values())
    lines.append(f"| **All** | **{statements:,}** | **{_share(covered, statements)}** |")
    return "### Test coverage\n\n" + "\n".join(lines) + "\n"


def _share(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "–"


if __name__ == "__main__":
    print(report())
