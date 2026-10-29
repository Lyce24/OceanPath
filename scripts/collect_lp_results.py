#!/usr/bin/env python3
"""
Collect linear-probe summary.json files and print a results table.

Layouts supported
─────────────────
1. New sweep layout (run_lp_baselines.sh, outputs/lp_2048/<TS>/...):
       <root>/<task>__<mode>/<cache_tag>/summary.json

2. Label-efficiency sweep with seeds (run_label_efficiency.sh):
       <root>/<TS>/frac_<ftag>/<task>__<mode>/seed_<N>/<cache_tag>/summary.json
   Seeded runs are aggregated across seeds by default (mean ± std over seeds);
   pass --by-seed to inspect individual seeds.

3. Legacy layouts (older Hydra default output_dir interpolation):
       <root>/<data>_<encoder>_<mode>_<protocol>/<cache_tag>/summary.json
   or the even older flat form without a cache_tag subdir.

Usage
─────
    # Latest sweep under outputs/lp_2048/
    python scripts/collect_lp_results.py --latest

    # Specific sweep dir
    python scripts/collect_lp_results.py --root outputs/lp_2048/20260427_172249

    # Label-efficiency sweep with per-seed breakdown
    python scripts/collect_lp_results.py \
        --root outputs/label_efficiency_2048/20260508_120000 --by-seed

    # Walk an entire root (e.g. all timestamps), CSV output
    python scripts/collect_lp_results.py --root outputs/lp_2048 --csv all_runs.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

# ── Formatting helpers ──────────────────────────────────────────────────────


def _fmt_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return ""
    if std is not None and not math.isnan(std) and std > 0:
        return f"{mean:.3f} +/- {std:.3f}"
    return f"{mean:.3f}"


def _ms(s: dict, key: str) -> str:
    """Format mean +/- std from a grouped_cv summary."""
    m = s.get(f"mean_{key}")
    sd = s.get(f"std_{key}")
    return _fmt_mean_std(m, sd)


def _ci(s: dict, prefix: str) -> str:
    """Format [ci_low, ci_high] from bootstrap results."""
    lo = s.get(f"{prefix}_ci_low")
    hi = s.get(f"{prefix}_ci_high")
    if lo is None or hi is None:
        return ""
    return f"[{lo:.3f}, {hi:.3f}]"


def _pt(s: dict, key: str) -> str:
    """Format a point-estimate metric."""
    v = s.get(key)
    if v is None:
        return ""
    return f"{v:.3f}"


def _fraction_label(value: float | int | str) -> str:
    pct = float(value) * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"{round(pct)}%"
    return f"{pct:g}%"


# ── Label mappings ──────────────────────────────────────────────────────────

# Task tag (from sweep script) → display label
TASK_LABELS = {
    # New short-tag format from run_lp_baselines.sh
    "nsclc_cv": "NSCLC subtyping",
    "nsclc_xfer": "NSCLC cross-cohort",
    "braf": "BRAF mutation",
    "panda": "PANDA grading",
    "gej": "GEJ severity",
    # Legacy long-tag format (data.name from older runs)
    "tcga_nsclc": "NSCLC subtyping",
    "cptac_nsclc": "NSCLC subtyping",
    "TCGA_BRAF": "BRAF mutation",
    "gej_univ2": "GEJ severity",
}

TASK_ORDER = {
    "NSCLC subtyping": 0,
    "NSCLC cross-cohort": 1,
    "BRAF mutation": 2,
    "PANDA grading": 3,
    "GEJ severity": 4,
}

# Mode tag (from sweep script) → display label
MODE_LABELS = {
    "mean_pool": "Mean Pool",
    "max_pool": "Max Pool",
    "mean_max_pool": "Mean+Max Pool",
    "aggregator_random": "Random-init Mamba2MIL",
    "aggregator": "Pretrained Mamba2MIL",
    "aggregator_vicreg": "VICReg Mamba2MIL",
    "aggregator_jepa": "JEPA Mamba2MIL",
    "aggregator_lejepa": "LeJEPA Mamba2MIL",
    "ssl": "SSL Mamba2MIL",
}

MODE_ORDER = {
    "Mean Pool": 0,
    "Max Pool": 1,
    "Mean+Max Pool": 2,
    "Random-init Mamba2MIL": 3,
    "Pretrained Mamba2MIL": 4,
    "VICReg Mamba2MIL": 5,
    "JEPA Mamba2MIL": 6,
    "LeJEPA Mamba2MIL": 7,
    "SSL Mamba2MIL": 8,
}

KNOWN_PROTOCOLS = ("_grouped_cv", "_external_test", "_predefined_split")
KNOWN_ENCODERS = ("_uni2h", "_univ2", "_univ1", "_uni_v2", "_uni_v1")

SEED_DIR_RE = re.compile(r"^seed_(?P<seed>-?\d+)$")


# ── Mode label from cache-tag (legacy fallback) ─────────────────────────────


def _mode_label_from_cache_tag(tag: str) -> str:
    t = tag.lower()
    if t.startswith("aggregator"):
        if "random_init" in t:
            return "Random-init Mamba2MIL"
        return "Pretrained Mamba2MIL"
    if t.startswith("mean_max_pool"):
        return "Mean+Max Pool"
    if t.startswith("max_pool"):
        return "Max Pool"
    if t.startswith("mean_pool"):
        return "Mean Pool"
    return tag


# ── Outer-dir parsing ───────────────────────────────────────────────────────


def _parse_new_layout(outer_dir: str) -> tuple[str, str] | None:
    """Parse '<task>__<mode>' style outer dirs from run_lp_baselines.sh."""
    if "__" not in outer_dir:
        return None
    task_tag, _, mode_tag = outer_dir.partition("__")
    return task_tag, mode_tag


def _parse_legacy_task(outer_dir: str, protocol: str) -> str:
    """Parse legacy '<data>_<encoder>_<mode>_<protocol>' outer dirs."""
    name = outer_dir
    for suf in KNOWN_PROTOCOLS:
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    for suf in ("_aggregator", "_mean_max_pool", "_max_pool", "_mean_pool"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    for suf in KNOWN_ENCODERS:
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    label = TASK_LABELS.get(name, name)
    if protocol == "external_test" and "nsclc" in name.lower():
        label = "NSCLC cross-cohort"
    return label


def _infer_mode_from_legacy_dir(outer_dir: str) -> str:
    """Old-style flat layout: guess mode tag from outer dir name."""
    lower = outer_dir.lower()
    for mode_suf, label in [
        ("_aggregator_", "aggregator"),
        ("_mean_max_pool_", "mean_max_pool"),
        ("_max_pool_", "max_pool"),
        ("_mean_pool_", "mean_pool"),
    ]:
        if mode_suf in lower:
            return label
    return outer_dir


# ── Collect ─────────────────────────────────────────────────────────────────


def _resolve_layout(rel_parts: tuple[str, ...]) -> tuple[str, str, str, int | None] | None:
    """Locate (outer_dir, cache_tag, sweep_tag, seed) from path components.

    The summary.json's parent is always the cache_tag dir. Walk up to find a
    '<task>__<mode>' outer dir. Any 'seed_<N>' segment between the outer dir
    and the cache_tag becomes the per-seed selector.
    """
    if len(rel_parts) < 2:
        return None
    summary_idx = len(rel_parts) - 1
    cache_tag = rel_parts[summary_idx - 1]

    for i in range(summary_idx - 2, -1, -1):
        if "__" in rel_parts[i]:
            sweep_parts = rel_parts[:i]
            intermediate = rel_parts[i + 1 : summary_idx - 1]
            seed: int | None = None
            for inter in intermediate:
                m = SEED_DIR_RE.match(inter)
                if m:
                    seed = int(m.group("seed"))
                    break
            sweep_tag = "/".join(sweep_parts)
            return rel_parts[i], cache_tag, sweep_tag, seed

    # Legacy two-deep layout: .../<outer>/<cache_tag>/summary.json
    if summary_idx >= 2:
        grandparent = rel_parts[summary_idx - 2]
        sweep_tag = "/".join(rel_parts[: summary_idx - 2])
        return grandparent, cache_tag, sweep_tag, None

    # Flattest legacy: .../<outer>/summary.json
    return cache_tag, _infer_mode_from_legacy_dir(cache_tag), "", None


def _classify(outer_dir: str, cache_tag: str, protocol: str) -> tuple[str, str]:
    """Return (task_label, mode_label) — preferring the new layout's tags."""
    new = _parse_new_layout(outer_dir)
    if new is not None:
        task_tag, mode_tag = new
        task_label = TASK_LABELS.get(task_tag, task_tag)
        mode_label = MODE_LABELS.get(mode_tag, _mode_label_from_cache_tag(cache_tag))
        return task_label, mode_label

    task_label = _parse_legacy_task(outer_dir, protocol)
    mode_label = _mode_label_from_cache_tag(cache_tag)
    return task_label, mode_label


# Per-task-type metric specs: (column header, key in summary, supports CI).
_BINARY_CV_METRICS = [
    ("AUROC", "auroc", "oof_auroc"),
    ("AUPRC", "auprc", None),
    ("Bal. Acc", "balanced_acc", None),
    ("Macro F1", "macro_f1", None),
]
_BINARY_PT_METRICS = [
    ("AUROC", "auroc", "auroc"),
    ("AUPRC", "auprc", None),
    ("Bal. Acc", "balanced_acc", None),
    ("Macro F1", "macro_f1", None),
]
_MULTI_METRICS = [
    ("QWK", "qwk", None),
    ("Macro F1", "macro_f1", None),
    ("Bal. Acc", "balanced_acc", None),
]


def _summary_per_seed_value(s: dict, key: str, protocol: str) -> float | None:
    """Best per-seed point estimate for a metric, regardless of protocol."""
    if protocol == "grouped_cv":
        v = s.get(f"mean_{key}")
    else:
        v = s.get(key)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v


def _row_from_single_summary(s: dict, task: str, mode: str, sweep_dir: str) -> dict:
    protocol = s.get("protocol", "unknown")
    task_type = s.get("task_type", "unknown")

    row: dict = {
        "Sweep": sweep_dir,
        "Task": task,
        "Protocol": protocol,
        "Mode": mode,
    }

    if "label_fraction" in s:
        row["Fraction"] = _fraction_label(s["label_fraction"])
        row["_frac_ord"] = float(s["label_fraction"])
    else:
        row["_frac_ord"] = 1.0

    if protocol == "grouped_cv":
        if task_type == "binary":
            row["AUROC"] = f"{_ms(s, 'auroc')} {_ci(s, 'oof_auroc')}".strip()
            row["AUPRC"] = _ms(s, "auprc")
            row["Bal. Acc"] = _ms(s, "balanced_acc")
            row["Macro F1"] = _ms(s, "macro_f1")
        else:
            row["QWK"] = _ms(s, "qwk")
            row["Macro F1"] = _ms(s, "macro_f1")
            row["Bal. Acc"] = _ms(s, "balanced_acc")
    else:
        if task_type == "binary":
            row["AUROC"] = f"{_pt(s, 'auroc')} {_ci(s, 'auroc')}".strip()
            row["AUPRC"] = _pt(s, "auprc")
            row["Bal. Acc"] = _pt(s, "balanced_acc")
            row["Macro F1"] = _pt(s, "macro_f1")
        else:
            row["QWK"] = _pt(s, "qwk")
            row["Macro F1"] = _pt(s, "macro_f1")
            row["Bal. Acc"] = _pt(s, "balanced_acc")

    row["_task_ord"] = TASK_ORDER.get(task, 99)
    row["_mode_ord"] = MODE_ORDER.get(mode, 99)
    return row


def _aggregate_seed_rows(records: list[dict]) -> dict:
    """Aggregate multiple per-seed summaries into one row (mean ± std over seeds)."""
    seeds = sorted({r["seed"] for r in records if r.get("seed") is not None})
    proto = records[0]["summary"].get("protocol", "unknown")
    task_type = records[0]["summary"].get("task_type", "unknown")

    metrics_spec = (
        _BINARY_CV_METRICS
        if (proto == "grouped_cv" and task_type == "binary")
        else _BINARY_PT_METRICS
        if (proto != "grouped_cv" and task_type == "binary")
        else _MULTI_METRICS
    )

    row: dict = {
        "Sweep": records[0]["sweep_dir"],
        "Task": records[0]["task"],
        "Protocol": proto,
        "Mode": records[0]["mode"],
        "Seeds": len(records),
    }

    if "label_fraction" in records[0]["summary"]:
        row["Fraction"] = _fraction_label(records[0]["summary"]["label_fraction"])
        row["_frac_ord"] = float(records[0]["summary"]["label_fraction"])
    else:
        row["_frac_ord"] = 1.0

    for header, key, ci_prefix in metrics_spec:
        per_seed_vals = [_summary_per_seed_value(r["summary"], key, proto) for r in records]
        per_seed_vals = [v for v in per_seed_vals if v is not None]
        if not per_seed_vals:
            row[header] = ""
            continue
        m = sum(per_seed_vals) / len(per_seed_vals)
        if len(per_seed_vals) > 1:
            var = sum((v - m) ** 2 for v in per_seed_vals) / (len(per_seed_vals) - 1)
            sd = math.sqrt(var)
        else:
            sd = 0.0
        cell = _fmt_mean_std(m, sd)

        # Show OOF/test bootstrap CI from seed=42 (or first seed) for context, when present.
        if ci_prefix is not None:
            ref = sorted(records, key=lambda r: r.get("seed") or 0)[0]["summary"]
            ci = _ci(ref, ci_prefix)
            if ci:
                cell = f"{cell} {ci}".strip()
        row[header] = cell

    row["_task_ord"] = TASK_ORDER.get(records[0]["task"], 99)
    row["_mode_ord"] = MODE_ORDER.get(records[0]["mode"], 99)
    if seeds:
        row["_seed_label"] = ",".join(str(s) for s in seeds)
    return row


def collect(root: Path, *, by_seed: bool) -> list[dict]:
    raw: list[dict] = []
    for p in sorted(root.rglob("summary.json")):
        with open(p) as f:
            try:
                s = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed summary: {p} ({e})", file=sys.stderr)
                continue

        rel = p.relative_to(root)
        layout = _resolve_layout(rel.parts)
        if layout is None:
            continue
        outer_dir, cache_tag, sweep_dir, seed = layout

        protocol = s.get("protocol", "unknown")
        task, mode = _classify(outer_dir, cache_tag, protocol)

        raw.append(
            {
                "summary": s,
                "task": task,
                "mode": mode,
                "sweep_dir": sweep_dir,
                "seed": seed,
                "path": str(p),
            }
        )

    if by_seed:
        rows: list[dict] = []
        for rec in raw:
            row = _row_from_single_summary(
                rec["summary"], rec["task"], rec["mode"], rec["sweep_dir"]
            )
            if rec["seed"] is not None:
                row["Seed"] = rec["seed"]
            row["_path"] = rec["path"]
            rows.append(row)
    else:
        # Group by (sweep, task, mode, fraction). Within each group, aggregate
        # across seeds. Records without a seed pass through as a single group.
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for rec in raw:
            frac = rec["summary"].get("label_fraction", 1.0)
            try:
                frac_key = round(float(frac), 6)
            except (TypeError, ValueError):
                frac_key = 1.0
            key = (rec["sweep_dir"], rec["task"], rec["mode"], frac_key)
            groups[key].append(rec)

        rows = []
        for _, recs in groups.items():
            seeded = [r for r in recs if r["seed"] is not None]
            unseeded = [r for r in recs if r["seed"] is None]

            # Single-seed (or seedless) groups → existing single-summary formatting.
            if len(recs) == 1 and not seeded:
                rec = recs[0]
                row = _row_from_single_summary(
                    rec["summary"], rec["task"], rec["mode"], rec["sweep_dir"]
                )
                row["_path"] = rec["path"]
                rows.append(row)
                continue

            if seeded:
                row = _aggregate_seed_rows(seeded)
                row["_path"] = ";".join(r["path"] for r in seeded)
                rows.append(row)
            for rec in unseeded:
                row = _row_from_single_summary(
                    rec["summary"], rec["task"], rec["mode"], rec["sweep_dir"]
                )
                row["_path"] = rec["path"]
                rows.append(row)

    rows.sort(
        key=lambda r: (
            r.get("Sweep", ""),
            r.get("_task_ord", 99),
            r.get("_frac_ord", 1.0),
            r.get("_mode_ord", 99),
            r.get("Seed", -1) if isinstance(r.get("Seed"), int) else -1,
        )
    )
    for r in rows:
        for k in ("_task_ord", "_mode_ord", "_frac_ord", "_seed_label"):
            r.pop(k, None)
    return rows


# ── Print ───────────────────────────────────────────────────────────────────


def _print_section(title: str, rows: list[dict], cols: list[str]) -> None:
    widths = {c: len(c) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(r.get(c, ""))))

    sep = "  "
    header = sep.join(c.ljust(widths[c]) for c in cols)
    rule = sep.join("-" * widths[c] for c in cols)

    print(title)
    print(rule)
    print(header)
    print(rule)

    prev_task = None
    for r in rows:
        task = r.get("Task", "")
        if prev_task is not None and task != prev_task:
            print(rule)
        prev_task = task
        print(sep.join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    print(rule)


def print_table(rows: list[dict], show_sweep: bool, by_seed: bool) -> None:
    if not rows:
        print("No results found.")
        return

    binary = [r for r in rows if "AUROC" in r]
    multi = [r for r in rows if "QWK" in r]
    show_fraction = any("Fraction" in r for r in rows)
    show_seeds = any("Seeds" in r for r in rows)
    show_seed_col = by_seed and any("Seed" in r for r in rows)

    base_cols_binary = ["Task", "Mode", "AUROC", "AUPRC", "Bal. Acc", "Macro F1"]
    base_cols_multi = ["Task", "Mode", "QWK", "Macro F1", "Bal. Acc"]
    if show_fraction:
        base_cols_binary = ["Task", "Fraction", "Mode", "AUROC", "AUPRC", "Bal. Acc", "Macro F1"]
        base_cols_multi = ["Task", "Fraction", "Mode", "QWK", "Macro F1", "Bal. Acc"]
    if show_seed_col:
        base_cols_binary.insert(base_cols_binary.index("Mode") + 1, "Seed")
        base_cols_multi.insert(base_cols_multi.index("Mode") + 1, "Seed")
    elif show_seeds:
        base_cols_binary.insert(base_cols_binary.index("Mode") + 1, "Seeds")
        base_cols_multi.insert(base_cols_multi.index("Mode") + 1, "Seeds")
    if show_sweep:
        base_cols_binary = ["Sweep", *base_cols_binary]
        base_cols_multi = ["Sweep", *base_cols_multi]

    if binary:
        _print_section("Binary classification tasks", binary, base_cols_binary)
    if multi:
        if binary:
            print()
        _print_section("Multiclass / ordinal tasks", multi, base_cols_multi)


# ── Root resolution ─────────────────────────────────────────────────────────

DEFAULT_SWEEP_ROOT = Path("outputs/lp_2048")
LEGACY_ROOTS = (
    Path("/mnt/d/YC.Liu/outputs/linear_probe"),
    Path("outputs/linear_probe"),
)


def _resolve_root(arg_root: str | None, latest: bool) -> Path:
    if arg_root:
        root = Path(arg_root)
        if latest:
            root = _pick_latest_subdir(root)
        return root

    for candidate in (DEFAULT_SWEEP_ROOT, *LEGACY_ROOTS):
        if candidate.is_dir():
            return _pick_latest_subdir(candidate) if latest else candidate

    return DEFAULT_SWEEP_ROOT


def _pick_latest_subdir(root: Path) -> Path:
    """Return the most recently modified subdirectory of *root* (or root itself)."""
    if not root.is_dir():
        return root
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if not subdirs:
        return root
    return max(subdirs, key=lambda p: p.stat().st_mtime)


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect linear-probe results.")
    ap.add_argument(
        "--root",
        default=None,
        help=(
            "Root directory to walk for summary.json files. "
            f"Default: {DEFAULT_SWEEP_ROOT} (falls back to {LEGACY_ROOTS[0]})."
        ),
    )
    ap.add_argument(
        "--latest",
        action="store_true",
        help="Pick the most recently modified subdirectory of --root (e.g. latest sweep).",
    )
    ap.add_argument(
        "--by-seed",
        action="store_true",
        help=(
            "Show one row per seed instead of aggregating. "
            "Default behavior aggregates across seed_<N> subdirs into mean ± std."
        ),
    )
    ap.add_argument("--csv", default=None, help="Save results to CSV.")
    args = ap.parse_args()

    root = _resolve_root(args.root, args.latest)
    if not root.is_dir():
        print(f"Directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Walking: {root}")

    rows = collect(root, by_seed=args.by_seed)
    if not rows:
        print(f"No summary.json files found under {root}", file=sys.stderr)
        sys.exit(1)

    sweeps = {r.get("Sweep", "") for r in rows}
    sweep_values = sweeps - {""}
    has_fraction = any("Fraction" in r for r in rows)
    show_sweep = len(sweep_values) > 1
    if has_fraction and sweep_values and all(str(s).startswith("frac_") for s in sweep_values):
        show_sweep = False

    print_table(rows, show_sweep=show_sweep, by_seed=args.by_seed)

    if args.csv:
        import csv as csv_mod

        csv_path = Path(args.csv)
        export_rows = [{k: v for k, v in r.items() if k != "_path"} for r in rows]
        all_keys = list(dict.fromkeys(k for r in export_rows for k in r))
        with open(csv_path, "w", newline="") as f:
            w = csv_mod.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            w.writerows(export_rows)
        print(f"\nCSV saved to {csv_path}")


if __name__ == "__main__":
    main()
