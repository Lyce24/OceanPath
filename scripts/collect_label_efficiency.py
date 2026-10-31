#!/usr/bin/env python
"""
Aggregate a label-efficiency sweep (scripts/run_label_efficiency.sh) into tables and
curves.

Every run is one (task, mode, labeled fraction, seed): a linear probe trained on the
seed's labeled subsample of the training patients and scored, pooled over the outer
folds (grouped CV) or on the full external cohort (external test). The seed fixes
which patients are labeled, so at a given fraction and seed every mode sees the same
labels and their difference is paired.

For each (task, mode, fraction) this reports the mean and SD over seeds of the
primary metric (AUROC for binary tasks, quadratic weighted kappa for PANDA), and the
paired difference to the random-init Mamba-2 aggregator.

Usage:
    python scripts/collect_label_efficiency.py --root $OCEANPATH_ROOT/outputs/label_efficiency \
        --out results/label_efficiency
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

TASKS = {
    "nsclc_cv": "TCGA-NSCLC, 5-fold CV (AUROC)",
    "nsclc_xfer": "TCGA-NSCLC → CPTAC-NSCLC (AUROC)",
    "braf": "TCGA colorectal BRAF, 5-fold CV (AUROC)",
    "panda": "PANDA ISUP grade, 5-fold CV (QWK)",
}
MODES = {
    "mean_pool": "Mean pool",
    "max_pool": "Max pool",
    "mean_max_pool": "Mean + max pool",
    "aggregator_random": "Mamba-2, random init",
    "ssl_vicreg": "Mamba-2, VICReg",
    "ssl_jepa": "Mamba-2, JEPA",
    "ssl_lejepa": "Mamba-2, LeJEPA",
    "ssl_lejepa_mc": "Mamba-2, LeJEPA-MC",
    # VICReg's last epoch, chosen without labels (the released checkpoints were chosen
    # by a probe score; see the README).
    "ssl_vicreg_last": "Mamba-2, VICReg (last epoch)",
}
BASELINE = "aggregator_random"
RUN_RE = re.compile(r"frac_[^/]+/(?P<task>[a-z_]+)__(?P<mode>[a-z_]+)/seed_(?P<seed>\d+)/[^/]+$")


def _metric(df: pd.DataFrame, task_type: str) -> float:
    probs = [c for c in df.columns if c.startswith("prob_")]
    if task_type == "binary":
        return float(roc_auc_score(df["label"], df["prob_1"]))
    pred = df[sorted(probs, key=lambda c: int(c[5:]))].to_numpy().argmax(1)
    return float(cohen_kappa_score(df["label"], pred, weights="quadratic"))


def collect_runs(root: Path) -> pd.DataFrame:
    """One row per finished run; a rerun of the same cell keeps the newest."""
    rows = []
    for summary in root.rglob("summary.json"):
        m = RUN_RE.search(str(summary.parent))
        if not m:
            continue
        s = json.loads(summary.read_text())
        preds_name = (
            "patient_predictions_test.csv"
            if s.get("protocol") == "external_test"
            else "patient_predictions_oof.csv"
        )
        preds = summary.parent / preds_name
        if not preds.is_file():
            continue
        rows.append(
            {
                "task": m["task"],
                "mode": m["mode"],
                "fraction": float(s["label_fraction"]),
                "seed": int(m["seed"]),
                "value": _metric(pd.read_csv(preds), s["task_type"]),
                "mtime": summary.stat().st_mtime,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("mtime").drop_duplicates(["task", "mode", "fraction", "seed"], keep="last")
    return df.drop(columns="mtime").sort_values(["task", "mode", "fraction", "seed"])


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    stats = (
        runs.groupby(["task", "mode", "fraction"])["value"]
        .agg(mean="mean", sd=lambda v: v.std(ddof=1) if len(v) > 1 else np.nan, n_seeds="count")
        .reset_index()
    )
    base = runs[runs["mode"] == BASELINE][["task", "fraction", "seed", "value"]]
    paired = runs.merge(base, on=["task", "fraction", "seed"], suffixes=("", "_base"))
    paired["delta"] = paired["value"] - paired["value_base"]
    deltas = (
        paired.groupby(["task", "mode", "fraction"])["delta"]
        .agg(delta_mean="mean", delta_sd=lambda v: v.std(ddof=1) if len(v) > 1 else np.nan)
        .reset_index()
    )
    return stats.merge(deltas, on=["task", "mode", "fraction"], how="left")


def _md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(" --- " for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


FIGURE = (
    '<p align="center">\n  <img src="results/label_efficiency/label_efficiency.png" width="100%" '
    'alt="Label-efficiency curves: linear-probe score against the share of labelled training '
    'patients, one panel per task" />\n</p>\n'
)


def markdown(summary: pd.DataFrame) -> str:
    out = [FIGURE]
    for task, title in TASKS.items():
        sub = summary[summary.task == task]
        if sub.empty:
            continue
        fractions = sorted(sub.fraction.unique())
        rows = []
        for mode, label in MODES.items():
            cells = []
            for f in fractions:
                r = sub[(sub["mode"] == mode) & (sub.fraction == f)]
                if r.empty:
                    cells.append("—")
                    continue
                r = r.iloc[0]
                sd = "" if np.isnan(r.sd) else f" ± {r.sd:.3f}"
                cells.append(f"{r['mean']:.3f}{sd}")
            if any(c != "—" for c in cells):
                rows.append([label, *cells])
        table = pd.DataFrame(rows, columns=["Embedding", *[f"{f:.0%}" for f in fractions]])
        out.append(f"#### {title}\n")
        out.append(_md_table(table))
        out.append("")
    return "\n".join(out)


def plot(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = [t for t in TASKS if t in set(summary.task)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.6), squeeze=False)
    styles = {
        "mean_pool": ("#9ca3af", ":"),
        "max_pool": ("#6b7280", ":"),
        "mean_max_pool": ("#374151", ":"),
        "aggregator_random": ("#111827", "--"),
        "ssl_vicreg": ("#2563eb", "-"),
        "ssl_jepa": ("#16a34a", "-"),
        "ssl_lejepa": ("#c2410c", "-"),
        "ssl_lejepa_mc": ("#9333ea", "-"),
        "ssl_vicreg_last": ("#2563eb", "-."),
    }
    for ax, task in zip(axes[0], tasks, strict=True):
        sub = summary[summary.task == task]
        for mode, label in MODES.items():
            m = sub[sub["mode"] == mode].sort_values("fraction")
            if m.empty:
                continue
            color, ls = styles[mode]
            ax.errorbar(
                m.fraction,
                m["mean"],
                yerr=m["sd"].fillna(0),
                label=label,
                color=color,
                ls=ls,
                marker="o",
                ms=3,
                lw=1.4,
                capsize=2,
            )
        ax.set_xscale("log")
        ax.set_xticks(sorted(sub.fraction.unique()))
        ax.set_xticklabels([f"{f:.0%}" for f in sorted(sub.fraction.unique())])
        ax.set_xlabel("Labeled training patients")
        ax.set_title(TASKS[task], fontsize=9)
        ax.grid(True, alpha=0.25)
    axes[0][0].set_ylabel("Linear probe score (mean ± SD over seeds)")
    axes[0][-1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    default_root = Path(os.environ.get("OCEANPATH_ROOT", ".")) / "outputs" / "label_efficiency"
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--out", type=Path, default=Path("results/label_efficiency"))
    args = parser.parse_args()

    runs = collect_runs(args.root)
    if runs.empty:
        raise SystemExit(f"no finished runs under {args.root}")
    summary = summarize(runs)
    args.out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.out / "runs.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    md = markdown(summary)
    (args.out / "tables.md").write_text(md + "\n")
    plot(summary, args.out / "label_efficiency.png")
    print(md)
    print(f"\n{len(runs)} runs")


if __name__ == "__main__":
    main()
