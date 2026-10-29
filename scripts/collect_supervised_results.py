#!/usr/bin/env python
"""
Aggregate the supervised baselines (scripts/run_supervised_baselines.sh) into the
README's results tables.

For every (task, initialisation) it reads the runs sup_<data>_<init>__s<seed> and
reports, per metric:
  - mean ± SD over seeds of the pooled out-of-fold metric (each seed's five
    held-out folds scored together, patient level);
  - a 95% CI from a patient-level bootstrap of the seed-averaged predictions;
  - for the external cohort (CPTAC-NSCLC), the same on the five fold models'
    averaged predictions;
  - the paired difference to the from-scratch Mamba-2 MIL baseline, with a
    paired patient bootstrap CI.

Binary tasks report AUROC; PANDA (ISUP 0-5) reports quadratic weighted kappa on
the argmax grade and macro one-vs-rest AUROC.

Usage:
    python scripts/collect_supervised_results.py --out results/supervised
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
    "tcga_nsclc": ("TCGA-NSCLC (LUAD vs LUSC)", "nsclc"),
    "TCGA_BRAF": ("TCGA colorectal BRAF mutation", "braf"),
    "panda": ("PANDA prostate ISUP grade", "panda"),
}
INITS = {
    "abmil__random": "ABMIL, from scratch",
    "mamba2mil__random": "Mamba-2 MIL, from scratch",
    "mamba2mil__vicreg": "Mamba-2 MIL, VICReg init",
    "mamba2mil__jepa": "Mamba-2 MIL, JEPA init",
    "mamba2mil__lejepa": "Mamba-2 MIL, LeJEPA init",
    "mamba2mil__lejepa_mc": "Mamba-2 MIL, LeJEPA-MC init",
}
BASELINE = "mamba2mil__random"
RUN_RE = re.compile(r"^sup_(?P<data>.+)_(?P<init>(abmil|mamba2mil)__[a-z_]+)__s(?P<seed>\d+)$")


def _prob_cols(df: pd.DataFrame) -> list[str]:
    return sorted((c for c in df.columns if c.startswith("prob_")), key=lambda c: int(c[5:]))


def _metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """probs: [n] positive-class probability (binary) or [n, K] (multiclass)."""
    if probs.ndim == 1:
        return {"auroc": roc_auc_score(labels, probs)}
    present = np.unique(labels)
    out = {"qwk": cohen_kappa_score(labels, probs.argmax(1), weights="quadratic")}
    if len(present) == probs.shape[1]:
        out["auroc_macro"] = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    return out


def _probs(df: pd.DataFrame) -> np.ndarray:
    cols = _prob_cols(df)
    if cols == ["prob_1"]:
        return df["prob_1"].to_numpy()
    return df[cols].to_numpy()


def _bootstrap(labels, probs, n_boot, seed, other=None):
    """Percentile CI of each metric (or of the paired difference to `other`)."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < len(np.unique(labels)):
            continue
        m = _metrics(labels[idx], probs[idx])
        if other is not None:
            ref = _metrics(labels[idx], other[idx])
            m = {k: m[k] - ref[k] for k in m if k in ref}
        for k, v in m.items():
            draws.setdefault(k, []).append(v)
    return {
        k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in draws.items()
    }


def _load_runs(root: Path) -> dict:
    runs: dict = {}
    for d in sorted(root.iterdir()):
        m = RUN_RE.match(d.name)
        if not m or not (d / "summary.json").is_file():
            continue
        oof = pd.read_csv(d / "patient_predictions_oof.csv")
        ext = sorted(d.glob("fold_*/patient_predictions_test_external.csv"))
        if ext:
            folds = [pd.read_csv(p).set_index("patient_id") for p in ext]
            cols = _prob_cols(folds[0])
            avg = folds[0][["label"]].copy()
            avg[cols] = sum(f[cols] for f in folds) / len(folds)
            ext_df = avg.reset_index()
        else:
            ext_df = None
        key = (m["data"], m["init"])
        runs.setdefault(key, {})[int(m["seed"])] = {"oof": oof, "external": ext_df, "dir": d}
    return runs


def _seed_average(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Per-patient mean of the seeds' probabilities (patients common to all seeds)."""
    cols = _prob_cols(frames[0])
    idx = frames[0].set_index("patient_id").index
    for f in frames[1:]:
        idx = idx.intersection(f.set_index("patient_id").index)
    base = frames[0].set_index("patient_id").loc[idx, ["label"]].copy()
    base[cols] = sum(f.set_index("patient_id").loc[idx, cols] for f in frames) / len(frames)
    return base.reset_index()


def collect(root: Path, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = _load_runs(root)
    per_seed, summary = [], []
    averaged: dict = {}
    for (data, init), seeds in sorted(runs.items()):
        for split in ("oof", "external"):
            frames = [s[split] for s in seeds.values() if s[split] is not None]
            if not frames:
                continue
            for seed, s in sorted(seeds.items()):
                if s[split] is None:
                    continue
                for metric, v in _metrics(s[split]["label"].to_numpy(), _probs(s[split])).items():
                    per_seed.append(
                        dict(data=data, init=init, split=split, seed=seed, metric=metric, value=v)
                    )
            averaged[(data, init, split)] = _seed_average(frames)

    ps = pd.DataFrame(per_seed)
    for (data, init, split), avg in sorted(averaged.items()):
        labels, probs = avg["label"].to_numpy(), _probs(avg)
        ci = _bootstrap(labels, probs, n_boot, seed=0)
        delta, delta_ci = {}, {}
        base = averaged.get((data, BASELINE, split))
        if base is not None and init != BASELINE:
            common = avg.merge(base, on="patient_id", suffixes=("", "_base"))
            lab = common["label"].to_numpy()
            mine = _probs(common[["patient_id", "label", *_prob_cols(avg)]])
            ref = common[[f"{c}_base" for c in _prob_cols(avg)]].to_numpy()
            ref = ref[:, 0] if ref.shape[1] == 1 else ref
            m_mine, m_ref = _metrics(lab, mine), _metrics(lab, ref)
            delta = {k: m_mine[k] - m_ref[k] for k in m_mine if k in m_ref}
            delta_ci = _bootstrap(lab, mine, n_boot, seed=1, other=ref)
        sub = ps[(ps.data == data) & (ps.init == init) & (ps.split == split)]
        for metric, vals in sub.groupby("metric")["value"]:
            summary.append(
                dict(
                    data=data,
                    init=init,
                    split=split,
                    metric=metric,
                    n_seeds=len(vals),
                    n_patients=len(avg),
                    mean=vals.mean(),
                    sd=vals.std(ddof=1) if len(vals) > 1 else float("nan"),
                    ci_low=ci.get(metric, (np.nan, np.nan))[0],
                    ci_high=ci.get(metric, (np.nan, np.nan))[1],
                    delta_vs_scratch_mamba2=delta.get(metric, np.nan),
                    delta_ci_low=delta_ci.get(metric, (np.nan, np.nan))[0],
                    delta_ci_high=delta_ci.get(metric, (np.nan, np.nan))[1],
                )
            )
    return ps, pd.DataFrame(summary)


def _fmt(r) -> str:
    sd = "" if np.isnan(r.sd) else f" ± {r.sd:.3f}"
    return f"{r['mean']:.3f}{sd} ({r.ci_low:.3f}–{r.ci_high:.3f})"


def _fmt_delta(r) -> str:
    if np.isnan(r.delta_vs_scratch_mamba2):
        return "—"
    return f"{r.delta_vs_scratch_mamba2:+.3f} ({r.delta_ci_low:+.3f} to {r.delta_ci_high:+.3f})"


def markdown(summary: pd.DataFrame) -> str:
    out = []
    columns = {
        "tcga_nsclc": [
            ("oof", "auroc", "TCGA-NSCLC 5-fold CV AUROC"),
            ("external", "auroc", "CPTAC-NSCLC external AUROC"),
        ],
        "TCGA_BRAF": [("oof", "auroc", "5-fold CV AUROC")],
        "panda": [("oof", "qwk", "5-fold CV QWK"), ("oof", "auroc_macro", "5-fold CV macro AUROC")],
    }
    for data, cols in columns.items():
        sub = summary[summary.data == data]
        if sub.empty:
            continue
        title, task_tag = TASKS[data]
        n_seeds = int(sub.n_seeds.max())
        out.append(f"#### {title}\n")
        header = (
            ["Model"] + [c[2] for c in cols] + [f"Δ vs Mamba-2 scratch ({cols[-1][2]})", "Command"]
        )
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join(" --- " for _ in header) + "|")
        for init, label in INITS.items():
            cells = []
            last = None
            for split, metric, _ in cols:
                row = sub[(sub.init == init) & (sub.split == split) & (sub.metric == metric)]
                cells.append(_fmt(row.iloc[0]) if len(row) else "—")
                last = row.iloc[0] if len(row) else None
            if all(c == "—" for c in cells):
                continue
            cmd = f"`bash scripts/run_supervised_baselines.sh --task {task_tag} --init {init}`"
            out.append(
                f"| {label} | "
                + " | ".join(cells)
                + f" | {_fmt_delta(last) if last is not None else '—'} | {cmd} |"
            )
        n_pat = int(sub[sub.split == "oof"].n_patients.max())
        out.append(
            f"\nMean ± SD over {n_seeds} seeds (95% patient-bootstrap CI of the seed-averaged predictions); n = {n_pat} patients.\n"
        )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    default_root = Path(os.environ.get("OCEANPATH_ROOT", ".")) / "outputs" / "supervised"
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--out", type=Path, default=Path("results/supervised"))
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    per_seed, summary = collect(args.root, args.n_boot)
    args.out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.out / "per_seed.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    md = markdown(summary)
    (args.out / "tables.md").write_text(md + "\n")
    runs = sorted(
        f"sup_{d}_{i}__s{s}"
        for d, i, s in per_seed[["data", "init", "seed"]].drop_duplicates().itertuples(index=False)
    )
    (args.out / "provenance.json").write_text(
        json.dumps({"n_boot": args.n_boot, "runs": runs}, indent=2)
    )
    print(md)


if __name__ == "__main__":
    main()
