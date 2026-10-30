#!/usr/bin/env python
"""
Aggregate the supervised baselines (scripts/run_supervised_baselines.sh) into the
README's results tables.

For every (task, initialisation) it reads the runs sup_<data>_<init>__s<seed> and
reports, per metric:
  - mean ± SD over seeds of the pooled out-of-fold metric (each seed's five
    held-out folds scored together, patient level);
  - a 95% CI of that seed mean from a patient-level bootstrap (each resample
    scores every seed and averages them);
  - for the external cohort (CPTAC-NSCLC), the same with each seed's five fold
    models' predictions averaged;
  - the difference of seed means to the from-scratch Mamba-2 MIL baseline, with
    a paired patient bootstrap CI (both models scored on the same resamples);
  - for reference, the metric of the seed ensemble (seed-averaged predictions).

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


def _seed_mean(labels, seed_probs, idx=None) -> dict[str, float]:
    """Mean over seeds of each metric, on every patient or on the resample `idx`."""
    if idx is not None:
        labels, seed_probs = labels[idx], [p[idx] for p in seed_probs]
    per_seed = [_metrics(labels, p) for p in seed_probs]
    keys = set.intersection(*(set(m) for m in per_seed))
    return {k: float(np.mean([m[k] for m in per_seed])) for k in keys}


def _bootstrap(labels, seed_probs, n_boot, seed, other=None):
    """Percentile CI of the seed-mean metrics (or of their difference to `other`'s),
    resampling patients; `seed_probs` and `other` hold one array per seed."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < len(np.unique(labels)):
            continue
        m = _seed_mean(labels, seed_probs, idx)
        if other is not None:
            ref = _seed_mean(labels, other, idx)
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


def _common(*frame_lists: list[pd.DataFrame]) -> pd.Index:
    """Patients present in every frame."""
    frames = [f for fl in frame_lists for f in fl]
    idx = frames[0].set_index("patient_id").index
    for f in frames[1:]:
        idx = idx.intersection(f.set_index("patient_id").index)
    return idx


def _aligned(frames: list[pd.DataFrame], patients: pd.Index) -> tuple[np.ndarray, list]:
    """Labels and one probability array per seed, in the order of `patients`."""
    rows = [f.set_index("patient_id").loc[patients] for f in frames]
    return rows[0]["label"].to_numpy(), [_probs(r) for r in rows]


def collect(root: Path, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = _load_runs(root)
    per_seed, summary = [], []
    by_split: dict = {}
    for (data, init), seeds in sorted(runs.items()):
        for split in ("oof", "external"):
            frames = [s[split] for _, s in sorted(seeds.items()) if s[split] is not None]
            if not frames:
                continue
            for seed, s in sorted(seeds.items()):
                if s[split] is None:
                    continue
                for metric, v in _metrics(s[split]["label"].to_numpy(), _probs(s[split])).items():
                    per_seed.append(
                        dict(data=data, init=init, split=split, seed=seed, metric=metric, value=v)
                    )
            by_split[(data, init, split)] = frames

    ps = pd.DataFrame(per_seed)
    for (data, init, split), frames in sorted(by_split.items()):
        patients = _common(frames)
        labels, probs = _aligned(frames, patients)
        ci = _bootstrap(labels, probs, n_boot, seed=0)
        ensemble = _metrics(labels, np.mean(probs, axis=0))
        delta, delta_ci = {}, {}
        base = by_split.get((data, BASELINE, split))
        if base is not None and init != BASELINE:
            shared = _common(frames, base)
            lab, mine = _aligned(frames, shared)
            _, ref = _aligned(base, shared)
            m_mine, m_ref = _seed_mean(lab, mine), _seed_mean(lab, ref)
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
                    n_patients=len(patients),
                    mean=vals.mean(),
                    sd=vals.std(ddof=1) if len(vals) > 1 else float("nan"),
                    ci_low=ci.get(metric, (np.nan, np.nan))[0],
                    ci_high=ci.get(metric, (np.nan, np.nan))[1],
                    delta_vs_scratch_mamba2=delta.get(metric, np.nan),
                    delta_ci_low=delta_ci.get(metric, (np.nan, np.nan))[0],
                    delta_ci_high=delta_ci.get(metric, (np.nan, np.nan))[1],
                    seed_ensemble=ensemble.get(metric, np.nan),
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
            f"\nMean ± SD over {n_seeds} seeds (95% patient-bootstrap CI of the seed mean); "
            f"Δ: difference of seed means, paired patient bootstrap; n = {n_pat} patients.\n"
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
