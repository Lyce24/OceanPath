#!/usr/bin/env python
"""
Account for the GPU time spent on SSL pretraining, from the Weights & Biases run records.

Every logged pretraining run (development runs, ablations and failures included) becomes
one row: objective, aggregator, manifest, site, GPU type and count, wall-clock hours and
GPU-hours (wall hours x GPUs). The runs whose checkpoints the downstream experiments use
are marked, and a short summary table is written beside the CSV.

Needs read access to the W&B projects (wandb login). Hostnames are reduced to a site
name so no machine name reaches the results.

Usage:
    python scripts/pretraining_compute.py --out results/pretraining
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECTS = ["oceanpath-pretrain", "oceanpath-pretrain-v2"]
# W&B run id behind each checkpoint the downstream experiments load.
RELEASED = {"hfutapft": "vicreg", "iedwjz6n": "jepa", "q7avi2sx": "lejepa", "jj0h2ske": "lejepa_mc"}


def _value(config: dict, key: str) -> dict:
    v = config.get(key) or {}
    return v.get("value", v) if isinstance(v, dict) else {}


def collect(entity: str | None) -> pd.DataFrame:
    import wandb

    api = wandb.Api(timeout=60)
    entity = entity or api.default_entity
    rows = []
    for project in PROJECTS:
        for run in api.runs(f"{entity}/{project}"):
            meta = run.metadata or {}
            host = meta.get("host") or ""
            gpus = int(meta.get("gpu_count") or 1)
            hours = float(run.summary.get("_runtime") or 0) / 3600
            data = _value(run.config, "data")
            method = _value(run.config, "pretrain_training").get("ssl_method")
            epoch = run.summary.get("epoch")
            rows.append(
                {
                    "run_id": run.id,
                    "created": run.created_at[:10],
                    "state": run.state,
                    "objective": method,
                    "aggregator": run.name.split("_uni2h_")[-1].rsplit("_", 1)[0]
                    if "_uni2h_" in run.name
                    else run.name,
                    "manifest": Path(data.get("csv_path") or "").name,
                    "site": "Brown CCV Oscar" if ".oscar" in host else "workstation",
                    "gpu": meta.get("gpu"),
                    "n_gpus": gpus,
                    "hours": round(hours, 3),
                    "gpu_hours": round(hours * gpus, 3),
                    "epochs": None if epoch is None else int(epoch) + 1,
                    "released_checkpoint": RELEASED.get(run.id, ""),
                }
            )
    return pd.DataFrame(rows).sort_values("created")


def summary(df: pd.DataFrame) -> str:
    rel = df[df.released_checkpoint != ""].set_index("released_checkpoint")
    lines = [
        "| Objective | Hardware | Epochs | Wall-clock | GPU-hours |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in ["vicreg", "jepa", "lejepa", "lejepa_mc"]:
        r = rel.loc[name]
        lines.append(
            f"| {name} | {r.n_gpus}x {r.gpu} ({r.site}) | {int(r.epochs)} | {r.hours:.1f} h | {r.gpu_hours:.1f} |"
        )
    lines.append(f"| **The four checkpoints** | | | | **{rel.gpu_hours.sum():.1f}** |")
    by_site = df.groupby("site").gpu_hours.sum()
    sites = ", ".join(f"{s} {h:.0f}" for s, h in by_site.items())
    lines.append(
        f"| All {len(df)} logged pretraining runs (development, ablations, failures) | | | | "
        f"**{df.gpu_hours.sum():.0f}** ({sites}) |"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--entity", default=None, help="W&B entity (default: the logged-in one)")
    parser.add_argument("--out", type=Path, default=Path("results/pretraining"))
    args = parser.parse_args()
    df = collect(args.entity)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "compute.csv", index=False)
    md = summary(df)
    (args.out / "compute.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
