#!/usr/bin/env python
"""
GPU memory and throughput of the MIL aggregators versus bag size.

For each aggregator in configs/model/ and each bag size N, one slide of N random
patch features (dimension D, UNI2-h: 1536) goes through either a training step
(forward, backward and AdamW update) or an inference forward pass, under bf16
autocast as in training. Reports peak allocated GPU memory and throughput
(patches/s); a size that runs out of memory is recorded as OOM and the
aggregator's larger sizes are skipped.

Linear-time aggregators (attention pooling, Perceiver, Mamba-2) should scale
linearly in N; the TITAN-style ViT builds an N x N ALiBi distance matrix and
TransMIL's Nystrom attention has its own landmark cost.

Usage:
    python scripts/benchmark_scaling.py --out results/scaling
    python scripts/benchmark_scaling.py --archs abmil mamba2mil --bag-sizes 1024 16384

Writes scaling.csv, environment.json, scaling.md and scaling.png under --out.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from oceanpath.models import build_classifier, list_aggregators

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "model"
DEFAULT_ARCHS = [
    "static",
    "abmil",
    "mhabmil",
    "perceiver",
    "transmil",
    "vit",
    "mamba2mil",
    "bimamba",
]
DEFAULT_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]


def _model(arch: str, in_dim: int) -> torch.nn.Module:
    cfg = OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / f"{arch}.yaml"))
    return build_classifier(arch=cfg["arch"], in_dim=in_dim, num_classes=2, model_cfg=cfg).cuda()


def _measure(arch: str, n: int, mode: str, in_dim: int, iters: int, warmup: int) -> dict:
    model = _model(arch, in_dim)
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, n, in_dim, device="cuda")
    mask = torch.ones(1, n, device="cuda")
    # Patch-grid coordinates (256 px patches at 20x) for the coordinate-aware models.
    side = int(np.ceil(np.sqrt(n)))
    grid = torch.stack(torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij"), -1)
    coords = (grid.reshape(-1, 2)[:n] * 256).float().unsqueeze(0).cuda()
    y = torch.ones(1, dtype=torch.long, device="cuda")

    if mode == "train":
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, mask=mask, coords=coords)
            F.cross_entropy(out.logits.float(), y).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
    else:
        model.eval()

        @torch.no_grad()
        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x, mask=mask, coords=coords)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    ms = float(np.median(times))
    return {
        "arch": arch,
        "mode": mode,
        "n_patches": n,
        "status": "ok",
        "median_ms": ms,
        "patches_per_s": n / (ms / 1000),
        "slides_per_s": 1000 / ms,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
        "params_m": n_params / 1e6,
    }


def _release() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _run_series(arch: str, mode: str, args: argparse.Namespace) -> None:
    """Child process: one aggregator and mode over increasing N, one JSON line per size.

    The PyTorch allocator is capped below the card's memory so an oversized bag
    raises a clean OutOfMemoryError; under WSL2 the driver can otherwise fall
    back to shared system memory and fail with an unrecoverable CUDA error.
    """
    torch.cuda.set_per_process_memory_fraction(0.95)
    for n in args.bag_sizes:
        try:
            row = _measure(arch, n, mode, args.in_dim, args.iters, args.warmup)
        except torch.OutOfMemoryError:
            row = {"arch": arch, "mode": mode, "n_patches": n, "status": "oom"}
        finally:
            _release()
        print(json.dumps(row), flush=True)
        if row["status"] == "oom":
            return


def run(args: argparse.Namespace) -> pd.DataFrame:
    """Each (aggregator, mode) series runs in its own process, so a CUDA failure
    ends only that series."""
    available = set(list_aggregators())
    rows = []
    for arch in args.archs:
        if arch not in available:
            print(f"skip {arch}: not registered (mamba-ssm missing?)")
            continue
        for mode in args.modes:
            cmd = [
                sys.executable,
                __file__,
                "--child",
                arch,
                mode,
                "--in-dim",
                str(args.in_dim),
                "--iters",
                str(args.iters),
                "--warmup",
                str(args.warmup),
                "--bag-sizes",
                *map(str, args.bag_sizes),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            series = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
            if proc.returncode != 0:
                done = {r["n_patches"] for r in series}
                failed = next((n for n in args.bag_sizes if n not in done), None)
                if failed is not None:
                    series.append(
                        {"arch": arch, "mode": mode, "n_patches": failed, "status": "error"}
                    )
            for row in series:
                if row["status"] == "ok":
                    print(
                        f"{arch:10s} {mode:5s} N={row['n_patches']:7d} {row['median_ms']:9.2f} ms "
                        f"{row['patches_per_s'] / 1e3:9.1f} k patch/s {row['peak_mem_gb']:6.2f} GB"
                    )
                else:
                    print(f"{arch:10s} {mode:5s} N={row['n_patches']:7d} {row['status'].upper()}")
            rows.extend(series)
    return pd.DataFrame(rows)


def _environment() -> dict:
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": props.name,
        "gpu_memory_gb": round(props.total_memory / 2**30, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "precision": "bf16 autocast",
        "batch": "1 slide per step",
    }


def _md_table(df: pd.DataFrame) -> str:
    """A pipe table without pandas' optional tabulate dependency."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(" --- " for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _markdown(df: pd.DataFrame) -> str:
    ok = df[df.status == "ok"]
    out = []
    for mode, title in [("train", "Training step"), ("infer", "Inference")]:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue

        def cell(r):
            if r.status in ("oom", "error"):
                return "OOM" if r.status == "oom" else "CUDA error"
            return f"{r.peak_mem_gb:.2f} GB · {r.patches_per_s / 1e3:,.0f}k/s"

        table = (
            sub.assign(cell=sub.apply(cell, axis=1))
            .pivot_table(index="arch", columns="n_patches", values="cell", aggfunc="first")
            .reindex([a for a in DEFAULT_ARCHS if a in set(sub.arch)])
            .fillna("—")
        )
        table.columns = [f"N = {c:,}" for c in table.columns]
        out.append(f"#### {title} (peak memory · patches per second)\n")
        out.append(_md_table(table.reset_index().rename(columns={"arch": "Aggregator"})))
        out.append("")
    del ok
    return "\n".join(out)


def _plot(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = df[(df.status == "ok") & (df["mode"] == "train")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {"mamba2mil": ("#c2410c", "-"), "bimamba": ("#ea580c", "--")}
    for arch, sub in ok.groupby("arch"):
        color, ls = styles.get(arch, (None, "-"))
        kw = {"label": arch, "marker": "o", "ms": 3.5, "lw": 1.6, "ls": ls}
        if color:
            kw["color"] = color
        axes[0].plot(sub.n_patches, sub.peak_mem_gb, **kw)
        axes[1].plot(sub.n_patches, sub.patches_per_s / 1e3, **kw)
    for ax, ylabel in zip(axes, ["Peak GPU memory (GB)", "Throughput (k patches/s)"], strict=True):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Patches per slide (N)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Training step, one slide, bf16, RTX A5000 (24 GB)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("results/scaling"))
    parser.add_argument("--archs", nargs="+", default=DEFAULT_ARCHS)
    parser.add_argument("--bag-sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument(
        "--modes", nargs="+", default=["train", "infer"], choices=["train", "infer"]
    )
    parser.add_argument("--in-dim", type=int, default=1536)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--child", nargs=2, metavar=("ARCH", "MODE"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    torch.manual_seed(0)
    if args.child:
        _run_series(*args.child, args)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    df = run(args)
    df.to_csv(args.out / "scaling.csv", index=False)
    (args.out / "environment.json").write_text(json.dumps(_environment(), indent=2))
    (args.out / "scaling.md").write_text(_markdown(df) + "\n")
    _plot(df, args.out / "scaling.png")
    print(_markdown(df))


if __name__ == "__main__":
    main()
