#!/usr/bin/env python
"""
Time to get one training bag from disk: per-slide HDF5 files versus OceanPath's
memory-mapped feature store.

A training bag is 2,048 patches of a slide, chosen by the spatially stratified sampler
that training uses, returned as a float32 tensor. The benchmark copies a fixed sample of
slides into one working directory, so every format is read from the same disk, builds an
OceanPath store from those copies, and times each reader twice:

  cold  one pass right after the reader's files are evicted from the page cache
        (posix_fadvise DONTNEED, no root needed): the first epoch, or data larger than RAM
  warm  repeated passes once the files are cached: later epochs on a machine whose RAM
        holds the features

The readers:

  release h5        the UNI2-h release files as downloaded: float32 [1, N, D], one HDF5
                    chunk per slide
  release h5, rows  the same files, reading only the sampled rows (h5py fancy indexing)
  fp16 h5           the same features rewritten as contiguous float16 [N, D]
  mmap store        OceanPath's chunked float16 store, read through MmapDataset

Per-bag times come from one process; DataLoader throughput (warm, slides per second) uses
worker processes, as training does.

Usage:
    python scripts/benchmark_data_loading.py \
        --manifest manifests/public/tcga_nsclc_labels.csv \
        --features-root $OCEANPATH_ROOT/features/UNI2_h_features \
        --work /tmp/oceanpath-bench --out results/data_loading

Writes data_loading.csv, data_loading.md and environment.json under --out.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from oceanpath.data.dataset import MmapDataset
from oceanpath.data.mmap_builder import MmapBuildConfig, build_mmap

BAG = 2048


def _sample(manifest: Path, root: Path, n: int, seed: int) -> list[tuple[str, Path]]:
    df = pd.read_csv(manifest)
    rows = df.sample(n=min(n, len(df)), random_state=seed)
    return [(r.slide_id, root / f"{r.slide_path}.h5") for r in rows.itertuples()]


def _prepare(slides: list[tuple[str, Path]], work: Path) -> tuple[Path, Path, Path]:
    release, fp16, store = work / "release", work / "fp16", work / "store"
    release.mkdir(parents=True, exist_ok=True)
    fp16.mkdir(parents=True, exist_ok=True)
    for slide_id, src in slides:
        dst = release / f"{slide_id}.h5"
        if not dst.exists():
            shutil.copyfile(src, dst)
        half = fp16 / f"{slide_id}.h5"
        if not half.exists():
            with h5py.File(dst, "r") as f, h5py.File(half, "w") as g:
                g["features"] = f["features"][0].astype(np.float16)
                g["coords"] = f["coords"][0]
    if not (store / "index_arrays.npz").exists():
        build_mmap(
            MmapBuildConfig(
                h5_dir=str(release),
                output_dir=str(store),
                feat_precision=16,
                save_coverage_thumbnails=False,
            )
        )
    return release, fp16, store


def _evict(files: list[Path]) -> None:
    """Drop the files' clean pages from the page cache."""
    for path in files:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def _store_bags(store: Path, slide_ids: list[str]) -> MmapDataset:
    return MmapDataset(
        str(store),
        slide_ids,
        labels=dict.fromkeys(slide_ids, 0),
        max_instances=BAG,
        is_train=True,
        cap_strategy="spatial_stratified",
        return_coords=False,
        force_float32=True,
    )


class _H5Bags(Dataset):
    """Training bags from per-slide HDF5 files, sampled exactly as MmapDataset samples."""

    def __init__(self, folder: Path, slide_ids: list[str], sampler: MmapDataset, rows: bool):
        self.paths = [folder / f"{s}.h5" for s in slide_ids]
        self.sampler = sampler
        self.rows = rows

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with h5py.File(self.paths[idx], "r") as f:
            feats, coords = f["features"], f["coords"]
            leading = feats.ndim == 3
            xy = coords[0] if leading else coords[:]
            n = xy.shape[0]
            pick = (
                np.sort(self.sampler._spatial_stratified_indices(xy, n, BAG, idx))
                if n > BAG
                else np.arange(n)
            )
            if self.rows:
                x = feats[0, pick] if leading else feats[pick]
            else:
                x = (feats[0] if leading else feats[:])[pick]
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def _pass_ms(dataset: Dataset) -> list[float]:
    times = []
    for i in range(len(dataset)):
        start = time.perf_counter()
        dataset[i]
        times.append((time.perf_counter() - start) * 1000)
    return times


def _loader_rate(dataset: Dataset, workers: int, epochs: int) -> float:
    loader = DataLoader(
        dataset, batch_size=None, shuffle=True, num_workers=workers, persistent_workers=True
    )
    for _ in loader:  # starts the workers
        pass
    start = time.perf_counter()
    for _ in range(epochs):
        for _ in loader:
            pass
    return epochs * len(dataset) / (time.perf_counter() - start)


def _md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(" --- " for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True, help="scratch directory on one disk")
    parser.add_argument("--out", type=Path, default=Path("results/data_loading"))
    parser.add_argument("--n-slides", type=int, default=100)
    parser.add_argument("--passes", type=int, default=3, help="timed warm passes")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(1)
    slides = _sample(args.manifest, args.features_root, args.n_slides, args.seed)
    release, fp16, store = _prepare(slides, args.work)
    slide_ids = [s for s, _ in slides]
    sampler = _store_bags(store, slide_ids)
    store_files = sorted(store.glob("*.bin"))
    # Each reader is built after its files leave the page cache, so the cold pass reads disk.
    readers: dict[str, tuple[list[Path], Callable[[], Dataset]]] = {
        "release h5": (
            [release / f"{s}.h5" for s in slide_ids],
            lambda: _H5Bags(release, slide_ids, sampler, rows=False),
        ),
        "release h5, rows": (
            [release / f"{s}.h5" for s in slide_ids],
            lambda: _H5Bags(release, slide_ids, sampler, rows=True),
        ),
        "fp16 h5": (
            [fp16 / f"{s}.h5" for s in slide_ids],
            lambda: _H5Bags(fp16, slide_ids, sampler, rows=False),
        ),
        "mmap store": (store_files, lambda: _store_bags(store, slide_ids)),
    }
    rows = []
    for name, (files, make) in readers.items():
        gc.collect()
        _evict(files)
        dataset = make()
        cold = np.asarray(_pass_ms(dataset))
        _pass_ms(dataset)  # warm the cache
        warm = np.asarray([t for _ in range(args.passes) for t in _pass_ms(dataset)])
        rate = _loader_rate(dataset, args.workers, epochs=2) if name != "release h5, rows" else None
        rows.append(
            {
                "reader": name,
                "cold_p50_ms": float(np.median(cold)),
                "cold_mean_ms": float(cold.mean()),
                "warm_p50_ms": float(np.median(warm)),
                "warm_p95_ms": float(np.percentile(warm, 95)),
                f"warm_loader_slides_per_s_{args.workers}_workers": rate,
            }
        )
        print(rows[-1], flush=True)
        del dataset

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "data_loading.csv", index=False)
    base = df.set_index("reader").loc["release h5"]
    rate_col = f"warm_loader_slides_per_s_{args.workers}_workers"
    pretty = pd.DataFrame(
        {
            "Reader": df.reader,
            "Cold, ms per bag (p50)": [
                f"{v:.0f} ({base.cold_p50_ms / v:.1f}×)" for v in df.cold_p50_ms
            ],
            "Warm, ms per bag (p50 / p95)": [
                f"{a:.1f} / {b:.1f}" for a, b in zip(df.warm_p50_ms, df.warm_p95_ms, strict=True)
            ],
            f"Warm DataLoader, {args.workers} workers": [
                "—" if pd.isna(v) else f"{v:,.0f} slides/s" for v in df[rate_col]
            ],
        }
    )
    patches = [int(n) for n in sampler.lengths]
    note = (
        f"{len(slide_ids)} TCGA-NSCLC slides (median {int(np.median(patches)):,} patches), "
        f"{BAG:,}-patch spatially stratified bags as float32; cold = one pass after evicting "
        f"the reader's files from the page cache, warm = {args.passes} passes on cached files. "
        "Speed-ups are against the release files."
    )
    (args.out / "data_loading.md").write_text(_md_table(pretty) + "\n\n" + note + "\n")
    env = {
        "cpu": platform.processor() or platform.machine(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "h5py": h5py.__version__,
        "numpy": np.__version__,
        "n_slides": len(slide_ids),
        "median_patches": int(np.median(patches)),
        "bag": BAG,
        "warm_passes": args.passes,
        "workers": args.workers,
    }
    (args.out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    print((args.out / "data_loading.md").read_text())


if __name__ == "__main__":
    main()
