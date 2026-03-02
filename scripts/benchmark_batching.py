#!/usr/bin/env python
"""
Benchmark batching strategies for SSL pretraining.

Loads a real mmap dataset and benchmarks all 7 batching strategies
measuring:
  - Epoch iteration time (after warmup)
  - Padding ratio (mean / p90)
  - Batch size distribution (mean / std / min / max)
  - Token utilization per batch
  - Peak memory per batch

By default, augmentation is disabled (identity) so the benchmark
isolates batching behavior. Use --augmentation to enable the default
SSL augmentation pipeline.

Usage:
    # Default: no augmentation, real mmap data
    python scripts/benchmark_batching.py --mmap_dir /path/to/mmap

    # With augmentation enabled
    python scripts/benchmark_batching.py --mmap_dir /path/to/mmap --augmentation

    # Only specific strategies
    python scripts/benchmark_batching.py --mmap_dir /path/to/mmap \
        --strategies bucket_batching token_budget

    # Limit slides for a quick test
    python scripts/benchmark_batching.py --mmap_dir /path/to/mmap --max_slides 200

    # Custom output dir
    python scripts/benchmark_batching.py --mmap_dir /path/to/mmap \
        --output_dir outputs/bench
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

ALL_STRATEGIES = [
    "pad_to_global",
    "pad_to_max_in_batch",
    "token_budget",
    "bucket_batching",
    "subsample_fixed_n",
    "regional_crops",
    "sequence_packing",
    "multi_crop",
    "jepa",
]

# Identity augmentation: keeps all patches, no perturbation.
# Both views are identical length -> clean batching benchmark.
IDENTITY_AUG_CFG = {
    "subsample_frac": [1.0, 1.0],
    "instance_dropout": 0.0,
    "feature_noise_std": 0.0,
    "feature_dropout": 0.0,
}

# Default SSL augmentation for optional --augmentation mode.
DEFAULT_AUG_CFG = {
    "subsample_frac": [0.5, 1.0],
    "instance_dropout": 0.1,
    "feature_noise_std": 0.02,
    "feature_dropout": 0.05,
}


# ═════════════════════════════════════════════════════════════════════════════
# Measurement helpers
# ═════════════════════════════════════════════════════════════════════════════


def _measure_padding_ratio(batch: dict) -> float:
    """Compute fraction of padded (zero-mask) positions across both views."""
    total = batch["mask1"].numel() + batch["mask2"].numel()
    active = batch["mask1"].sum().item() + batch["mask2"].sum().item()
    return 1.0 - active / total if total > 0 else 0.0


def _measure_batch_memory_bytes(batch: dict) -> int:
    """Sum of tensor sizes in the batch dict."""
    return sum(
        v.nelement() * v.element_size() for v in batch.values() if isinstance(v, torch.Tensor)
    )


# ═════════════════════════════════════════════════════════════════════════════
# Dataset truncation helper
# ═════════════════════════════════════════════════════════════════════════════


def _truncate_dataset(ds, max_slides: int) -> None:
    """Truncate a PretrainDataset to the first ``max_slides`` slides in-place.

    Handles all parallel per-slide lists and the preloaded coord cache.
    """
    ds.slide_ids = ds.slide_ids[:max_slides]
    ds.lengths = ds.lengths[:max_slides]
    ds.feat_chunk_ids = ds.feat_chunk_ids[:max_slides]
    ds.feat_offsets = ds.feat_offsets[:max_slides]
    ds.coord_chunk_ids = ds.coord_chunk_ids[:max_slides]
    ds.coord_offsets = ds.coord_offsets[:max_slides]
    if ds._preloaded_coords is not None:
        ds._preloaded_coords = ds._preloaded_coords[:max_slides]


# ═════════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═════════════════════════════════════════════════════════════════════════════


def _benchmark_strategy(
    strategy: str,
    mmap_dir: str,
    batch_size: int,
    max_instances: int | None,
    dataset_max_instances: int | None,
    num_workers: int,
    n_warmup_batches: int,
    n_epochs: int,
    batching_cfg: dict,
    augmentation_cfg: dict,
    coords_aware: bool,
    max_slides: int | None,
    seed: int,
) -> dict:
    """Run benchmark for a single strategy, return metrics dict."""
    from oceanpath.data.pretrain_datamodule import PretrainDataModule

    cfg_overrides = dict(batching_cfg)
    cfg_overrides.pop("strategy", None)

    dm = PretrainDataModule(
        mmap_dir=mmap_dir,
        batch_size=batch_size,
        max_instances=max_instances,
        dataset_max_instances=dataset_max_instances,
        num_workers=num_workers,
        val_frac=0.1,
        prefetch_factor=4,
        seed=seed,
        coords_aware=coords_aware,
        augmentation_cfg=augmentation_cfg,
        batching_strategy=strategy,
        batching_cfg=cfg_overrides,
    )
    dm.setup()

    if max_slides is not None:
        n_total = len(dm.train_dataset)
        if n_total > max_slides:
            _truncate_dataset(dm.train_dataset, max_slides)
            logger.info("  Limited to %d/%d slides", max_slides, n_total)

    n_train = len(dm.train_dataset)
    bag_sizes = dm.train_dataset.get_bag_sizes()

    if n_train == 0:
        logger.warning("  No training slides found — skipping strategy.")
        return {"strategy": strategy, "error": "No training slides"}

    logger.info(
        "  Train slides: %d, bag sizes: mean=%.0f, median=%.0f, min=%d, max=%d",
        n_train,
        bag_sizes.mean(),
        np.median(bag_sizes),
        bag_sizes.min(),
        bag_sizes.max(),
    )

    # ── Warmup pass ──────────────────────────────────────────────────────
    loader = dm.train_dataloader()
    warmup_count = 0
    for _batch in loader:
        warmup_count += 1
        if warmup_count >= n_warmup_batches:
            break
    logger.info("  Warmup: %d batches", warmup_count)

    # ── Timed pass ───────────────────────────────────────────────────────
    padding_ratios = []
    batch_sizes = []
    token_utils = []
    peak_memories = []
    epoch_times = []

    for epoch in tqdm(range(n_epochs), desc="Epochs", unit="epoch"):
        loader = dm.train_dataloader()
        if hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch + 1)

        t0 = time.perf_counter()
        for batch in tqdm(loader, desc="Batches", unit="batch", leave=False):
            bs = batch["view1"].shape[0]
            batch_sizes.append(bs)
            padding_ratios.append(_measure_padding_ratio(batch))
            token_utils.append(batch["mask1"].sum().item() + batch["mask2"].sum().item())
            peak_memories.append(_measure_batch_memory_bytes(batch))
        epoch_times.append(time.perf_counter() - t0)

    if not batch_sizes:
        return {
            "strategy": strategy,
            "n_train_slides": n_train,
            "n_epochs": n_epochs,
            "n_batches_per_epoch": 0,
            "epoch_time_mean_s": 0.0,
            "epoch_time_std_s": 0.0,
            "padding_ratio_mean": 0.0,
            "padding_ratio_p50": 0.0,
            "padding_ratio_p90": 0.0,
            "batch_size_mean": 0.0,
            "batch_size_std": 0.0,
            "batch_size_min": 0,
            "batch_size_max": 0,
            "token_util_mean": 0.0,
            "token_util_total": 0.0,
            "peak_memory_mean_kb": 0.0,
            "peak_memory_max_kb": 0.0,
            "note": "No batches produced (drop_last may have dropped all)",
        }

    padding_arr = np.array(padding_ratios)
    batch_arr = np.array(batch_sizes)
    token_arr = np.array(token_utils)
    mem_arr = np.array(peak_memories)
    time_arr = np.array(epoch_times)

    return {
        "strategy": strategy,
        "n_train_slides": n_train,
        "n_epochs": n_epochs,
        "n_batches_per_epoch": len(batch_sizes) // n_epochs,
        "epoch_time_mean_s": round(float(time_arr.mean()), 3),
        "epoch_time_std_s": round(float(time_arr.std()), 3),
        "padding_ratio_mean": round(float(padding_arr.mean()), 4),
        "padding_ratio_p50": round(float(np.percentile(padding_arr, 50)), 4),
        "padding_ratio_p90": round(float(np.percentile(padding_arr, 90)), 4),
        "batch_size_mean": round(float(batch_arr.mean()), 2),
        "batch_size_std": round(float(batch_arr.std()), 2),
        "batch_size_min": int(batch_arr.min()),
        "batch_size_max": int(batch_arr.max()),
        "token_util_mean": round(float(token_arr.mean()), 1),
        "token_util_total": round(float(token_arr.sum()), 0),
        "peak_memory_mean_kb": round(float(mem_arr.mean()) / 1024, 1),
        "peak_memory_max_kb": round(float(mem_arr.max()) / 1024, 1),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Table formatting
# ═════════════════════════════════════════════════════════════════════════════


def _format_table(results: list[dict]) -> str:
    """Format results as a comparison table."""
    headers = [
        "Strategy",
        "Bat/Ep",
        "Time(s)",
        "PadMean",
        "PadP50",
        "PadP90",
        "BS_Mean",
        "BS_Std",
        "BS_Min",
        "BS_Max",
        "Tok_Mean",
        "Mem_Mean(KB)",
        "Mem_Max(KB)",
    ]
    rows = []
    for r in results:
        rows.append(
            [
                r["strategy"],
                str(r["n_batches_per_epoch"]),
                f"{r['epoch_time_mean_s']:.3f}",
                f"{r['padding_ratio_mean']:.4f}",
                f"{r['padding_ratio_p50']:.4f}",
                f"{r['padding_ratio_p90']:.4f}",
                f"{r['batch_size_mean']:.1f}",
                f"{r['batch_size_std']:.1f}",
                str(r["batch_size_min"]),
                str(r["batch_size_max"]),
                f"{r['token_util_mean']:.0f}",
                f"{r['peak_memory_mean_kb']:.1f}",
                f"{r['peak_memory_max_kb']:.1f}",
            ]
        )

    col_widths = [max(len(h), max(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_line = "|" + "|".join(f" {h:^{col_widths[i]}} " for i, h in enumerate(headers)) + "|"

    lines = [sep, header_line, sep]
    for row in rows:
        line = "|" + "|".join(f" {row[i]:>{col_widths[i]}} " for i in range(len(headers))) + "|"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark SSL pretraining batching strategies on real mmap data."
    )
    p.add_argument(
        "--mmap_dir",
        type=str,
        required=True,
        help="Path to mmap directory (built by build_mmap.py).",
    )
    p.add_argument(
        "--augmentation",
        action="store_true",
        default=False,
        help="Enable SSL augmentation (default: off = identity, no perturbation).",
    )
    p.add_argument(
        "--max_slides",
        type=int,
        default=None,
        help="Limit training slides for quick benchmarks (default: use all).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Nominal batch size (default: 128).",
    )
    p.add_argument(
        "--max_instances",
        type=int,
        default=4096,
        help="Max instances cap per view at collator level (default: 4096).",
    )
    p.add_argument(
        "--dataset_max_instances",
        type=int,
        default=8192,
        help="Dataset-level cap before augmentation (default: 8192).",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader workers (default: 8).",
    )
    p.add_argument(
        "--n_warmup_batches",
        type=int,
        default=10,
        help="Number of warmup batches before timing (default: 10).",
    )
    p.add_argument(
        "--n_epochs",
        type=int,
        default=3,
        help="Number of timed epochs per strategy (default: 3).",
    )
    p.add_argument(
        "--token_budget",
        type=int,
        default=128000,
        help="Token budget for token-budget strategies (default: 8192).",
    )
    p.add_argument(
        "--fixed_n",
        type=int,
        default=4096,
        help="Fixed N for subsample_fixed_n / regional_crops (default: 512).",
    )
    p.add_argument(
        "--global_max",
        type=int,
        default=4096,
        help="Global max padded length for pad_to_global strategy (default: 4096).",
    )
    p.add_argument(
        "--crop_frac",
        type=float,
        default=0.5,
        help="Crop fraction for regional_crops strategy (default: 0.5).",
    )
    p.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        help="Strategies to benchmark (default: all).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for report.json and table (default: outputs/benchmark_batching).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        force=True,
    )
    args = parse_args()

    mmap_dir = Path(args.mmap_dir)
    if not mmap_dir.is_dir():
        raise FileNotFoundError(f"Mmap directory not found: {mmap_dir}")
    if not (mmap_dir / "index_arrays.npz").is_file():
        raise FileNotFoundError(
            f"No index_arrays.npz in {mmap_dir}. Build with: python scripts/build_mmap.py"
        )

    strategies = args.strategies or ALL_STRATEGIES

    if args.augmentation:
        augmentation_cfg = DEFAULT_AUG_CFG
        coords_aware = True
        logger.info("Augmentation: ENABLED (default SSL augmentation)")
    else:
        augmentation_cfg = IDENTITY_AUG_CFG
        coords_aware = False
        logger.info("Augmentation: DISABLED (identity, no perturbation)")

    batching_cfg = {
        "n_buckets": 15,
        "token_budget": args.token_budget,
        "max_batch_size": 64,
        "min_batch_size": 2,
        "fixed_n": args.fixed_n,
        "global_max": args.global_max,
        "crop_frac": args.crop_frac,
        "sort_within_batch": True,
        "drop_last": True,
        "global_crop_n": args.fixed_n,
        "local_crop_n": max(1, args.fixed_n // 4),
        "n_local_crops": 6,
        "context_ratio": 0.5,
    }

    # Log dataset info
    idx = np.load(str(mmap_dir / "index_arrays.npz"), allow_pickle=True)
    all_lengths = idx["lengths"]
    logger.info(
        "Mmap: %s — %d slides, total patches: %s, "
        "bag sizes: mean=%.0f, median=%.0f, min=%d, max=%d",
        mmap_dir,
        len(all_lengths),
        f"{all_lengths.sum():,}",
        all_lengths.mean(),
        np.median(all_lengths),
        all_lengths.min(),
        all_lengths.max(),
    )
    logger.info(
        "Config: batch_size=%d, max_instances=%s, dataset_max_instances=%s, num_workers=%d",
        args.batch_size,
        args.max_instances,
        args.dataset_max_instances,
        args.num_workers,
    )
    if args.max_slides:
        logger.info("Limiting to %d slides", args.max_slides)
    logger.info(
        "Warmup batches: %d, timed epochs: %d",
        args.n_warmup_batches,
        args.n_epochs,
    )
    logger.info("Strategies: %s\n", strategies)

    # ── Benchmark each strategy ──────────────────────────────────────────
    results = []
    for strategy in strategies:
        logger.info("=" * 60)
        logger.info("Benchmarking: %s", strategy)
        logger.info("=" * 60)
        try:
            metrics = _benchmark_strategy(
                strategy=strategy,
                mmap_dir=str(mmap_dir),
                batch_size=args.batch_size,
                max_instances=args.max_instances,
                dataset_max_instances=args.dataset_max_instances,
                num_workers=args.num_workers,
                n_warmup_batches=args.n_warmup_batches,
                n_epochs=args.n_epochs,
                batching_cfg=batching_cfg,
                augmentation_cfg=augmentation_cfg,
                coords_aware=coords_aware,
                max_slides=args.max_slides,
                seed=args.seed,
            )
            results.append(metrics)

            if "error" in metrics:
                logger.error("  %s", metrics["error"])
                continue

            logger.info(
                "  Batches/epoch: %d, Time: %.3fs +/- %.3fs",
                metrics["n_batches_per_epoch"],
                metrics["epoch_time_mean_s"],
                metrics["epoch_time_std_s"],
            )
            logger.info(
                "  Padding: mean=%.4f, p50=%.4f, p90=%.4f",
                metrics["padding_ratio_mean"],
                metrics["padding_ratio_p50"],
                metrics["padding_ratio_p90"],
            )
            logger.info(
                "  Batch size: mean=%.1f, std=%.1f, range=[%d, %d]",
                metrics["batch_size_mean"],
                metrics["batch_size_std"],
                metrics["batch_size_min"],
                metrics["batch_size_max"],
            )
            logger.info(
                "  Memory: mean=%.1fKB, max=%.1fKB\n",
                metrics["peak_memory_mean_kb"],
                metrics["peak_memory_max_kb"],
            )
        except Exception:
            logger.exception("  FAILED")
            results.append({"strategy": strategy, "error": "exception (see logs)"})

    # ── Output ───────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/benchmark_batching")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(
            {"args": vars(args), "results": results},
            f,
            indent=2,
        )
    logger.info("Report saved to %s", report_path)

    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        table = _format_table(valid_results)
        table_path = output_dir / "comparison_table.txt"
        table_path.write_text(table)
        logger.info("\n%s", table)
        logger.info("\nTable saved to %s", table_path)


if __name__ == "__main__":
    main()
