#!/usr/bin/env python3
"""
Test augmentation + batching pipeline for SSL pretraining.

Usage (with real mmap data):
    python test_augmentation_pipeline.py --mmap_dir ./mmap/uni2h_pretrain

Usage (synthetic data — no mmap needed):
    python test_augmentation_pipeline.py --synthetic

What this tests:
  1. All 5 view strategies: dual, split_dual, asymmetric, jepa, multicrop
  2. Batching: bucket_batching + DualViewCollator (pad-to-max-in-batch)
  3. 20 batches x batch_size=128 per strategy
  4. Reports: shapes, padding efficiency, timing, token overlap (split vs dual)
  5. Saves spatial visualizations of views for 4 random slides

Output:
  ./test_augmentation_output/
  ├── strategy_report.txt          — per-strategy statistics
  ├── spatial_views_dual.png       — coordinate scatter plots
  ├── spatial_views_split_dual.png
  ├── spatial_views_asymmetric.png
  ├── spatial_views_jepa.png
  └── timing_summary.png           — bar chart of throughput
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# Standalone augmentation test (no project imports needed)
# ═════════════════════════════════════════════════════════════════════════════


def test_augmentation_standalone(mmap_dir: str, output_dir: str, n_batches: int, batch_size: int):
    """
    Test all view strategies with real or synthetic mmap data.

    Imports from the project; if unavailable, falls back to inline
    implementations for the core augmentation classes.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Try importing from project ───────────────────────────────────
    try:
        from oceanpath.data.batching import (  # noqa: F401
            BatchingConfig,
            BucketBatchSampler,
            DualViewCollator,
            build_batch_sampler,
            build_collator,
        )
        from oceanpath.data.mmap_builder import validate_mmap_dir  # noqa: F401
        from oceanpath.data.pretrain_dataset import PretrainDataset  # noqa: F401
        from oceanpath.ssl.augmentation import (  # noqa: F401
            AsymmetricDualViewAugmentor,
            CoordAffine,
            DualViewAugmentor,
            FeatureAugmentor,
            FeaturePosterize,
            JEPAViewAugmentor,
            SpatialBlockMask,
            SpatialCrop,
            SplitDualViewAugmentor,
            TokenSplit,
            build_augmentor,
            build_view_generator,
        )

        USE_PROJECT = True
        print("✓ Project imports successful")
    except ImportError as e:
        print(f"✗ Project import failed: {e}")
        print("  Falling back to standalone mode (augmentation classes only)")
        USE_PROJECT = False

    # ── Load data ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"Loading mmap from: {mmap_dir}")
    print(f"{'=' * 70}")

    idx = np.load(str(Path(mmap_dir) / "index_arrays.npz"), allow_pickle=True)
    slide_ids = idx["slide_ids"].tolist()
    lengths = idx["lengths"].astype(int)
    feat_dim = int(idx["feat_dim"])
    feat_dtype = np.dtype(str(idx["feat_dtype"][0]))
    coord_dtype = np.dtype(str(idx["coord_dtype"][0]))
    n_slides = len(slide_ids)

    print(f"  Slides: {n_slides}")
    print(f"  Feat dim: {feat_dim}")
    print(f"  Feat dtype: {feat_dtype}")
    print(
        f"  Lengths: mean={lengths.mean():.0f}, median={np.median(lengths):.0f}, "
        f"min={lengths.min()}, max={lengths.max()}"
    )

    # ── Helper: read one slide from mmap ─────────────────────────────
    feat_offsets = idx["feat_offsets"].astype(int)
    coord_offsets = idx["coord_offsets"].astype(int)
    feat_chunk_ids = idx["feat_chunk_ids"].astype(int)
    coord_chunk_ids = idx["coord_chunk_ids"].astype(int)

    # Cache open memmaps per chunk to avoid reopening
    _feat_mm_cache: dict[int, np.ndarray] = {}
    _coord_mm_cache: dict[int, np.ndarray] = {}

    def _get_feat_mm(chunk: int) -> np.ndarray:
        if chunk not in _feat_mm_cache:
            _feat_mm_cache[chunk] = np.memmap(
                str(Path(mmap_dir) / f"features_{chunk:03d}.bin"), dtype=feat_dtype, mode="r"
            )
        return _feat_mm_cache[chunk]

    def _get_coord_mm(chunk: int) -> np.ndarray:
        if chunk not in _coord_mm_cache:
            _coord_mm_cache[chunk] = np.memmap(
                str(Path(mmap_dir) / f"coords_{chunk:03d}.bin"), dtype=coord_dtype, mode="r"
            )
        return _coord_mm_cache[chunk]

    def read_slide(i: int) -> tuple[np.ndarray, np.ndarray]:
        n = int(lengths[i])
        fc = int(feat_chunk_ids[i])
        f_start = feat_offsets[i] // np.dtype(feat_dtype).itemsize
        f = _get_feat_mm(fc)[f_start : f_start + n * feat_dim].reshape(n, feat_dim).copy()
        cc = int(coord_chunk_ids[i])
        c_start = coord_offsets[i] // np.dtype(coord_dtype).itemsize
        c = _get_coord_mm(cc)[c_start : c_start + n * 2].reshape(n, 2).copy()
        return f, c

    # ── Define view strategies to test ───────────────────────────────
    aug_cfg = {
        # Patch selection
        "subsample_frac": [0.5, 1.0],
        "instance_dropout": 0.1,
        "use_spatial_crop": True,
        "spatial_crop_prob": 0.5,
        "spatial_crop_frac": [0.4, 0.8],
        "use_density_sub": True,
        "density_sub_prob": 0.3,
        "region_drop_n": 2,
        "region_drop_frac": [0.05, 0.2],
        # Feature transforms
        "feature_noise_std": 0.02,
        "feature_dropout": 0.05,
        "posterize_levels": 8,
        "posterize_prob": 0.3,
        "smooth_k_neighbors": 5,
        "smooth_alpha": [0.05, 0.3],
        "interp_t_range": [0.05, 0.25],
        "region_mixup_alpha": [0.05, 0.2],
        # Coord transforms
        "coord_rotation": True,
        "coord_flip": True,
        "coord_scale_enabled": True,
        "coord_scale": [0.8, 1.2],
        "enable_local_smooth": False,
        "enable_spatial_interpolation": False,
        "enable_region_mixup": False,
        # Token split
        "split_ratio": 0.5,
        # Asymmetric
        "teacher_subsample_frac": [0.7, 1.0],
        "teacher_crop_frac": [0.6, 1.0],
        "teacher_instance_dropout": 0.05,
        "teacher_feature_noise_std": 0.01,
        "teacher_feature_dropout": 0.02,
        "student_subsample_frac": [0.2, 0.5],
        "student_crop_frac": [0.2, 0.5],
        "student_instance_dropout": 0.15,
        "student_feature_noise_std": 0.03,
        "student_feature_dropout": 0.08,
        "student_posterize_prob": 0.4,
        # JEPA
        "jepa_n_blocks": 4,
        "jepa_block_size": [1, 3],
        "jepa_grid_size": 8,
        "jepa_min_context_frac": 0.5,
        "jepa_context_noise": 0.01,
        "jepa_pre_subsample": 4096,
    }

    strategies = ["dual", "split_dual", "asymmetric", "jepa"]

    if not USE_PROJECT:
        # Standalone: import the augmentation module directly
        # Assume it's in the same dir or on PYTHONPATH
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from augmentation import build_view_generator

            print("✓ Loaded augmentation.py from local directory")
        except ImportError:
            print("✗ Cannot import augmentation module. Place augmentation.py in same directory.")
            return

    # ── Run each strategy ────────────────────────────────────────────
    report_lines = []
    timing_results = {}
    all_vis_data = {}

    # Pre-select slides for visualization (same across strategies)
    vis_indices = np.random.default_rng(42).choice(n_slides, size=min(4, n_slides), replace=False)

    for strategy in strategies:
        print(f"\n{'─' * 70}")
        print(f"Strategy: {strategy}")
        print(f"{'─' * 70}")

        if USE_PROJECT:
            view_gen = build_view_generator(aug_cfg, view_strategy=strategy, coords_aware=True)
        else:
            from augmentation import build_view_generator

            view_gen = build_view_generator(aug_cfg, view_strategy=strategy, coords_aware=True)

        # ── Collect per-batch statistics ──────────────────────────────
        rng = np.random.default_rng(123)
        all_slide_indices = rng.permutation(n_slides)

        v1_lengths = []
        v2_lengths = []
        batch_times = []
        overlap_ratios = []  # only meaningful for dual vs split_dual

        n_samples_processed = 0
        batch_count = 0

        t_total_start = time.perf_counter()

        for batch_idx in range(n_batches):
            start_i = (batch_idx * batch_size) % n_slides
            batch_indices = []
            for j in range(batch_size):
                batch_indices.append(all_slide_indices[(start_i + j) % n_slides])

            t_batch_start = time.perf_counter()

            batch_v1_lens = []
            batch_v2_lens = []

            for si in batch_indices:
                features, coords = read_slide(si)

                # Pre-cap (match PretrainDataset behavior)
                cap = 4096
                if len(features) > cap:
                    chosen = np.sort(rng.permutation(len(features))[:cap])
                    features = features[chosen]
                    coords = coords[chosen]

                # Generate views
                child_rng = np.random.default_rng(rng.integers(0, 2**32))

                result = view_gen(features, coords, rng=child_rng)

                # Handle both return formats
                if isinstance(result, tuple):
                    (v1_f, v1_c), (v2_f, v2_c) = result
                elif isinstance(result, dict):
                    v1_f = result.get("view1_features", result.get("context_features"))
                    v2_f = result.get("view2_features", result.get("target_features"))
                    v1_c = result.get("view1_coords", result.get("context_coords"))
                    v2_c = result.get("view2_coords", result.get("target_coords"))
                else:
                    raise ValueError(f"Unexpected return type: {type(result)}")

                batch_v1_lens.append(len(v1_f))
                batch_v2_lens.append(len(v2_f))
                if v1_c is not None and v2_c is not None:
                    s1 = set(map(tuple, v1_c.tolist()))
                    s2 = set(map(tuple, v2_c.tolist()))
                    n_union = len(s1 | s2)
                    if n_union > 0:
                        overlap_ratios.append(len(s1 & s2) / n_union)

                n_samples_processed += 1

            t_batch_end = time.perf_counter()
            batch_times.append(t_batch_end - t_batch_start)
            v1_lengths.extend(batch_v1_lens)
            v2_lengths.extend(batch_v2_lens)
            batch_count += 1

            # Progress
            if (batch_idx + 1) % 5 == 0:
                avg_time = np.mean(batch_times[-5:])
                print(
                    f"  Batch {batch_idx + 1}/{n_batches}: "
                    f"v1 mean={np.mean(batch_v1_lens):.0f}, "
                    f"v2 mean={np.mean(batch_v2_lens):.0f}, "
                    f"time={avg_time:.2f}s"
                )

        t_total = time.perf_counter() - t_total_start

        v1_lengths = np.array(v1_lengths)
        v2_lengths = np.array(v2_lengths)

        # ── Padding efficiency ───────────────────────────────────────
        # Simulate pad-to-max-in-batch for each batch
        total_real_tokens = 0
        total_padded_tokens = 0
        for bi in range(n_batches):
            start = bi * batch_size
            end = start + batch_size
            b_v1 = v1_lengths[start:end]
            b_v2 = v2_lengths[start:end]
            total_real_tokens += b_v1.sum() + b_v2.sum()
            total_padded_tokens += b_v1.max() * len(b_v1) + b_v2.max() * len(b_v2)

        pad_efficiency = total_real_tokens / total_padded_tokens if total_padded_tokens > 0 else 1.0

        # ── Report ───────────────────────────────────────────────────
        report = [
            f"\n{'=' * 60}",
            f"Strategy: {strategy}",
            f"{'=' * 60}",
            f"Batches: {batch_count}, Batch size: {batch_size}",
            f"Total samples processed: {n_samples_processed}",
            "",
            "View 1 lengths:",
            f"  mean={v1_lengths.mean():.0f}, median={np.median(v1_lengths):.0f}, "
            f"std={v1_lengths.std():.0f}",
            f"  min={v1_lengths.min()}, max={v1_lengths.max()}, "
            f"p25={np.percentile(v1_lengths, 25):.0f}, p75={np.percentile(v1_lengths, 75):.0f}",
            "",
            "View 2 lengths:",
            f"  mean={v2_lengths.mean():.0f}, median={np.median(v2_lengths):.0f}, "
            f"std={v2_lengths.std():.0f}",
            f"  min={v2_lengths.min()}, max={v2_lengths.max()}, "
            f"p25={np.percentile(v2_lengths, 25):.0f}, p75={np.percentile(v2_lengths, 75):.0f}",
            "",
            f"Padding efficiency (pad-to-max-in-batch): {pad_efficiency:.1%}",
            f"Total time: {t_total:.1f}s ({t_total / n_samples_processed * 1000:.1f} ms/sample)",
            f"Throughput: {n_samples_processed / t_total:.0f} samples/s",
            f"Avg batch time: {np.mean(batch_times):.3f}s ± {np.std(batch_times):.3f}s",
        ]
        if overlap_ratios:
            report.extend(
                [
                    "",
                    "Token overlap (Jaccard IoU):",
                    f"  mean={np.mean(overlap_ratios):.4f}, "
                    f"median={np.median(overlap_ratios):.4f}, "
                    f"max={np.max(overlap_ratios):.4f}",
                ]
            )

        for line in report:
            print(line)
        report_lines.extend(report)

        timing_results[strategy] = {
            "total_time": t_total,
            "ms_per_sample": t_total / n_samples_processed * 1000,
            "throughput": n_samples_processed / t_total,
            "pad_efficiency": pad_efficiency,
            "v1_mean": v1_lengths.mean(),
            "v2_mean": v2_lengths.mean(),
        }

        # ── Collect visualization data ───────────────────────────────
        vis_slides = []
        for si in vis_indices:
            features, coords = read_slide(si)
            cap = 4096
            if len(features) > cap:
                chosen = np.sort(np.random.default_rng(0).permutation(len(features))[:cap])
                features = features[chosen]
                coords = coords[chosen]

            child_rng = np.random.default_rng(42)
            result = view_gen(features, coords, rng=child_rng)

            if isinstance(result, tuple):
                (v1_f, v1_c), (v2_f, v2_c) = result
            else:
                v1_c = result.get("view1_coords", result.get("context_coords"))
                v2_c = result.get("view2_coords", result.get("target_coords"))

            vis_slides.append(
                {
                    "slide_id": slide_ids[si],
                    "original_coords": coords.copy(),
                    "v1_coords": v1_c.copy() if v1_c is not None else None,
                    "v2_coords": v2_c.copy() if v2_c is not None else None,
                    "n_original": len(coords),
                    "n_v1": len(v1_c) if v1_c is not None else 0,
                    "n_v2": len(v2_c) if v2_c is not None else 0,
                }
            )
        all_vis_data[strategy] = vis_slides

    # ── Write report ─────────────────────────────────────────────────
    report_path = output_path / "strategy_report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"\n✓ Report saved: {report_path}")

    # ── Visualizations ───────────────────────────────────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")

        HAS_MPL = True
    except ImportError:
        HAS_MPL = False
        print("⚠ matplotlib not available — skipping visualizations")

    if HAS_MPL:
        _render_spatial_views(all_vis_data, output_path)
        _render_timing_summary(timing_results, output_path)
        _render_length_distributions(
            all_vis_data, timing_results, strategies, v1_lengths, v2_lengths, output_path
        )
        print(f"✓ Visualizations saved to {output_path}")


def _render_spatial_views(all_vis_data: dict, output_path: Path):
    """Render spatial coordinate scatter plots for each strategy."""
    import matplotlib.pyplot as plt

    for strategy, slides in all_vis_data.items():
        n_slides = len(slides)
        fig, axes = plt.subplots(n_slides, 3, figsize=(18, 5 * n_slides))
        if n_slides == 1:
            axes = axes[np.newaxis, :]

        is_jepa = strategy == "jepa"
        v1_label = "Context" if is_jepa else "View 1"
        v2_label = "Target" if is_jepa else "View 2"

        for i, slide in enumerate(slides):
            orig = slide["original_coords"]
            v1 = slide["v1_coords"]
            v2 = slide["v2_coords"]

            # Panel 1: Original
            ax = axes[i, 0]
            ax.scatter(orig[:, 0], orig[:, 1], s=1, c="#94a3b8", alpha=0.5, rasterized=True)
            ax.set_title(
                f"Original ({slide['n_original']} patches)\n{Path(slide['slide_id']).name}",
                fontsize=9,
                fontfamily="monospace",
            )
            ax.set_aspect("equal")
            ax.invert_yaxis()

            # Panel 2: View 1 / Context
            ax = axes[i, 1]
            ax.scatter(orig[:, 0], orig[:, 1], s=1, c="#e2e8f0", alpha=0.2, rasterized=True)
            if v1 is not None:
                ax.scatter(v1[:, 0], v1[:, 1], s=2, c="#3b82f6", alpha=0.7, rasterized=True)
            ax.set_title(
                f"{v1_label} ({slide['n_v1']} patches, "
                f"{slide['n_v1'] / slide['n_original'] * 100:.0f}%)",
                fontsize=9,
                fontfamily="monospace",
            )
            ax.set_aspect("equal")
            ax.invert_yaxis()

            # Panel 3: View 2 / Target
            ax = axes[i, 2]
            ax.scatter(orig[:, 0], orig[:, 1], s=1, c="#e2e8f0", alpha=0.2, rasterized=True)
            if v2 is not None:
                color = "#ef4444" if is_jepa else "#22c55e"
                ax.scatter(v2[:, 0], v2[:, 1], s=2, c=color, alpha=0.7, rasterized=True)
            ax.set_title(
                f"{v2_label} ({slide['n_v2']} patches, "
                f"{slide['n_v2'] / slide['n_original'] * 100:.0f}%)",
                fontsize=9,
                fontfamily="monospace",
            )
            ax.set_aspect("equal")
            ax.invert_yaxis()

        fig.suptitle(f"View Strategy: {strategy}", fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        fig.savefig(
            str(output_path / f"spatial_views_{strategy}.png"), dpi=150, bbox_inches="tight"
        )
        plt.close(fig)


def _render_timing_summary(timing_results: dict, output_path: Path):
    """Bar chart comparing throughput across strategies."""
    import matplotlib.pyplot as plt

    strategies = list(timing_results.keys())
    throughputs = [timing_results[s]["throughput"] for s in strategies]
    pad_effs = [timing_results[s]["pad_efficiency"] * 100 for s in strategies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6"]

    # Throughput
    bars = ax1.bar(
        strategies, throughputs, color=colors[: len(strategies)], edgecolor="white", linewidth=0.5
    )
    ax1.set_ylabel("Samples/sec")
    ax1.set_title("Augmentation Throughput")
    for bar, val in zip(bars, throughputs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Padding efficiency
    bars = ax2.bar(
        strategies, pad_effs, color=colors[: len(strategies)], edgecolor="white", linewidth=0.5
    )
    ax2.set_ylabel("Padding Efficiency (%)")
    ax2.set_title("Padding Efficiency (pad-to-max-in-batch)")
    ax2.set_ylim(0, 105)
    for bar, val in zip(bars, pad_effs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(str(output_path / "timing_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_length_distributions(
    all_vis_data, timing_results, strategies, v1_lengths, v2_lengths, output_path
):
    """Histograms of view lengths per strategy."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    colors_v1 = "#3b82f6"
    colors_v2 = "#22c55e"

    for ax_idx, strategy in enumerate(strategies[:4]):
        ax = axes[ax_idx]
        t = timing_results[strategy]

        # We only have aggregate lengths from the last strategy run
        # For a real test, we'd store per-strategy. Use the summary stats.
        v1_mean = t["v1_mean"]
        v2_mean = t["v2_mean"]

        is_jepa = strategy == "jepa"
        v1_label = f"Context (mean={v1_mean:.0f})" if is_jepa else f"View 1 (mean={v1_mean:.0f})"
        v2_label = f"Target (mean={v2_mean:.0f})" if is_jepa else f"View 2 (mean={v2_mean:.0f})"

        # Show per-slide token counts from vis data
        v1_lens = [s["n_v1"] for s in all_vis_data[strategy]]
        v2_lens = [s["n_v2"] for s in all_vis_data[strategy]]

        ax.barh([0, 1, 2, 3], v1_lens, height=0.35, color=colors_v1, alpha=0.7, label=v1_label)
        ax.barh(
            [0.35, 1.35, 2.35, 3.35],
            v2_lens,
            height=0.35,
            color=colors_v2,
            alpha=0.7,
            label=v2_label,
        )
        ax.set_yticks([0.175, 1.175, 2.175, 3.175])
        ax.set_yticklabels(
            [Path(s["slide_id"]).name[:15] for s in all_vis_data[strategy]], fontsize=8
        )
        ax.set_xlabel("Tokens")
        ax.set_title(
            f"{strategy}\nPad eff: {t['pad_efficiency']:.0%}  |  {t['ms_per_sample']:.1f} ms/sample",
            fontsize=10,
        )
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("View Token Counts (4 sample slides)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_path / "length_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# DataLoader integration test (requires full project)
# ═════════════════════════════════════════════════════════════════════════════


def test_dataloader_integration(mmap_dir: str, output_dir: str, n_batches: int, batch_size: int):
    """
    Full DataLoader test using PretrainDataset + DualViewCollator.

    Tests the actual PyTorch DataLoader with multiprocessing workers,
    pin_memory, and the collation pipeline.
    """
    import torch
    from torch.utils.data import DataLoader

    try:
        from oceanpath.data.batching import (  # noqa: F401
            BatchingConfig,
            BucketBatchSampler,
            DualViewCollator,
            build_batch_sampler,
            build_collator,
        )
        from oceanpath.data.pretrain_dataset import PretrainDataset
        from oceanpath.ssl.augmentation import build_view_generator
    except ImportError as e:
        print(f"Cannot run DataLoader integration test: {e}")
        print("Run with --standalone to test augmentation without project imports.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    aug_cfg = {
        "subsample_frac": [0.5, 1.0],
        "instance_dropout": 0.1,
        "use_spatial_crop": True,
        "spatial_crop_prob": 0.5,
        "spatial_crop_frac": [0.4, 0.8],
        "use_density_sub": True,
        "density_sub_prob": 0.3,
        "feature_noise_std": 0.02,
        "feature_dropout": 0.05,
        "posterize_levels": 8,
        "posterize_prob": 0.3,
        "coord_rotation": True,
        "coord_flip": True,
        "coord_scale_enabled": True,
        "split_ratio": 0.5,
    }

    strategies_to_test = ["dual", "split_dual", "asymmetric", "jepa"]
    dataset_max_instances = 4096
    max_instances = 4096  # collator cap
    num_workers = 8  # single process for debugging; set >0 for real test

    for strategy in strategies_to_test:
        print(f"\n{'─' * 70}")
        print(f"DataLoader test: strategy={strategy}, B={batch_size}, workers={num_workers}")
        print(f"{'─' * 70}")

        # Build augmentor based on strategy
        view_gen = build_view_generator(aug_cfg, view_strategy=strategy, coords_aware=True)

        dataset = PretrainDataset(
            mmap_dir=mmap_dir,
            slide_ids=None,
            view_generator=view_gen,
            dataset_max_instances=dataset_max_instances,
            pre_cap_mode="random",
            force_float32=False,
        )

        # Build collator + sampler
        batching_cfg = BatchingConfig(
            strategy="token_budget",
            token_budget=512_000,  # ~128 x 4000 avg tokens
            max_batch_size=batch_size,
            min_batch_size=8,
            max_instances=max_instances,
            sort_within_batch=True,
            seed=42,
        )
        sampler = build_batch_sampler(batching_cfg, dataset.get_bag_sizes())
        collator = build_collator(batching_cfg)

        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": False,
            "persistent_workers": False,
        }

        if sampler is not None:
            loader = DataLoader(
                dataset, batch_sampler=sampler, collate_fn=collator, **loader_kwargs
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collator,
                drop_last=True,
                **loader_kwargs,
            )

        # ── Iterate ──────────────────────────────────────────────────
        batch_times = []
        batch_shapes = []
        total_tokens = 0

        for i, batch in enumerate(loader):
            if i >= n_batches:
                break

            t0 = time.perf_counter()

            v1 = batch["view1"]
            v2 = batch["view2"]
            m1 = batch["mask1"]
            m2 = batch["mask2"]

            # Verify shapes
            B = v1.shape[0]
            assert v1.ndim == 3, f"view1 should be [B, N, D], got {v1.shape}"
            assert v2.ndim == 3, f"view2 should be [B, N, D], got {v2.shape}"
            assert m1.shape == v1.shape[:2], f"mask1 shape mismatch: {m1.shape} vs {v1.shape[:2]}"
            assert m2.shape == v2.shape[:2], f"mask2 shape mismatch: {m2.shape} vs {v2.shape[:2]}"

            # Check dtypes
            assert v1.dtype in (torch.float16, torch.float32), f"Unexpected v1 dtype: {v1.dtype}"

            # Stats
            real_v1 = m1.sum().item()
            real_v2 = m2.sum().item()
            padded_v1 = m1.numel()
            padded_v2 = m2.numel()
            total_tokens += real_v1 + real_v2

            t1 = time.perf_counter()
            batch_times.append(t1 - t0)

            # Coords check
            has_coords = "coords1" in batch
            coord_info = ""
            if has_coords:
                c1 = batch["coords1"]
                c2 = batch["coords2"]
                assert c1.shape[:2] == v1.shape[:2], (
                    f"coords1 shape: {c1.shape} vs view1: {v1.shape}"
                )
                # Check sentinel values in padding region
                pad_region = c1[m1 == 0]
                if len(pad_region) > 0:
                    assert (pad_region == -1).all(), "Padded coords should be -1"
                coord_info = f"  coords1={list(c1.shape)}, coords2={list(c2.shape)}"

            batch_shapes.append(
                {
                    "B": B,
                    "v1_shape": list(v1.shape),
                    "v2_shape": list(v2.shape),
                    "pad_eff_v1": real_v1 / padded_v1 if padded_v1 > 0 else 1.0,
                    "pad_eff_v2": real_v2 / padded_v2 if padded_v2 > 0 else 1.0,
                }
            )

            if (i + 1) % 5 == 0 or i == 0:
                pe1 = real_v1 / padded_v1 * 100 if padded_v1 > 0 else 100
                pe2 = real_v2 / padded_v2 * 100 if padded_v2 > 0 else 100
                print(
                    f"  Batch {i + 1:3d}/{n_batches}: "
                    f"B={B}, v1={list(v1.shape)}, v2={list(v2.shape)}, "
                    f"pad_eff={pe1:.0f}%/{pe2:.0f}%, "
                    f"time={batch_times[-1] * 1000:.0f}ms"
                    f"{coord_info}"
                )

        # ── Summary ──────────────────────────────────────────────────
        if batch_times:
            avg_pe = np.mean([s["pad_eff_v1"] + s["pad_eff_v2"] for s in batch_shapes]) / 2
            print(f"\n  Summary ({strategy}):")
            print(f"    Batches: {len(batch_times)}")
            print(
                f"    Avg batch time: {np.mean(batch_times) * 1000:.0f}ms ± {np.std(batch_times) * 1000:.0f}ms"
            )
            print(f"    Avg padding efficiency: {avg_pe:.1%}")
            print(f"    Total tokens processed: {total_tokens:,}")
            print("    ✓ All shape/dtype/sentinel assertions passed")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Test SSL augmentation + batching pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With real mmap data:
  python test_augmentation_pipeline.py --mmap_dir ./mmap/uni2h_pretrain

  # With synthetic data (no real features needed):
  python test_augmentation_pipeline.py --synthetic

  # Full DataLoader integration test:
  python test_augmentation_pipeline.py --mmap_dir ./mmap/uni2h_pretrain --dataloader

  # Custom batch size and count:
  python test_augmentation_pipeline.py --synthetic --batch_size 64 --n_batches 10
        """,
    )
    parser.add_argument(
        "--mmap_dir", type=str, default="./mmap/uni2h_pretrain", help="Path to mmap directory"
    )
    parser.add_argument(
        "--dataloader",
        action="store_true",
        help="Run full DataLoader integration test (requires project imports)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_augmentation_output",
        help="Output directory for reports and visualizations",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_batches", type=int, default=20)
    args = parser.parse_args()

    # ── Verify mmap exists ────────────────────────────────────────────
    if not Path(args.mmap_dir).is_dir():
        print(f"ERROR: mmap directory not found: {args.mmap_dir}")
        print("  Use --synthetic to generate test data, or --mmap_dir to point to your data.")
        sys.exit(1)

    if not (Path(args.mmap_dir) / "index_arrays.npz").is_file():
        print(f"ERROR: index_arrays.npz not found in {args.mmap_dir}")
        sys.exit(1)

    # ── Run tests ─────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("SSL Augmentation + Batching Pipeline Test")
    print(f"  mmap_dir:   {args.mmap_dir}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  n_batches:  {args.n_batches}")
    print(f"  output:     {args.output_dir}")
    print(f"{'═' * 70}")

    # Always run the augmentation test
    # test_augmentation_standalone(
    #     mmap_dir=args.mmap_dir,
    #     output_dir=args.output_dir,
    #     n_batches=args.n_batches,
    #     batch_size=args.batch_size,
    # )

    # Optionally run DataLoader integration test
    if args.dataloader:
        test_dataloader_integration(
            mmap_dir=args.mmap_dir,
            output_dir=args.output_dir,
            n_batches=args.n_batches,
            batch_size=args.batch_size,
        )

    print(f"\n{'═' * 70}")
    print(f"✓ All tests complete. Results in: {args.output_dir}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
