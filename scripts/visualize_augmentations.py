#!/usr/bin/env python3
"""
Visualize augmented views for each SSL view-generation strategy.

Generates a multi-panel figure per strategy showing:
  - Original slide patch layout (all patches)
  - View 1 patches (spatial coverage)
  - View 2 / target patches (spatial coverage)
  - Overlap / disjoint analysis

Usage:
    python visualize_augmentations.py                          # synthetic data, all strategies
    python visualize_augmentations.py --mode single --strategy JEPA --n_repeats 4
    python visualize_augmentations.py --mode features          # feature change heatmap
    python visualize_augmentations.py --mmap_dir /path/to/mmap --slide_idx 42
    python visualize_augmentations.py --output views.png       # save instead of display
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oceanpath.ssl.augmentation import (
    AsymmetricDualViewAugmentor,
    DualViewAugmentor,
    FeatureAugmentor,
    FeaturePosterize,
    JEPAViewAugmentor,
    LocalFeatureSmooth,
    SpatialBlockMask,
    SpatialCrop,
    SpatialRegionDrop,
    SplitDualViewAugmentor,
)
from oceanpath.data.mmap_builder import _spatial_stratified_subsample

# ── Color scheme ─────────────────────────────────────────────────────────
C_BG = "#1a1d27"
C_GRID = "#2e3345"
C_ALL = "#555b73"
C_VIEW1 = "#6ec1e4"
C_VIEW2 = "#e8845c"
C_OVERLAP = "#b48eed"
C_DROPPED = "#3a3f52"
C_TEXT = "#e2e4ea"
C_DIM = "#9499ad"


def load_mmap_slide(mmap_dir, slide_idx=0, idx_arrays=None):
    p = Path(mmap_dir)
    idx = (
        idx_arrays
        if idx_arrays is not None
        else np.load(str(p / "index_arrays.npz"), allow_pickle=True)
    )
    lengths = idx["lengths"].astype(int)
    slide_idx = min(slide_idx, len(lengths) - 1)
    n = lengths[slide_idx]
    sid = str(idx["slide_ids"][slide_idx])
    fd = int(idx["feat_dim"])
    fdt = np.dtype(str(idx["feat_dtype"][0]))
    cdt = np.dtype(str(idx["coord_dtype"][0]))
    cd = int(idx["coord_dim"]) if "coord_dim" in idx else 2

    fc, fo = int(idx["feat_chunk_ids"][slide_idx]), int(idx["feat_offsets"][slide_idx])
    mm = np.memmap(str(p / f"features_{fc:03d}.bin"), dtype=fdt, mode="r")
    s = fo // fdt.itemsize
    features = mm[s : s + n * fd].reshape(n, fd).copy()

    cc, co = int(idx["coord_chunk_ids"][slide_idx]), int(idx["coord_offsets"][slide_idx])
    mm2 = np.memmap(str(p / f"coords_{cc:03d}.bin"), dtype=cdt, mode="r")
    s2 = co // cdt.itemsize
    coords = mm2[s2 : s2 + n * cd].reshape(n, cd).copy()
    return features, coords, sid


# ── Build all strategies ─────────────────────────────────────────────────


def _aug(light=False, post_split=False):
    if light:
        return FeatureAugmentor(
            subsample_frac=(0.7, 1.0),
            instance_dropout=0.05,
            feature_noise_std=0.01,
            feature_dropout=0.02,
            spatial_crop=SpatialCrop(crop_frac=(0.6, 1.0)),
            use_spatial_crop=True,
            spatial_crop_prob=0.6,
        )
    if post_split:
        # Gentler augmentation AFTER 50/50 disjoint split
        return FeatureAugmentor(
            subsample_frac=(0.7, 0.95),
            instance_dropout=0.1,
            feature_noise_std=0.15,
            feature_dropout=0.1,
            spatial_crop=SpatialCrop(crop_frac=(0.6, 0.9)),
            spatial_region_drop=SpatialRegionDrop(n_regions=1, region_frac=(0.05, 0.15)),
            feature_posterize=FeaturePosterize(n_levels=8, prob=0.2),
            use_spatial_crop=True,
            spatial_crop_prob=0.4,
        )
    return FeatureAugmentor(
        subsample_frac=(0.4, 0.8),
        instance_dropout=0.1,
        feature_noise_std=0.02,
        feature_dropout=0.05,
        spatial_crop=SpatialCrop(crop_frac=(0.3, 0.7)),
        spatial_region_drop=SpatialRegionDrop(n_regions=2, region_frac=(0.05, 0.15)),
        local_smooth=LocalFeatureSmooth(k_neighbors=5, alpha_range=(0.05, 0.2)),
        feature_posterize=FeaturePosterize(n_levels=8, prob=0.3),
        use_spatial_crop=True,
        spatial_crop_prob=0.5,
    )


def build_all_strategies():
    """Build strategies WITHOUT CoordAffine so overlap classification works."""
    base = _aug(light=False)
    light = _aug(light=True)
    heavy = FeatureAugmentor(
        subsample_frac=(0.2, 0.5),
        instance_dropout=0.15,
        feature_noise_std=0.03,
        feature_dropout=0.08,
        spatial_crop=SpatialCrop(crop_frac=(0.2, 0.5)),
        # coord_affine removed
        feature_posterize=FeaturePosterize(n_levels=6, prob=0.5),
        use_spatial_crop=True,
        spatial_crop_prob=0.8,
    )
    ctx_aug = FeatureAugmentor(
        subsample_frac=(1.0, 1.0),
        instance_dropout=0.0,
        feature_noise_std=0.01,
        feature_dropout=0.0,
        # coord_affine removed
    )
    return {
        "DualView (SimCLR baseline)": DualViewAugmentor(base),
        "SplitDual (VICReg + SPT)": SplitDualViewAugmentor(_aug(post_split=True), split_ratio=0.5),
        "Asymmetric (BYOL)": AsymmetricDualViewAugmentor(light, heavy),
        "Asymmetric+Split (BYOL+SPT)": AsymmetricDualViewAugmentor(
            light, heavy, use_split=True, split_ratio=0.6
        ),
        "JEPA (context+target)": JEPAViewAugmentor(
            block_mask=SpatialBlockMask(n_blocks=4, block_size_range=(1, 3), grid_size=8),
            context_augmentor=ctx_aug,
            # coord_affine removed
            pre_subsample=3000,
        ),
    }


# ── Plotting helpers ─────────────────────────────────────────────────────


def _classify_by_index(N_all, v1_idx, v2_idx):
    """0=dropped, 1=v1-only, 2=v2-only, 3=overlap.  Index-based, vectorized."""
    labels = np.zeros(N_all, dtype=int)
    v1 = np.asarray(v1_idx, dtype=int)
    v2 = np.asarray(v2_idx, dtype=int)
    if len(v1):
        labels[v1] = 1
    if len(v2):
        overlap = labels[v2] == 1  # already marked as v1
        labels[v2[overlap]] = 3  # promote to overlap
        labels[v2[~overlap]] = 2  # v2-only
    return labels


def _get_view_indices(all_coords, v_coords_pre_affine):
    """Map view coords (before affine) back to original indices via exact match."""
    lookup = {}
    for i, c in enumerate(all_coords):
        lookup[tuple(c.tolist())] = i
    return [lookup[tuple(c.tolist())] for c in v_coords_pre_affine if tuple(c.tolist()) in lookup]


def _style_ax(ax):
    ax.set_facecolor(C_BG)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.tick_params(colors=C_GRID, labelsize=5)
    for sp in ax.spines.values():
        sp.set_color(C_GRID)


def plot_row(axes, all_c, v1_c, v2_c, name, is_jepa=False):
    s = 3
    a = 0.7
    lbl1 = "Context" if is_jepa else "View 1"
    lbl2 = "Target blocks" if is_jepa else "View 2"

    axes[0].scatter(all_c[:, 0], all_c[:, 1], c=C_ALL, s=s, alpha=0.5, edgecolors="none")
    axes[0].set_title(f"Original ({len(all_c)})", fontsize=8, color=C_TEXT, pad=4)

    axes[1].scatter(all_c[:, 0], all_c[:, 1], c=C_DROPPED, s=s * 0.5, alpha=0.15, edgecolors="none")
    axes[1].scatter(v1_c[:, 0], v1_c[:, 1], c=C_VIEW1, s=s, alpha=a, edgecolors="none")
    axes[1].set_title(f"{lbl1} ({len(v1_c)})", fontsize=8, color=C_VIEW1, pad=4)

    axes[2].scatter(all_c[:, 0], all_c[:, 1], c=C_DROPPED, s=s * 0.5, alpha=0.15, edgecolors="none")
    axes[2].scatter(v2_c[:, 0], v2_c[:, 1], c=C_VIEW2, s=s, alpha=a, edgecolors="none")
    axes[2].set_title(f"{lbl2} ({len(v2_c)})", fontsize=8, color=C_VIEW2, pad=4)

    v1_idx = _get_view_indices(all_c, v1_c)
    v2_idx = _get_view_indices(all_c, v2_c)
    labels = _classify_by_index(len(all_c), v1_idx, v2_idx)
    for v, c_ in [(0, C_DROPPED), (1, C_VIEW1), (2, C_VIEW2), (3, C_OVERLAP)]:
        m = labels == v
        if m.any():
            axes[3].scatter(
                all_c[m, 0], all_c[m, 1], c=c_, s=s, alpha=a if v > 0 else 0.15, edgecolors="none"
            )
    n_ov = (labels == 3).sum()
    n_dr = (labels == 0).sum()
    axes[3].set_title(f"Overlap:{n_ov}  Drop:{n_dr}", fontsize=7, color=C_DIM, pad=4)

    for ax in axes:
        _style_ax(ax)


# ── Main visualizations ─────────────────────────────────────────────────


def visualize_all(features, coords, seed=42, title_prefix="", output_path=None):
    strats = build_all_strategies()
    n = len(strats)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.5 * n), facecolor=C_BG)
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        f"{title_prefix}SSL View Strategies — Spatial Coverage\n({len(features)} patches, D={features.shape[1]})",
        fontsize=13,
        color=C_TEXT,
        fontweight="bold",
        y=0.98,
    )

    for row, (name, gen) in enumerate(strats.items()):
        rng = np.random.default_rng(seed)
        (v1f, v1c), (v2f, v2c) = gen(features, coords, rng=rng)
        if v1c is None:
            v1c = np.zeros((len(v1f), 2), dtype=np.int32)
        if v2c is None:
            v2c = np.zeros((len(v2f), 2), dtype=np.int32)
        axes[row, 0].annotate(
            name,
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-60, 0),
            textcoords="offset points",
            fontsize=9,
            color=C_TEXT,
            fontweight="bold",
            ha="right",
            va="center",
        )
        plot_row(axes[row], coords, v1c, v2c, name, is_jepa="JEPA" in name)

    legend = [
        mpatches.Patch(color=c, label=lbl)
        for c, lbl in [
            (C_VIEW1, "View 1 / Context"),
            (C_VIEW2, "View 2 / Target"),
            (C_OVERLAP, "Overlap"),
            (C_DROPPED, "Dropped"),
        ]
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        fontsize=9,
        facecolor=C_BG,
        edgecolor=C_GRID,
        labelcolor=C_TEXT,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.005),
    )
    plt.tight_layout(rect=[0.08, 0.03, 1.0, 0.96])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def visualize_single(features, coords, strategy="JEPA", n_repeats=4, seed=42, output_path=None):
    strats = build_all_strategies()
    matches = [k for k in strats if strategy.lower() in k.lower()]
    if not matches:
        print(f"Unknown: {strategy}. Available: {list(strats.keys())}")
        return None
    sname = matches[0]
    gen = strats[sname]
    is_jepa = "JEPA" in sname

    fig, axes = plt.subplots(n_repeats, 4, figsize=(16, 3.5 * n_repeats), facecolor=C_BG)
    if n_repeats == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        f"{sname} — {n_repeats} random draws", fontsize=13, color=C_TEXT, fontweight="bold", y=0.98
    )

    for row in range(n_repeats):
        rng = np.random.default_rng(seed + row)
        (v1f, v1c), (v2f, v2c) = gen(features, coords, rng=rng)
        if v1c is None:
            v1c = np.zeros((len(v1f), 2), dtype=np.int32)
        if v2c is None:
            v2c = np.zeros((len(v2f), 2), dtype=np.int32)
        axes[row, 0].annotate(
            f"seed={seed + row}",
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-40, 0),
            textcoords="offset points",
            fontsize=7,
            color=C_DIM,
            ha="right",
            va="center",
        )
        plot_row(axes[row], coords, v1c, v2c, sname, is_jepa=is_jepa)

    plt.tight_layout(rect=[0.06, 0.02, 1.0, 0.96])
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def visualize_features(features, coords, seed=42, output_path=None):
    """Show per-patch feature change heatmap."""
    # Use augmentor WITHOUT coord transforms so coords match for comparison
    aug = FeatureAugmentor(
        subsample_frac=(0.6, 0.9),
        instance_dropout=0.05,
        feature_noise_std=0.02,
        feature_dropout=0.05,
        local_smooth=LocalFeatureSmooth(k_neighbors=5, alpha_range=(0.05, 0.2)),
        feature_posterize=FeaturePosterize(n_levels=8, prob=0.5),
        # No coord_affine — we need coords to match for comparison
    )
    rng = np.random.default_rng(seed)
    af, ac = aug(features.copy(), coords.copy(), rng=rng)

    orig_set = {tuple(c): i for i, c in enumerate(coords)}
    mo, ma = [], []
    for j, c in enumerate(ac):
        k = tuple(c.tolist())
        if k in orig_set:
            mo.append(orig_set[k])
            ma.append(j)
    if not mo:
        print("No matched patches")
        return None
    mo, ma = np.array(mo), np.array(ma)

    of_ = features[mo].astype(np.float32)
    afu = af[ma].astype(np.float32)
    delta = np.linalg.norm(afu - of_, axis=1) / (np.linalg.norm(of_, axis=1) + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=C_BG)
    norms = np.linalg.norm(features.astype(np.float32), axis=1)
    sc1 = axes[0].scatter(
        coords[:, 0], coords[:, 1], c=norms, s=4, cmap="viridis", alpha=0.7, edgecolors="none"
    )
    axes[0].set_title("Original L2 Norm", fontsize=9, color=C_TEXT)
    plt.colorbar(sc1, ax=axes[0], shrink=0.7)

    an = np.linalg.norm(af.astype(np.float32), axis=1)
    sc2 = axes[1].scatter(
        ac[:, 0], ac[:, 1], c=an, s=4, cmap="viridis", alpha=0.7, edgecolors="none"
    )
    axes[1].set_title(f"Augmented L2 Norm ({len(af)} patches)", fontsize=9, color=C_TEXT)
    plt.colorbar(sc2, ax=axes[1], shrink=0.7)

    sc3 = axes[2].scatter(
        coords[mo, 0],
        coords[mo, 1],
        c=delta,
        s=4,
        cmap="hot",
        alpha=0.7,
        edgecolors="none",
        vmin=0,
        vmax=np.percentile(delta, 95),
    )
    axes[2].set_title("Feature Change (rel. ΔL2)", fontsize=9, color=C_TEXT)
    plt.colorbar(sc3, ax=axes[2], shrink=0.7)

    for ax in axes:
        _style_ax(ax)
    fig.suptitle("Feature-Space Augmentation Effect", fontsize=12, color=C_TEXT, fontweight="bold")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mmap_dir", type=str, required=True)
    p.add_argument("--n_slides", type=int, default=20, help="Number of slides to randomly sample")
    p.add_argument(
        "--slide_idx", type=int, default=None, help="Specific slide index (overrides --n_slides)"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output_dir",
        type=str,
        default="./visualizations/aug_sss",
        help="Directory to save visualizations",
    )
    p.add_argument("--mode", choices=["all", "single", "features"], default="all")
    p.add_argument("--strategy", type=str, default="JEPA")
    p.add_argument("--n_repeats", type=int, default=4)
    p.add_argument("--max_patches", type=int, default=12000, help="Cap patches per slide for viz")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Resolve which slide indices to visualize
    p_mmap = Path(args.mmap_dir)
    idx_arrays = np.load(str(p_mmap / "index_arrays.npz"), allow_pickle=True)
    n_total = len(idx_arrays["lengths"])

    if args.slide_idx is not None:
        slide_indices = [min(args.slide_idx, n_total - 1)]
    else:
        k = min(args.n_slides, n_total)
        slide_indices = sorted(rng.choice(n_total, size=k, replace=False).tolist())

    print(f"Sampling {len(slide_indices)} slide(s) from {n_total} in {args.mmap_dir}")

    for si in slide_indices:
        features, coords, sid = load_mmap_slide(args.mmap_dir, si, idx_arrays=idx_arrays)
        print(f"[{si}] '{sid}': {len(features)} patches, D={features.shape[1]}")

        if len(features) > args.max_patches:
            cap_idx = _spatial_stratified_subsample(
                coords, args.max_patches, grid_size=16,
                rng=np.random.RandomState(args.seed),
            )
            features, coords = features[cap_idx], coords[cap_idx]
            print(f"  Capped to {len(features)} (spatial_stratified)")

        # Sanitize slide id for filename
        safe_sid = str(sid).replace("/", "_").replace("\\", "_").replace(" ", "_")
        tp = f"{sid} — "

        if args.mode == "all":
            out_path = out_dir / f"{safe_sid}_views_all.png"
            visualize_all(features, coords, args.seed, tp, str(out_path))
        elif args.mode == "single":
            out_path = out_dir / f"{safe_sid}_views_{args.strategy}.png"
            visualize_single(
                features, coords, args.strategy, args.n_repeats, args.seed, str(out_path)
            )
        elif args.mode == "features":
            out_path = out_dir / f"{safe_sid}_features.png"
            visualize_features(features, coords, args.seed, str(out_path))

    print(f"Done. {len(slide_indices)} figure(s) saved to {out_dir}")


if __name__ == "__main__":
    main()
