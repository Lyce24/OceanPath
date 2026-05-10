#!/usr/bin/env python3
"""
Visualize VICReg / LeJEPA dual-view augmentations for SSL pretraining.

Pipeline (oceanpath.ssl.data.augmentation.WSIDualViewAugmentor):
    full N tokens
    -> shared D4 transform (one of 8 rotations/reflections, both views together)
    -> split into two view index sets (controlled by split_overlap)
    -> coordinate crop per view
    -> random token masking per view
    -> hierarchical refill to exactly fixed_n unique tokens per view

VICReg vs LeJEPA: by design, both methods use the SAME augmentation config
(see augmentation.build_augmentor docstring). The method differences are in
the loss / regularizer downstream, not in the dual-view sampler. Showing
them as separate rows here is a sanity check, not a contrast — any visible
difference between the two rows is purely from per-row RNG.

Usage:
    python visualize_augmentations.py --mmap_dir /path/to/mmap
    python visualize_augmentations.py --mmap_dir /path/to/mmap --mode single \\
        --strategy vicreg --n_repeats 4
    python visualize_augmentations.py --mmap_dir /path/to/mmap \\
        --config_yaml configs/pretrain_training/vicreg.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oceanpath.ssl.data.augmentation import (
    WSIDualViewAugmentConfig,
    WSIDualViewAugmentor,
    build_augmentor,
)
from oceanpath.data.mmap_builder import _spatial_stratified_subsample

# =========================================================================
# Colors
# =========================================================================

C_BG = "#1a1d27"
C_GRID = "#2e3345"
C_ALL = "#555b73"
C_VIEW1 = "#6ec1e4"
C_VIEW2 = "#e8845c"
C_OVERLAP = "#b48eed"
C_DROPPED = "#3a3f52"
C_TEXT = "#e2e4ea"
C_DIM = "#9499ad"


# =========================================================================
# Data loading
# =========================================================================


def load_mmap_slide(mmap_dir: str, slide_idx: int = 0, idx_arrays=None):
    p = Path(mmap_dir)

    idx = (
        idx_arrays
        if idx_arrays is not None
        else np.load(str(p / "index_arrays.npz"), allow_pickle=True)
    )

    lengths = idx["lengths"].astype(int)
    slide_idx = min(slide_idx, len(lengths) - 1)

    n = int(lengths[slide_idx])
    sid = str(idx["slide_ids"][slide_idx])

    feat_dim = int(idx["feat_dim"])
    feat_dtype = np.dtype(str(idx["feat_dtype"][0]))
    coord_dtype = np.dtype(str(idx["coord_dtype"][0]))
    coord_dim = int(idx["coord_dim"]) if "coord_dim" in idx else 2

    feat_chunk = int(idx["feat_chunk_ids"][slide_idx])
    feat_offset = int(idx["feat_offsets"][slide_idx])

    feat_mm = np.memmap(
        str(p / f"features_{feat_chunk:03d}.bin"),
        dtype=feat_dtype,
        mode="r",
    )
    feat_start = feat_offset // feat_dtype.itemsize
    features = feat_mm[feat_start : feat_start + n * feat_dim].reshape(n, feat_dim).copy()

    coord_chunk = int(idx["coord_chunk_ids"][slide_idx])
    coord_offset = int(idx["coord_offsets"][slide_idx])

    coord_mm = np.memmap(
        str(p / f"coords_{coord_chunk:03d}.bin"),
        dtype=coord_dtype,
        mode="r",
    )
    coord_start = coord_offset // coord_dtype.itemsize
    coords = coord_mm[coord_start : coord_start + n * coord_dim].reshape(n, coord_dim).copy()

    if coords.shape[1] > 2:
        coords = coords[:, :2]

    return features, coords, sid


# =========================================================================
# Config loading (optional YAML overrides)
# =========================================================================


def _load_yaml_overrides(config_yaml: str | None) -> dict:
    """Read the `augmentation:` block from a pretrain_training/*.yaml.

    Returns a dict suitable for unpacking into build_augmentor as kwargs.
    Empty dict if no path given. Unknown keys are surfaced by
    build_augmentor's own validation.
    """
    if not config_yaml:
        return {}

    try:
        import yaml  # PyYAML
    except ImportError:
        print(
            f"PyYAML not installed — ignoring --config_yaml {config_yaml}; "
            f"using dataclass defaults instead."
        )
        return {}

    p = Path(config_yaml)
    if not p.exists():
        print(f"--config_yaml {config_yaml} not found; using dataclass defaults.")
        return {}

    with p.open() as f:
        cfg = yaml.safe_load(f) or {}

    aug = cfg.get("augmentation", {}) or {}

    # YAML returns lists for tuple-typed fields; the dataclass accepts both.
    return dict(aug)


# =========================================================================
# Strategy building (VICReg + LeJEPA only)
# =========================================================================


def build_all_strategies(yaml_overrides: dict | None = None):
    """Two dual-view rows: VICReg and LeJEPA.

    Both methods are built with the SAME augmentation config — that is the
    design contract enforced by build_augmentor. Keeping them as separate
    rows is a visual sanity check (the two rows should differ only in RNG).
    """
    overrides = dict(yaml_overrides or {})

    return {
        "VICReg dual-view": {
            "method": "vicreg",
            "augmentor": build_augmentor("vicreg", **overrides),
        },
        "LeJEPA dual-view": {
            "method": "lejepa",
            "augmentor": build_augmentor("lejepa", **overrides),
        },
    }


# =========================================================================
# Index matching / overlap helpers
# =========================================================================


def _nearest_original_indices(
    all_coords: np.ndarray,
    view_coords: np.ndarray,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Map view coordinates back to nearest indices in a reference pool."""
    if view_coords is None or len(view_coords) == 0:
        return np.empty(0, dtype=np.int64)

    all_f = all_coords.astype(np.float32, copy=False)
    view_f = view_coords.astype(np.float32, copy=False)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(all_f)
        _, idx = tree.query(view_f, k=1)
        return idx.astype(np.int64)

    except Exception:
        out = []
        for start in range(0, len(view_f), chunk_size):
            q = view_f[start : start + chunk_size]
            d2 = ((q[:, None, :] - all_f[None, :, :]) ** 2).sum(axis=-1)
            out.append(np.argmin(d2, axis=1).astype(np.int64))
        return np.concatenate(out, axis=0)


def _classify_by_index(
    n_all: int,
    v1_idx: np.ndarray,
    v2_idx: np.ndarray,
) -> np.ndarray:
    """0=dropped, 1=view1 only, 2=view2 only, 3=overlap."""
    labels = np.zeros(n_all, dtype=np.int64)

    v1 = np.unique(np.asarray(v1_idx, dtype=np.int64))
    v2 = np.unique(np.asarray(v2_idx, dtype=np.int64))

    if len(v1):
        labels[v1] = 1

    if len(v2):
        overlap = labels[v2] == 1
        labels[v2[overlap]] = 3
        labels[v2[~overlap]] = 2

    return labels


# =========================================================================
# Plotting helpers
# =========================================================================


def _style_ax(ax):
    ax.set_facecolor(C_BG)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.tick_params(colors=C_GRID, labelsize=5)

    for sp in ax.spines.values():
        sp.set_color(C_GRID)


def _scatter(ax, coords, color, size=3, alpha=0.7):
    if coords is None or len(coords) == 0:
        return

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=color,
        s=size,
        alpha=alpha,
        edgecolors="none",
    )


def _shared_xy_limits(*coord_arrays):
    """Compute axis-equal limits across multiple coord arrays."""
    valids = [c for c in coord_arrays if c is not None and len(c) > 0]
    if not valids:
        return None
    all_pts = np.concatenate(valids, axis=0)
    pad = max(1.0, 0.02 * max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])))
    return (
        (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad),
        (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad),
    )


def plot_dual_row(
    axes,
    all_coords: np.ndarray,
    view1_coords: np.ndarray,
    view2_coords: np.ndarray,
):
    """Plot one dual-view strategy row.

    The shared D4 applies before split, so view1 and view2 live in the same
    post-orientation frame as each other but may be rotated/flipped relative
    to the original. The shared xy limits ensure both layouts fit cleanly.
    """
    s = 3

    xy = _shared_xy_limits(all_coords, view1_coords, view2_coords)

    _scatter(axes[0], all_coords, C_ALL, size=s, alpha=0.50)
    axes[0].set_title(f"Original ({len(all_coords)})", fontsize=8, color=C_TEXT, pad=4)

    _scatter(axes[1], all_coords, C_DROPPED, size=s * 0.5, alpha=0.10)
    _scatter(axes[1], view1_coords, C_VIEW1, size=s, alpha=0.70)
    axes[1].set_title(f"View 1 ({len(view1_coords)})", fontsize=8, color=C_VIEW1, pad=4)

    _scatter(axes[2], all_coords, C_DROPPED, size=s * 0.5, alpha=0.10)
    _scatter(axes[2], view2_coords, C_VIEW2, size=s, alpha=0.70)
    axes[2].set_title(f"View 2 ({len(view2_coords)})", fontsize=8, color=C_VIEW2, pad=4)

    if len(view1_coords) > 0 and len(view2_coords) > 0:
        union_coords = np.concatenate([view1_coords, view2_coords], axis=0)
        rounded = np.round(union_coords).astype(np.int64)
        _, uniq_idx = np.unique(rounded, axis=0, return_index=True)
        ref_coords = union_coords[np.sort(uniq_idx)]

        v1_idx = _nearest_original_indices(ref_coords, view1_coords)
        v2_idx = _nearest_original_indices(ref_coords, view2_coords)
        labels = _classify_by_index(len(ref_coords), v1_idx, v2_idx)

        for label, color, alpha in [
            (0, C_DROPPED, 0.15),
            (1, C_VIEW1, 0.70),
            (2, C_VIEW2, 0.70),
            (3, C_OVERLAP, 0.85),
        ]:
            m = labels == label
            if m.any():
                _scatter(axes[3], ref_coords[m], color, size=s, alpha=alpha)

        n_overlap = int((labels == 3).sum())
        n_v1_only = int((labels == 1).sum())
        n_v2_only = int((labels == 2).sum())
        axes[3].set_title(
            f"Overlap: {n_overlap}  V1-only: {n_v1_only}  V2-only: {n_v2_only}",
            fontsize=7,
            color=C_DIM,
            pad=4,
        )
    else:
        axes[3].set_title("Overlap: n/a", fontsize=7, color=C_DIM, pad=4)

    for ax in axes:
        _style_ax(ax)
        if xy is not None:
            ax.set_xlim(*xy[0])
            ax.set_ylim(*xy[1])
            ax.invert_yaxis()


# =========================================================================
# Main visualizations
# =========================================================================


def _row_subtitle(cfg: WSIDualViewAugmentConfig) -> str:
    """One-line subtitle showing the knobs that actually shape each view."""
    return (
        f"d4={cfg.d4_mode}  split_overlap={cfg.split_overlap:.2f}  "
        f"crop={cfg.crop_area_range}  mask={cfg.mask_ratio_range}  "
        f"fixed_n={cfg.fixed_n}"
    )


def visualize_all(
    features: np.ndarray,
    coords: np.ndarray,
    seed: int = 42,
    title_prefix: str = "",
    output_path: str | None = None,
    yaml_overrides: dict | None = None,
):
    strategies = build_all_strategies(yaml_overrides)

    n_rows = len(strategies)
    fig, axes = plt.subplots(
        n_rows,
        4,
        figsize=(16, 3.5 * n_rows),
        facecolor=C_BG,
    )

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"{title_prefix}VICReg vs LeJEPA augmentations\n"
        f"({len(features)} patches, D={features.shape[1]}; "
        f"shared D4 + split + crop + mask + hierarchical refill — identical config by design)",
        fontsize=13,
        color=C_TEXT,
        fontweight="bold",
        y=0.985,
    )

    for row, (name, entry) in enumerate(strategies.items()):
        rng = np.random.default_rng(seed + row)
        aug: WSIDualViewAugmentor = entry["augmentor"]

        sub = _row_subtitle(aug.cfg)

        axes[row, 0].annotate(
            f"{name}\n({sub})",
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-65, 0),
            textcoords="offset points",
            fontsize=7,
            color=C_TEXT,
            fontweight="bold",
            ha="right",
            va="center",
        )

        (v1f, v1c), (v2f, v2c) = aug(features, coords, rng=rng)

        if v1c is None:
            v1c = np.zeros((len(v1f), 2), dtype=np.float32)
        if v2c is None:
            v2c = np.zeros((len(v2f), 2), dtype=np.float32)

        plot_dual_row(
            axes[row],
            all_coords=coords,
            view1_coords=v1c,
            view2_coords=v2c,
        )

    legend = [
        mpatches.Patch(color=C_VIEW1, label="View 1"),
        mpatches.Patch(color=C_VIEW2, label="View 2"),
        mpatches.Patch(color=C_OVERLAP, label="Overlap"),
        mpatches.Patch(color=C_DROPPED, label="Not selected"),
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

    plt.tight_layout(rect=[0.10, 0.035, 1.0, 0.965])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        print(f"Saved: {output_path}")

    plt.close(fig)
    return fig


def _resolve_strategy(strategy: str, strategies: dict):
    """Resolve a `--strategy` string against the strategy dict."""

    def norm(s: str) -> str:
        return s.lower().replace("-", "_").replace(" ", "_")

    qry = norm(strategy)

    exact_method = [k for k, v in strategies.items() if norm(v["method"]) == qry]
    if exact_method:
        return exact_method[0]

    exact_name = [k for k in strategies if norm(k) == qry]
    if exact_name:
        return exact_name[0]

    subs = [k for k, v in strategies.items() if qry in norm(k) or qry in norm(v["method"])]
    if len(subs) == 1:
        return subs[0]
    if len(subs) > 1:
        print(f"Ambiguous strategy '{strategy}'. Matches: {subs}")
        return None

    return None


def visualize_single(
    features: np.ndarray,
    coords: np.ndarray,
    strategy: str = "vicreg",
    n_repeats: int = 4,
    seed: int = 42,
    output_path: str | None = None,
    yaml_overrides: dict | None = None,
):
    strategies = build_all_strategies(yaml_overrides)

    name = _resolve_strategy(strategy, strategies)

    if name is None:
        print(f"Unknown strategy: {strategy}")
        print("Available:")
        for k, v in strategies.items():
            print(f"  - {k} / method={v['method']}")
        return None

    entry = strategies[name]
    aug: WSIDualViewAugmentor = entry["augmentor"]

    fig, axes = plt.subplots(
        n_repeats,
        4,
        figsize=(16, 3.5 * n_repeats),
        facecolor=C_BG,
    )

    if n_repeats == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"{name} — {n_repeats} random draws  ({_row_subtitle(aug.cfg)})",
        fontsize=12,
        color=C_TEXT,
        fontweight="bold",
        y=0.985,
    )

    for row in range(n_repeats):
        rng = np.random.default_rng(seed + row)

        axes[row, 0].annotate(
            f"seed={seed + row}",
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-45, 0),
            textcoords="offset points",
            fontsize=8,
            color=C_DIM,
            ha="right",
            va="center",
        )

        (v1f, v1c), (v2f, v2c) = aug(features, coords, rng=rng)

        if v1c is None:
            v1c = np.zeros((len(v1f), 2), dtype=np.float32)
        if v2c is None:
            v2c = np.zeros((len(v2f), 2), dtype=np.float32)

        plot_dual_row(
            axes[row],
            all_coords=coords,
            view1_coords=v1c,
            view2_coords=v2c,
        )

    plt.tight_layout(rect=[0.06, 0.025, 1.0, 0.96])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        print(f"Saved: {output_path}")

    plt.close(fig)
    return fig


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mmap_dir", type=str, required=True)
    parser.add_argument("--n_slides", type=int, default=100)
    parser.add_argument("--slide_idx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualizations/aug",
    )

    parser.add_argument(
        "--mode",
        choices=["all", "single"],
        default="all",
        help=("all: VICReg + LeJEPA rows side by side. single: one method, n_repeats rows."),
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="vicreg",
        choices=["vicreg", "lejepa"],
        help="Strategy for --mode single.",
    )

    parser.add_argument("--n_repeats", type=int, default=4)

    parser.add_argument(
        "--max_patches",
        type=int,
        default=24000,
        help="Cap patches before visualization for figure clarity / speed.",
    )

    parser.add_argument(
        "--config_yaml",
        type=str,
        default=None,
        help=(
            "Optional path to a pretrain_training/*.yaml. The 'augmentation:' "
            "block is unpacked as overrides to build_augmentor — same path the "
            "DataModule uses. Default: WSIDualViewAugmentConfig dataclass defaults."
        ),
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    yaml_overrides = _load_yaml_overrides(args.config_yaml)
    if yaml_overrides:
        print(f"Loaded augmentation overrides from {args.config_yaml}: {yaml_overrides}")

    p_mmap = Path(args.mmap_dir)
    idx_arrays = np.load(str(p_mmap / "index_arrays.npz"), allow_pickle=True)
    n_total = len(idx_arrays["lengths"])

    if args.slide_idx is not None:
        slide_indices = [min(args.slide_idx, n_total - 1)]
    else:
        k = min(args.n_slides, n_total)
        slide_indices = sorted(rng.choice(n_total, size=k, replace=False).tolist())

    print(f"Sampling {len(slide_indices)} slide(s) from {n_total} in {args.mmap_dir}")
    print("Visualization design:")
    print("  - augmentor pipeline: shared D4 -> split -> crop -> mask -> hierarchical refill")
    print("  - VICReg and LeJEPA share the same augmentation config by design")
    print("  - row-to-row differences are RNG only; downstream loss differs, not augmentation")

    for si in slide_indices:
        features, coords, sid = load_mmap_slide(
            args.mmap_dir,
            si,
            idx_arrays=idx_arrays,
        )

        print(f"[{si}] '{sid}': {len(features)} patches, D={features.shape[1]}")

        if len(features) > args.max_patches:
            cap_idx = np.sort(rng.permutation(len(features))[: args.max_patches])
            features = features[cap_idx]
            coords = coords[cap_idx]
            print(f"  Capped to {len(features)} patches for visualization")

        safe_sid = str(sid).replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")

        title_prefix = f"{sid} — "

        if args.mode == "all":
            out_path = out_dir / f"{safe_sid}_aug_all.png"
            visualize_all(
                features=features,
                coords=coords,
                seed=args.seed,
                title_prefix=title_prefix,
                output_path=str(out_path),
                yaml_overrides=yaml_overrides,
            )

        elif args.mode == "single":
            safe_strategy = args.strategy.replace("/", "_").replace(" ", "_")
            out_path = out_dir / f"{safe_sid}_aug_{safe_strategy}.png"
            visualize_single(
                features=features,
                coords=coords,
                strategy=args.strategy,
                n_repeats=args.n_repeats,
                seed=args.seed,
                output_path=str(out_path),
                yaml_overrides=yaml_overrides,
            )

    print(f"Done. {len(slide_indices)} figure(s) saved to {out_dir}")


if __name__ == "__main__":
    main()
