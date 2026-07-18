"""Coverage-QC visualization for mmap builds.

The mmap build (``oceanpath.storage.build_mmap``) emits coverage *data* — per
slide, which patches were kept vs. discarded when a bag was capped. This module
turns that data into diagnostic images:

- :func:`build_coverage_grid` / :func:`infer_patch_size` compute the uint8
  coverage grid (pure numpy, no plotting).
- :func:`render_coverage_thumbnail` / :func:`render_summary_montage` render one
  slide / a montage of the worst slides.
- :func:`write_coverage_report` is the single entry point the build calls to
  write the stats CSV, per-slide thumbnails, and the montage.

matplotlib is imported lazily inside the rendering functions so that importing
this module (e.g. for :func:`build_coverage_grid`) stays cheap and never forces
a plotting dependency on the build's hot path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
_coverage_cmap: Any | None = None


# ── Coverage grid (pure data) ───────────────────────────────────────────────────


def infer_patch_size(coords: np.ndarray) -> int:
    """
    Infer the patch stride from coordinate differences.

    Looks at the most common nonzero difference between adjacent x-coords
    when sorted. Falls back to 256 if detection fails.
    """
    if len(coords) < 2:
        return 256

    order = np.lexsort((coords[:, 1], coords[:, 0]))
    sorted_coords = coords[order]

    dx = np.diff(sorted_coords[:, 0])
    dx_nonzero = dx[dx > 0]

    if len(dx_nonzero) == 0:
        dy = np.diff(sorted_coords[:, 1])
        dy_nonzero = dy[dy > 0]
        if len(dy_nonzero) == 0:
            return 256
        vals, counts = np.unique(dy_nonzero, return_counts=True)
        return int(vals[counts.argmax()])

    vals, counts = np.unique(dx_nonzero, return_counts=True)
    return int(vals[counts.argmax()])


def build_coverage_grid(
    all_coords: np.ndarray,
    kept_coords: np.ndarray | None,
    patch_size: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Build a spatial coverage grid from coordinates.

    Parameters
    ----------
    all_coords : (N_total, 2)
        All patch positions from original H5.
    kept_coords : (N_kept, 2) or None
        Patch positions stored in mmap. If None, all patches are kept.
    patch_size : int or None
        Inferred if None.

    Returns
    -------
    grid : np.ndarray, dtype uint8
        2D grid where 0=background, 1=discarded, 2=kept.
    patch_size : int
        The patch stride used.
    """
    if patch_size is None:
        patch_size = infer_patch_size(all_coords)

    xy_min = all_coords.min(axis=0)

    # Vectorized grid coordinate computation
    all_gx = ((all_coords[:, 0] - xy_min[0]) // patch_size).astype(np.intp)
    all_gy = ((all_coords[:, 1] - xy_min[1]) // patch_size).astype(np.intp)

    grid_w = int(all_gx.max()) + 1
    grid_h = int(all_gy.max()) + 1
    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

    if kept_coords is None:
        # All kept — vectorized scatter
        grid[all_gy, all_gx] = 2
    else:
        # Mark all as discarded, then overwrite kept — fully vectorized
        grid[all_gy, all_gx] = 1
        kept_gx = ((kept_coords[:, 0] - xy_min[0]) // patch_size).astype(np.intp)
        kept_gy = ((kept_coords[:, 1] - xy_min[1]) // patch_size).astype(np.intp)
        grid[kept_gy, kept_gx] = 2

    return grid, patch_size


# ── Rendering (matplotlib) ──────────────────────────────────────────────────────


def get_coverage_cmap():
    """
    Return a singleton colormap for coverage visualization.

    Creates it once and caches — avoids matplotlib's colormap registry
    leak when ListedColormap() is called repeatedly.
    """
    global _coverage_cmap
    if _coverage_cmap is None:
        from matplotlib.colors import ListedColormap

        _coverage_cmap = ListedColormap(["#f8f9fa", "#ef4444", "#22c55e"], name="_coverage_qc")
    return _coverage_cmap


def render_coverage_thumbnail(
    grid: np.ndarray,
    slide_id: str,
    n_total: int,
    n_kept: int,
    cmap=None,
    max_thumb_size: int = 800,
    dpi: int = 100,
):
    """
    Render a spatial coverage map for one slide.

    Parameters
    ----------
    grid : (H, W) uint8
        Coverage grid (0=bg, 1=discarded, 2=kept).
    slide_id : str
    n_total : int
    n_kept : int
    cmap : matplotlib colormap, optional
        Reuse a single colormap across calls to avoid registry leak.
        If None, uses the shared singleton.
    max_thumb_size : int
    dpi : int

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if cmap is None:
        cmap = get_coverage_cmap()

    n_discarded = n_total - n_kept
    coverage_pct = 100 * n_kept / n_total if n_total > 0 else 0
    was_capped = n_kept < n_total

    grid_h, grid_w = grid.shape
    aspect = grid_h / grid_w if grid_w > 0 else 1
    fig_w = min(max_thumb_size / dpi, 8)
    fig_h = fig_w * aspect

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, interpolation="nearest", origin="upper")
    ax.set_axis_off()

    ax.set_title(
        f"{slide_id}\n"
        f"Total: {n_total:,}  Kept: {n_kept:,}  Discarded: {n_discarded:,}  "
        f"Coverage: {coverage_pct:.1f}%",
        fontsize=8,
        fontfamily="monospace",
        pad=6,
    )

    legend_elements = [
        Patch(facecolor="#22c55e", edgecolor="none", label=f"Kept ({n_kept:,})"),
    ]
    if was_capped:
        legend_elements.append(
            Patch(facecolor="#ef4444", edgecolor="none", label=f"Discarded ({n_discarded:,})")
        )
    legend_elements.append(Patch(facecolor="#f8f9fa", edgecolor="#ddd", label="Background"))
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=6,
        framealpha=0.9,
        handlelength=1,
        handleheight=1,
    )

    fig.tight_layout()
    return fig


def render_summary_montage(
    stats: list[dict],
    grids: dict[str, np.ndarray],
    output_path: Path,
    n_worst: int = 20,
) -> None:
    """
    Render a montage of the N slides with lowest coverage.

    Parameters
    ----------
    stats : list[dict]
        Per-slide coverage statistics.
    grids : dict[str, np.ndarray]
        Pre-built coverage grids keyed by slide_id (only capped slides needed).
    output_path : Path
        Where to save the montage PNG.
    n_worst : int
        Number of worst-coverage slides to include.
    """
    import matplotlib.pyplot as plt

    capped = [s for s in stats if s["was_capped"]]
    capped.sort(key=lambda s: s["coverage_pct"])
    worst = capped[:n_worst]

    if not worst:
        logger.info("No capped slides — skipping summary montage.")
        return

    n_cols = min(5, len(worst))
    n_rows = (len(worst) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), dpi=100)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    cmap = get_coverage_cmap()

    for plot_i, s in enumerate(worst):
        row, col = divmod(plot_i, n_cols)
        ax = axes[row, col]

        slide_id = s["slide_id"]
        if slide_id in grids:
            ax.imshow(
                grids[slide_id],
                cmap=cmap,
                vmin=0,
                vmax=2,
                interpolation="nearest",
                origin="upper",
            )
            ax.set_title(
                f"{Path(slide_id).name}\n{s['coverage_pct']:.1f}% kept "
                f"({s['n_kept']:,}/{s['n_total']:,})",
                fontsize=7,
                fontfamily="monospace",
            )
        else:
            ax.text(
                0.5,
                0.5,
                "Grid not\navailable",
                ha="center",
                va="center",
                fontsize=8,
            )
            ax.set_title(f"{Path(slide_id).name}", fontsize=7)

        ax.set_axis_off()

    # Hide unused axes
    for plot_i in range(len(worst), n_rows * n_cols):
        row, col = divmod(plot_i, n_cols)
        axes[row, col].set_axis_off()

    fig.suptitle(
        "Lowest-Coverage Slides (most patches discarded)\nGreen = kept  |  Red = discarded",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Summary montage saved: %s", output_path)


# ── Report writer (the single entry point the build calls) ──────────────────────


def write_coverage_report(
    *,
    qc_dir: Path,
    thumb_dir: Path,
    qc_stats: list[dict],
    qc_grids: dict[str, np.ndarray],
    montage_n: int = 20,
    render_batch: int = 50,
    render_thumbnails: bool = True,
    render_montage: bool = True,
    only_capped: bool = True,
) -> None:
    """Write the coverage QC report: stats CSV, per-slide thumbnails, montage.

    ``qc_stats`` and ``qc_grids`` are the *data* collected during the mmap build
    (grids come from :func:`build_coverage_grid`). This function owns all of the
    matplotlib rendering, so the storage build path stays free of plotting code.

    Rendering is done in batches with explicit cleanup because matplotlib leaks
    ~200-500 KB per figure even after ``close()``.
    """
    import csv as csv_mod
    import gc

    if not qc_stats:
        return

    qc_dir.mkdir(parents=True, exist_ok=True)

    # ── Stats CSV ─────────────────────────────────────────────────────────
    csv_path = qc_dir / "coverage_stats.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=qc_stats[0].keys())
        writer.writeheader()
        writer.writerows(qc_stats)

    capped = [s for s in qc_stats if s["was_capped"]]
    if not capped:
        logger.info("Coverage QC: no slides were capped.")
        if only_capped:
            logger.info("Coverage QC saved to %s", qc_dir)
            return
    else:
        coverages = [s["coverage_pct"] for s in capped]
        logger.info(
            "Coverage QC: %d/%d slides capped — mean=%.1f%%, min=%.1f%%, median=%.1f%%",
            len(capped),
            len(qc_stats),
            float(np.mean(coverages)),
            float(np.min(coverages)),
            float(np.median(coverages)),
        )

    # ── Batch-render thumbnails for capped slides ─────────────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib not installed — skipping coverage thumbnails. "
            "Install with: pip install matplotlib"
        )
        return

    from tqdm import tqdm

    cmap = get_coverage_cmap()
    render_stats = capped if only_capped else qc_stats
    stats_by_id = {s["slide_id"]: s for s in render_stats}
    n_rendered = 0

    if render_thumbnails:
        thumb_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Rendering %d coverage thumbnails (batch size %d) ...",
            len(qc_grids),
            render_batch,
        )

        for slide_id, grid in tqdm(
            qc_grids.items(),
            desc="Rendering thumbnails",
            unit="slide",
            total=len(qc_grids),
        ):
            stat = stats_by_id.get(slide_id)
            if stat is None:
                continue

            fig = render_coverage_thumbnail(
                grid=grid,
                slide_id=slide_id,
                n_total=stat["n_total"],
                n_kept=stat["n_kept"],
                cmap=cmap,
            )
            thumb_path = thumb_dir / f"{slide_id.replace('/', '_')}.png"
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(thumb_path), bbox_inches="tight", dpi=100)

            fig.clf()
            plt.close(fig)
            del fig

            n_rendered += 1
            if n_rendered % render_batch == 0:
                plt.close("all")
                gc.collect()

    plt.close("all")
    gc.collect()
    if render_thumbnails:
        logger.info("Rendered %d thumbnails.", n_rendered)

    # ── Montage from cached grids ─────────────────────────────────────────
    if render_montage:
        render_summary_montage(
            stats=qc_stats,
            grids=qc_grids,
            output_path=qc_dir / "coverage_summary.png",
            n_worst=montage_n,
        )
    plt.close("all")
    gc.collect()

    logger.info("Coverage QC saved to %s", qc_dir)
