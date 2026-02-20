"""
UMAP visualization of slide-level embeddings.

Produces publication-quality scatter plots of the MIL representation space:
  - Class-colored UMAP with sample counts per class
  - Optional SVM decision boundary overlay
  - Fold-aware: uses OOF embeddings to avoid optimistic artifacts
  - Combined train+test view with marker differentiation

Data flow:
  embeddings_val.npz (from training folds) → standardize → PCA → UMAP → plot

Design notes:
  - Uses OOF (out-of-fold) embeddings by default. Each slide's embedding
    comes from the fold where it was in the validation set, so no slide's
    embedding was produced by a model that trained on that slide.
  - For ensemble models in "live" mode, embeddings are averaged across
    fold models (see ensemble.py).
  - PCA to 20 dims before UMAP is standard practice: removes noise, speeds
    up neighbor graph construction, and makes UMAP more stable.
  - Fixed random_state for deterministic output.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Colorblind-friendly palette (Wong 2011)
DEFAULT_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _resolve_class_names(
    labels: np.ndarray,
    class_names: dict | None,
) -> dict:
    """
    Ensure class_names is a valid {int: str} mapping covering all labels.

    Handles: None, empty dict, partial dict (missing classes), and
    label values not present in the mapping.
    """
    unique_labels = sorted(int(c) for c in np.unique(labels) if c >= 0)

    if not class_names:
        # None or empty dict → auto-generate from labels
        return {c: str(c) for c in unique_labels}

    # Ensure all labels in the data have an entry
    resolved = dict(class_names)
    for c in unique_labels:
        if c not in resolved:
            resolved[c] = str(c)

    return resolved


def _resolve_colors(class_names: dict, colors: dict | None = None) -> dict:
    """Build {class_id: hex_color} using the default palette."""
    if colors:
        # Fill any missing classes
        result = dict(colors)
        for i, c in enumerate(sorted(class_names.keys())):
            if c not in result:
                result[c] = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        return result

    return {
        c: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i, c in enumerate(sorted(class_names.keys()))
    }


# ═════════════════════════════════════════════════════════════════════════════
# Embedding collection from training outputs
# ═════════════════════════════════════════════════════════════════════════════


def load_oof_embeddings(train_dir: str, n_folds: int) -> pd.DataFrame | None:
    """
    Load OOF (out-of-fold) embeddings from training output.

    Each fold saved embeddings_val.npz containing slide_ids and embeddings
    for its validation set. Concatenating gives the full OOF embedding set
    where each slide's embedding comes from the fold that held it out.

    Returns DataFrame with columns: slide_id, fold, embedding (np.ndarray).
    Returns None if no embeddings found.
    """
    train_dir = Path(train_dir)
    records = []

    for fold_idx in range(n_folds):
        emb_path = train_dir / f"fold_{fold_idx}" / "embeddings_val.npz"
        if not emb_path.is_file():
            logger.warning(f"No embeddings_val.npz for fold {fold_idx}")
            continue

        data = np.load(str(emb_path), allow_pickle=True)
        slide_ids = data["slide_ids"]
        embeddings = data["embeddings"]

        for sid, emb in zip(slide_ids, embeddings):
            records.append(
                {
                    "slide_id": str(sid),
                    "fold": fold_idx,
                    "embedding": emb,
                }
            )

    if not records:
        return None

    df = pd.DataFrame(records)
    logger.info(f"Loaded OOF embeddings: {len(df)} slides from {n_folds} folds")
    return df


def load_test_embeddings(train_dir: str, n_folds: int) -> pd.DataFrame | None:
    """
    Load test embeddings from training output.

    For k-fold with a held-out test set, each fold produces
    embeddings_test.npz on the same test set. We average across folds
    to get a consensus embedding per test slide.
    """
    train_dir = Path(train_dir)
    all_folds: dict[str, list[np.ndarray]] = {}

    for fold_idx in range(n_folds):
        emb_path = train_dir / f"fold_{fold_idx}" / "embeddings_test.npz"
        if not emb_path.is_file():
            continue

        data = np.load(str(emb_path), allow_pickle=True)
        slide_ids = data["slide_ids"]
        embeddings = data["embeddings"]

        for sid, emb in zip(slide_ids, embeddings):
            sid = str(sid)
            if sid not in all_folds:
                all_folds[sid] = []
            all_folds[sid].append(emb)

    if not all_folds:
        return None

    records = []
    for sid, emb_list in all_folds.items():
        avg_emb = np.mean(emb_list, axis=0)
        records.append({"slide_id": sid, "fold": -1, "embedding": avg_emb})

    df = pd.DataFrame(records)
    logger.info(f"Loaded test embeddings: {len(df)} slides (averaged across folds)")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# UMAP computation
# ═════════════════════════════════════════════════════════════════════════════


def compute_umap(
    embeddings: np.ndarray,
    n_pca: int = 20,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    seed: int = 42,
) -> np.ndarray:
    """
    Standardize → PCA → UMAP → 2D coordinates.

    Parameters
    ----------
    embeddings : [N, D] array of slide embeddings.
    n_pca : PCA components before UMAP (capped at min(n_pca, D, N-1)).
    n_neighbors : UMAP neighbor count.
    min_dist : UMAP min_dist.
    metric : UMAP distance metric.
    seed : random state for reproducibility.

    Returns
    -------
    [N, 2] UMAP coordinates.
    """
    import umap.umap_ as umap
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Standardize features
    scaler = StandardScaler()
    emb_std = scaler.fit_transform(embeddings)

    # PCA dimensionality reduction
    n_components = min(n_pca, emb_std.shape[1], emb_std.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=seed)
    emb_pca = pca.fit_transform(emb_std)

    var_explained = pca.explained_variance_ratio_.sum()
    logger.info(
        f"PCA: {emb_std.shape[1]}D → {n_components}D ({var_explained:.1%} variance explained)"
    )

    # UMAP projection
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    coords_2d = reducer.fit_transform(emb_pca)

    return coords_2d


# ═════════════════════════════════════════════════════════════════════════════
# UMAP plotting
# ═════════════════════════════════════════════════════════════════════════════


def plot_umap(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    class_names: dict | None = None,
    colors: dict | None = None,
    title: str = "UMAP of Slide-Level Embeddings",
    highlight_slide_ids: list | None = None,
    all_slide_ids: list | None = None,
    save_dpi: int = 200,
    show_decision_boundary: bool = False,
) -> None:
    """
    Plot UMAP scatter with class colors and sample counts.

    Parameters
    ----------
    coords_2d : [N, 2] UMAP coordinates.
    labels : [N] integer labels.
    output_path : where to save the plot.
    class_names : {int: str} mapping. None or {} → auto from labels.
    colors : {int: str} color mapping per class.
    title : plot title.
    highlight_slide_ids : slide IDs to highlight (circled in red).
    all_slide_ids : all slide IDs (for matching highlights).
    show_decision_boundary : fit SVM and show decision regions.
    save_dpi : output DPI.
    """
    import matplotlib.pyplot as plt

    # Resolve class names and colors robustly
    class_names = _resolve_class_names(labels, class_names)
    colors = _resolve_colors(class_names, colors)

    fig, ax = plt.subplots(figsize=(9, 7))

    # SVM decision boundary (behind scatter points)
    if show_decision_boundary:
        # Use curly braces for direct set comprehension
        unique_in_data = sorted({int(c) for c in np.unique(labels) if c >= 0})
        if len(unique_in_data) >= 2:
            _draw_decision_boundary(
                ax,
                coords_2d,
                labels,
                colors,
                unique_in_data,
            )

    # Scatter per class (with counts in legend)
    for cls_id in sorted(class_names.keys()):
        mask = labels == cls_id
        n = int(mask.sum())
        if n == 0:
            continue
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            s=35,
            alpha=0.85,
            color=colors.get(cls_id, "#333333"),
            label=f"{class_names[cls_id]} (n={n})",
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )

    # Highlight specific slides
    if highlight_slide_ids and all_slide_ids:
        id_set = set(highlight_slide_ids)
        highlight_mask = np.array([sid in id_set for sid in all_slide_ids])
        if highlight_mask.any():
            ax.scatter(
                coords_2d[highlight_mask, 0],
                coords_2d[highlight_mask, 1],
                s=120,
                facecolors="none",
                edgecolors="red",
                linewidths=2.0,
                label="Highlighted",
                zorder=5,
            )

    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Clean up axis for publication quality
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    # Legend — only draw if there are labelled artists
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            loc="best",
            fontsize=9,
            framealpha=0.9,
            edgecolor="#cccccc",
        )

    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"UMAP plot → {output_path}")


def _draw_decision_boundary(
    ax,
    coords_2d: np.ndarray,
    labels: np.ndarray,
    colors: dict,
    unique_classes: list[int],
) -> None:
    """
    Fit SVM on 2D UMAP coords and draw decision regions.

    Parameters
    ----------
    ax : matplotlib Axes.
    coords_2d : [N, 2] UMAP coordinates.
    labels : [N] integer class labels.
    colors : {class_id: hex_color} mapping.
    unique_classes : sorted list of class IDs present in the data.
                     Must have len >= 2 (caller should check).
    """
    import matplotlib.colors as mcolors
    from sklearn.svm import SVC

    if len(unique_classes) < 2:
        logger.warning(
            f"Decision boundary requires >= 2 classes, got {len(unique_classes)} — skipping"
        )
        return

    # Filter to samples with valid labels
    valid_mask = np.isin(labels, unique_classes)
    if valid_mask.sum() < 10:
        logger.warning(f"Only {valid_mask.sum()} valid samples — skipping decision boundary")
        return

    X = coords_2d[valid_mask]
    y = labels[valid_mask]

    try:
        svm = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=42,
        )
        svm.fit(X, y)
    except Exception as e:
        logger.warning(f"SVM fitting failed: {e} — skipping decision boundary")
        return

    # Build mesh for contour
    margin = 1.5
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Lighten class colors for background fill
    bg_colors = []
    for cls_id in unique_classes:
        rgb = mcolors.to_rgb(colors.get(cls_id, "#cccccc"))
        # Blend towards white for subtle background
        bg_colors.append(tuple(min(1.0, c * 0.3 + 0.7) for c in rgb))

    # Build contourf levels: one per class boundary
    # levels must bracket each unique class value
    levels = [c - 0.5 for c in unique_classes] + [unique_classes[-1] + 0.5]

    ax.contourf(
        xx,
        yy,
        Z,
        levels=levels,
        colors=bg_colors,
        alpha=0.20,
        zorder=1,
    )

    # Dashed boundary lines between classes
    boundary_levels = [c + 0.5 for c in unique_classes[:-1]]
    if boundary_levels:
        ax.contour(
            xx,
            yy,
            Z,
            levels=boundary_levels,
            colors="#888888",
            linewidths=1.0,
            linestyles="--",
            zorder=2,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Combined train+test UMAP
# ═════════════════════════════════════════════════════════════════════════════


def plot_train_test_umap(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    train_ids: list[str],
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    test_ids: list[str],
    output_path: str,
    class_names: dict | None = None,
    colors: dict | None = None,
    n_pca: int = 20,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    seed: int = 42,
    save_dpi: int = 200,
    title: str = "UMAP Train + Test",
) -> None:
    """
    Fit PCA+UMAP on combined train+test embeddings, plot with markers
    distinguishing train (circles) from test (triangles).
    """
    import matplotlib.pyplot as plt
    import umap.umap_ as umap_lib
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Combine and fit joint embedding
    all_emb = np.vstack([train_embeddings, test_embeddings])
    all_labels = np.concatenate([train_labels, test_labels])
    splits = np.array(["train"] * len(train_ids) + ["test"] * len(test_ids))

    # Resolve names/colors from combined labels
    class_names = _resolve_class_names(all_labels, class_names)
    colors = _resolve_colors(class_names, colors)

    # Standardise → PCA → UMAP (joint fit)
    scaler = StandardScaler()
    n_comp = min(n_pca, all_emb.shape[1], all_emb.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=seed)
    emb_pca = pca.fit_transform(scaler.fit_transform(all_emb))

    reducer = umap_lib.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    all_2d = reducer.fit_transform(emb_pca)

    # Plot with different markers for train/test
    fig, ax = plt.subplots(figsize=(9, 7))

    for cls_id in sorted(class_names.keys()):
        name = class_names[cls_id]
        color = colors.get(cls_id, "#333333")

        for split, marker, sz, alpha in [
            ("train", "o", 30, 0.7),
            ("test", "^", 60, 0.9),
        ]:
            mask = (all_labels == cls_id) & (splits == split)
            if not mask.any():
                continue
            ax.scatter(
                all_2d[mask, 0],
                all_2d[mask, 1],
                s=sz,
                alpha=alpha,
                color=color,
                marker=marker,
                label=f"{name} ({split}, n={mask.sum()})",
                edgecolors="white" if split == "train" else "black",
                linewidths=0.3 if split == "train" else 0.5,
            )

    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=8, framealpha=0.9, edgecolor="#cccccc")

    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Train+test UMAP → {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Save UMAP data
# ═════════════════════════════════════════════════════════════════════════════


def save_umap_data(
    coords_2d: np.ndarray,
    slide_ids: list[str],
    labels: np.ndarray,
    output_path: str,
    metadata_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Save UMAP coordinates + metadata as Parquet for downstream analysis.

    Returns the assembled DataFrame.
    """
    df = pd.DataFrame(
        {
            "slide_id": slide_ids,
            "umap_1": coords_2d[:, 0],
            "umap_2": coords_2d[:, 1],
            "label": labels,
        }
    )

    # Merge metadata if available
    if metadata_df is not None and "slide_id" in metadata_df.columns:
        meta_cols = [c for c in metadata_df.columns if c != "slide_id"]
        if meta_cols:
            df = df.merge(
                metadata_df[["slide_id", *meta_cols]],
                on="slide_id",
                how="left",
            )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(output_path), index=False, engine="pyarrow")
    logger.info(f"UMAP data saved: {len(df)} slides → {output_path}")
    return df
