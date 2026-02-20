"""
Attention heatmap visualization for MIL models.

Produces publication-quality attention overlays on WSI thumbnails:
  1. Alpha compositing — tissue stays crisp outside attention regions
  2. Perceptually uniform colormap (inferno) with transparency ramp
  3. Two-panel layout: Raw WSI | Blended Overlay + Top-k boxes
  4. Top-k patches shown as paired grid: patch crop | location on slide
  5. Coordinate scatter fallback when raw WSI is unavailable

Supports both single models and ensemble models (averaged attention via
EnsembleWrapper, which normalises per-fold before averaging).

Coordinate source priority (matching the original notebook):
  1. H5 files at {feature_h5_dir}/{slide_id}.h5
     → "coords" dataset  +  coords.attrs["patch_size_level0"]
     This is the primary path; TRIDENT writes patch_size_level0 at
     extraction time, making it the most reliable source.
  2. Memmap dataset — fallback when H5 files are unavailable.

Data flow:
  H5 coords + memmap features → model forward (return_attention=True)
  → normalize attention → overlay on slide thumbnail → save
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# H5 coordinate loading (mirrors original notebook exactly)
# ═════════════════════════════════════════════════════════════════════════════


def load_h5_coords(
    h5_path: str,
    h5_coord_key: str = "coords",
) -> tuple[np.ndarray, dict]:
    """
    Load coordinates and their attributes from a TRIDENT H5 feature file.

    Mirrors the original notebook:
    >>> with h5py.File(file_path, "r") as f:
    ...     coords = f["coords"][:]
    ...     coords_attrs = dict(f["coords"].attrs)

    TRIDENT writes each slide's H5 with:
      - "features" dataset: [N, D] patch embeddings
      - "coords" dataset:   [N, 2] level-0 pixel coordinates (x, y)
      - "coords".attrs:     dict with patch_size_level0, mag, etc.

    Parameters
    ----------
    h5_path : path to {slide_id}.h5.
    h5_coord_key : HDF5 dataset key for coordinates.

    Returns
    -------
    coords : [N, 2] int32 array of (x, y) level-0 pixel coordinates.
    coords_attrs : dict of coordinate attributes (includes patch_size_level0).
    """
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        if h5_coord_key not in f:
            raise KeyError(f"Key '{h5_coord_key}' not found in {h5_path}")
        coords = f[h5_coord_key][:]
        coords_attrs = dict(f[h5_coord_key].attrs)

    # Handle (1, N, 2) → (N, 2)
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords.squeeze(0)

    return coords.astype(np.int32), coords_attrs


def get_patch_size_from_h5(
    feature_h5_dir: str,
    slide_ids: list[str],
) -> int | None:
    """
    Read patch_size_level0 from the first available H5 file's coords attrs.

    This is the most reliable source because TRIDENT writes it at extraction
    time, recording the exact level-0 pixel footprint of each patch.

    Returns None if no H5 files contain the attribute.
    """
    h5_dir = Path(feature_h5_dir)
    if not h5_dir.is_dir():
        return None

    for sid in slide_ids:
        h5_path = h5_dir / f"{sid}.h5"
        if not h5_path.is_file():
            continue
        try:
            _, attrs = load_h5_coords(str(h5_path))
            ps = attrs.get("patch_size_level0")
            if ps is not None:
                logger.info(f"patch_size_level0={int(ps)} from H5 coords attrs ({h5_path.name})")
                return int(ps)
        except Exception as e:
            logger.debug(f"Could not read attrs from {h5_path}: {e}")

    return None


# ═════════════════════════════════════════════════════════════════════════════
# Attention normalisation
# ═════════════════════════════════════════════════════════════════════════════


def normalize_attention(
    attn: np.ndarray,
    clip_percentile: tuple[float, float] = (1, 99),
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize attention to [0, 1] with percentile clipping.

    Parameters
    ----------
    attn : 1-D array of attention scores (pre- or post-softmax).
    clip_percentile : percentile range for winsorising outliers.
    eps : numerical stability constant.

    Returns
    -------
    Normalized attention in [0, 1] as float32.
    """
    attn = np.asarray(attn, dtype=np.float64).ravel()
    if clip_percentile is not None:
        lo, hi = np.percentile(attn, list(clip_percentile))
        attn = np.clip(attn, lo, hi)
    a_min, a_max = attn.min(), attn.max()
    if (a_max - a_min) < eps:
        return np.zeros_like(attn, dtype=np.float32)
    return ((attn - a_min) / (a_max - a_min + eps)).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# 2-D overlay construction
# ═════════════════════════════════════════════════════════════════════════════


def build_attention_overlay(
    scores: np.ndarray,
    coords: np.ndarray,
    patch_size_level0: int,
    canvas_size: tuple[int, int],
    downsample: float,
) -> np.ndarray:
    """
    Build a 2-D attention heatmap from per-patch scores.

    Mirrors the original notebook's ``create_overlay`` but uses
    (width, height) canvas_size convention consistent with PIL.

    Parameters
    ----------
    scores : [N] normalized attention scores in [0, 1].
    coords : [N, 2] level-0 pixel coordinates (x, y).
    patch_size_level0 : patch size in level-0 pixels.
    canvas_size : (width, height) of the output canvas.
    downsample : downsample factor from level-0 to canvas level.

    Returns
    -------
    overlay : [H, W] float32 array, NaN where no tissue.
    """
    scale = 1.0 / downsample
    ps = int(np.ceil(patch_size_level0 * scale))
    W, H = canvas_size

    overlay = np.zeros((H, W), dtype=np.float64)
    counter = np.zeros((H, W), dtype=np.uint16)

    for idx in range(len(scores)):
        x0 = int(coords[idx, 0] * scale)
        y0 = int(coords[idx, 1] * scale)
        x1 = min(x0 + ps, W)
        y1 = min(y0 + ps, H)
        if x1 <= x0 or y1 <= y0:
            continue
        overlay[y0:y1, x0:x1] += scores[idx]
        counter[y0:y1, x0:x1] += 1

    tissue_mask = counter > 0
    overlay[tissue_mask] /= counter[tissue_mask]
    overlay[~tissue_mask] = np.nan

    return overlay.astype(np.float32)


def smooth_overlay(overlay: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur the overlay while preserving NaN background."""
    nan_mask = np.isnan(overlay)
    filled = np.nan_to_num(overlay, nan=0.0)
    blurred = cv2.GaussianBlur(filled, (0, 0), sigmaX=sigma, sigmaY=sigma)
    blurred[nan_mask] = np.nan
    return blurred


# ═════════════════════════════════════════════════════════════════════════════
# Colormap + blending
# ═════════════════════════════════════════════════════════════════════════════


def _build_transparent_cmap(
    base_name: str = "inferno",
    n_bins: int = 256,
    transparent_bins: int = 40,
    alpha_cap: float = 0.85,
):
    """Build a colormap where the lowest scores fade to transparent."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    base = plt.get_cmap(base_name)
    colors = base(np.linspace(0, 1, n_bins))
    colors[:transparent_bins, 3] = np.linspace(0, alpha_cap, transparent_bins)
    colors[transparent_bins:, 3] = alpha_cap
    return mcolors.ListedColormap(colors)


def apply_colormap_rgba(
    overlay: np.ndarray,
    cmap_name: str = "inferno",
    transparent_bins: int = 40,
    alpha_cap: float = 0.85,
):
    """
    Apply a transparent colormap to the overlay.

    Returns (rgba_uint8_array, ScalarMappable for colorbar).
    """
    import matplotlib.pyplot as plt

    cmap = _build_transparent_cmap(cmap_name, 256, transparent_bins, alpha_cap)

    rgba = np.zeros((*overlay.shape, 4), dtype=np.uint8)
    valid = ~np.isnan(overlay)
    rgba[valid] = (cmap(overlay[valid]) * 255).astype(np.uint8)

    mappable = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    mappable.set_array(overlay[valid])
    return rgba, mappable


def alpha_blend(base_rgb, overlay_rgba: np.ndarray):
    """Composite RGBA heatmap onto RGB tissue image via PIL alpha compositing."""
    from PIL import Image

    heat = Image.fromarray(overlay_rgba)
    if not isinstance(base_rgb, Image.Image):
        base_rgb = Image.fromarray(base_rgb)
    base = base_rgb.convert("RGBA")
    return Image.alpha_composite(base, heat).convert("RGB")


# ═════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═════════════════════════════════════════════════════════════════════════════


def draw_topk_boxes(
    img,
    topk_indices: np.ndarray,
    coords: np.ndarray,
    scores: np.ndarray,
    patch_size_level0: int,
    downsample: float,
    outline_color: str = "white",
    line_width: int = 2,
):
    """Draw numbered bounding boxes on img for top-k patches."""
    from PIL import Image, ImageDraw

    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.copy()
    draw = ImageDraw.Draw(img)
    scale = 1.0 / downsample
    ps = int(patch_size_level0 * scale)

    for rank, i in enumerate(topk_indices):
        cx = int(coords[i, 0] * scale)
        cy = int(coords[i, 1] * scale)
        draw.rectangle(
            [cx, cy, cx + ps, cy + ps],
            outline=outline_color,
            width=line_width,
        )
        draw.text((cx + 3, cy + 2), str(rank + 1), fill=outline_color)
    return img


# ═════════════════════════════════════════════════════════════════════════════
# Slide thumbnail loading
# ═════════════════════════════════════════════════════════════════════════════


def load_slide_thumbnail(
    slide_path: str,
    vis_level: int = 2,
    target_size: tuple[int, int] | None = None,
):
    """
    Load a low-resolution thumbnail from the WSI.

    Tries SDPC first for .sdpc files, then OpenSlide, then TRIDENT fallback.
    Returns (PIL.Image, downsample_factor, level_used) or (None, None, None).
    """
    from PIL import Image

    slide_path = str(slide_path)
    ext = Path(slide_path).suffix.lower()

    # SDPC files → TRIDENT reader
    if ext == ".sdpc":
        return _load_sdpc_thumbnail(slide_path, vis_level, target_size)

    # Standard formats → OpenSlide
    try:
        import openslide

        slide = openslide.OpenSlide(slide_path)
        level = min(vis_level, slide.level_count - 1)
        dims = slide.level_dimensions[level]
        ds = slide.level_downsamples[level]
        img = slide.read_region((0, 0), level, dims).convert("RGB")
        if target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        return img, ds, level
    except Exception:
        pass

    # Fallback for any format
    try:
        return _load_sdpc_thumbnail(slide_path, vis_level, target_size)
    except Exception as e:
        logger.warning(f"Cannot open slide {slide_path}: {e}")
        return None, None, None


def _load_sdpc_thumbnail(slide_path, vis_level, target_size):
    from PIL import Image
    from trident import SDPCWSI

    slide = SDPCWSI(slide_path=slide_path, lazy_init=False)
    level = min(vis_level, len(slide.level_dimensions) - 1)
    ds = slide.level_downsamples[level]
    dims = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, dims).convert("RGB")
    if target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    return img, ds, level


# ═════════════════════════════════════════════════════════════════════════════
# Slide directory scanning
# ═════════════════════════════════════════════════════════════════════════════


def discover_slides_in_dir(
    slide_dir: str,
    extensions: list[str],
) -> dict[str, str]:
    """
    Scan a directory for WSI files and build {slide_id: full_path} mapping.

    Mirrors the original notebook where slides were discovered from a
    flat directory. Searches non-recursively for files matching the
    given extensions.

    Parameters
    ----------
    slide_dir : directory to scan.
    extensions : list of extensions to match (e.g., [".svs", ".sdpc"]).

    Returns
    -------
    dict mapping slide_id (stem) → absolute file path.
    """
    slide_dir = Path(slide_dir)
    if not slide_dir.is_dir():
        logger.warning(f"Slide directory not found: {slide_dir}")
        return {}

    ext_set = {e.lower() for e in extensions}
    found = {}

    for p in sorted(slide_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in ext_set:
            found[p.stem] = str(p)

    logger.info(f"Discovered {len(found)} WSIs in {slide_dir}")
    return found


# ═════════════════════════════════════════════════════════════════════════════
# Main public API
# ═════════════════════════════════════════════════════════════════════════════


def visualize_slide_attention(
    slide_id: str,
    attention: np.ndarray,
    coords: np.ndarray,
    pred_class: int,
    pred_prob: float,
    true_label: int,
    patch_size_level0: int,
    output_dir: str,
    slide_path: str | None = None,
    class_names: dict | None = None,
    vis_level: int = 2,
    top_k: int = 10,
    cmap_name: str = "inferno",
    blur_sigma_mult: float = 0.4,
    transparent_bins: int = 40,
    alpha_cap: float = 0.85,
    save_dpi: int = 200,
    n_models: int = 1,
) -> dict:
    """
    End-to-end attention visualisation for a single slide.

    Produces:
      - {slide_id}_attention_overview.png: two-panel (raw WSI | blended + top-k)
      - {slide_id}_blended_attention.png: standalone blended image
      - top_patches/: individual top-k patch crops with location context

    Falls back to coordinate scatter plot if no raw WSI is available.

    Parameters
    ----------
    slide_id : slide identifier.
    attention : [N] raw or softmax attention (will be renormalized to [0,1]).
    coords : [N, 2] level-0 pixel coordinates (x, y).
    pred_class, pred_prob, true_label : prediction info for annotation.
    patch_size_level0 : patch size in level-0 pixels (from H5 coords attrs).
    output_dir : where to save.
    slide_path : full path to WSI file (optional; without it, coord-scatter only).
    class_names : {int: str} mapping for labels.
    n_models : number of models in ensemble (for annotation text).

    Returns
    -------
    dict with output file paths and top-k patch info.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = {}

    # Build annotation text
    pred_name = class_names.get(pred_class, str(pred_class))
    true_name = class_names.get(true_label, str(true_label))
    ensemble_tag = f" (ensemble, {n_models} models)" if n_models > 1 else ""
    annotation = f"True: {true_name}  |  Pred: {pred_name} ({pred_prob:.4f}){ensemble_tag}"

    # Normalize attention → [0, 1]
    scores = normalize_attention(attention, clip_percentile=(1, 99))

    # Top-k patch indices (descending score)
    topk_indices = np.argsort(scores)[-top_k:][::-1]
    topk_info = [
        {
            "rank": r + 1,
            "index": int(i),
            "score": float(scores[i]),
            "coord_x": int(coords[i, 0]),
            "coord_y": int(coords[i, 1]),
        }
        for r, i in enumerate(topk_indices)
    ]

    result = {
        "slide_id": slide_id,
        "annotation": annotation,
        "top_k_patches": topk_info,
        "files": [],
    }

    # Try loading slide thumbnail
    slide_img, downsample = None, None
    if slide_path and Path(slide_path).is_file():
        slide_img, downsample, _ = load_slide_thumbnail(
            slide_path,
            vis_level=vis_level,
        )

    if slide_img is not None and downsample is not None:
        _render_with_tissue(
            slide_id,
            slide_img,
            downsample,
            scores,
            coords,
            patch_size_level0,
            topk_indices,
            annotation,
            output_dir,
            cmap_name,
            blur_sigma_mult,
            transparent_bins,
            alpha_cap,
            save_dpi,
            result,
            slide_path,
            vis_level,
            top_k,
        )
    else:
        _render_coord_scatter(
            slide_id,
            scores,
            coords,
            topk_indices,
            annotation,
            output_dir,
            cmap_name,
            save_dpi,
            result,
        )

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Rendering backends
# ═════════════════════════════════════════════════════════════════════════════


def _render_with_tissue(
    slide_id,
    slide_img,
    downsample,
    scores,
    coords,
    patch_size_level0,
    topk_indices,
    annotation,
    output_dir,
    cmap_name,
    blur_sigma_mult,
    transparent_bins,
    alpha_cap,
    save_dpi,
    result,
    slide_path,
    vis_level,
    top_k,
):
    """Render attention overlay on tissue thumbnail."""
    import matplotlib.pyplot as plt

    canvas_size = slide_img.size  # (W, H)

    # Build + smooth overlay
    overlay = build_attention_overlay(
        scores,
        coords,
        patch_size_level0,
        canvas_size,
        downsample,
    )
    patch_px = patch_size_level0 / downsample
    sigma = patch_px * blur_sigma_mult
    overlay = smooth_overlay(overlay, sigma)

    # Colormap → RGBA
    rgba, mappable = apply_colormap_rgba(
        overlay,
        cmap_name,
        transparent_bins,
        alpha_cap,
    )

    # Alpha composite onto tissue
    blended = alpha_blend(slide_img, rgba)

    # Draw top-k boxes on blended image
    blended_boxes = draw_topk_boxes(
        blended,
        topk_indices,
        coords,
        scores,
        patch_size_level0,
        downsample,
    )

    # ── Two-panel figure ──────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(20, 8),
        constrained_layout=True,
    )
    ax1.imshow(np.array(slide_img))
    ax1.set_title(f"{slide_id} — Raw WSI", fontsize=13)
    ax1.axis("off")
    ax2.imshow(np.array(blended_boxes))
    ax2.set_title(f"{slide_id} — Attention + Top-{top_k}", fontsize=13)
    ax2.axis("off")
    fig.colorbar(
        mappable,
        ax=[ax1, ax2],
        fraction=0.03,
        pad=0.02,
        label="Normalised Attention",
    )
    fig.suptitle(annotation, fontsize=14, y=0.02, va="bottom")

    overview_path = output_dir / f"{slide_id}_attention_overview.png"
    fig.savefig(str(overview_path), dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)
    result["files"].append(str(overview_path))

    # Standalone blended image
    blend_path = output_dir / f"{slide_id}_blended_attention.png"
    blended_boxes.save(str(blend_path))
    result["files"].append(str(blend_path))

    # Top-k patch crops (requires raw slide)
    try:
        _save_topk_patches(
            slide_id,
            topk_indices,
            scores,
            coords,
            patch_size_level0,
            slide_path,
            slide_img,
            downsample,
            output_dir,
            save_dpi,
        )
    except Exception as e:
        logger.warning(f"Could not save top-k patches for {slide_id}: {e}")


def _render_coord_scatter(
    slide_id,
    scores,
    coords,
    topk_indices,
    annotation,
    output_dir,
    cmap_name,
    save_dpi,
    result,
):
    """Fallback: scatter plot of attention at patch coordinates."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=scores,
        cmap=cmap_name,
        s=3,
        alpha=0.8,
    )
    ax.scatter(
        coords[topk_indices, 0],
        coords[topk_indices, 1],
        facecolors="none",
        edgecolors="white",
        s=80,
        linewidths=1.5,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"{slide_id} — Attention (coord-based)", fontsize=13)
    fig.colorbar(sc, ax=ax, label="Normalised Attention")
    fig.suptitle(annotation, fontsize=12, y=0.02, va="bottom")

    path = output_dir / f"{slide_id}_attention_coords.png"
    fig.savefig(str(path), dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)
    result["files"].append(str(path))


def _save_topk_patches(
    slide_id,
    topk_indices,
    scores,
    coords,
    patch_size_level0,
    slide_path,
    slide_thumb,
    downsample,
    output_dir,
    save_dpi,
):
    """Save individual top-k patch crops with location context."""
    import matplotlib.pyplot as plt
    from PIL import ImageDraw

    patch_dir = output_dir / "top_patches"
    patch_dir.mkdir(exist_ok=True)

    ext = Path(slide_path).suffix.lower()

    # Open slide for reading patches at level 0
    slide = None
    if ext == ".sdpc":
        try:
            from trident import SDPCWSI

            slide = SDPCWSI(slide_path=slide_path, lazy_init=False)
        except Exception:
            pass
    else:
        try:
            import openslide

            slide = openslide.OpenSlide(slide_path)
        except Exception:
            pass

    if slide is None:
        return

    scale = 1.0 / downsample

    for rank, i in enumerate(topk_indices):
        x, y = int(coords[i, 0]), int(coords[i, 1])

        try:
            if ext == ".sdpc":
                patch = slide.read_region(
                    (x, y),
                    0,
                    (patch_size_level0, patch_size_level0),
                )
            else:
                patch = slide.read_region(
                    (x, y),
                    0,
                    (patch_size_level0, patch_size_level0),
                ).convert("RGB")
        except Exception:
            continue

        # Side-by-side: patch crop | location on slide
        fig, (ax_p, ax_l) = plt.subplots(
            1,
            2,
            figsize=(12, 5),
            gridspec_kw={"width_ratios": [1, 1.5]},
            constrained_layout=True,
        )
        ax_p.imshow(patch)
        ax_p.set_title(f"Patch {rank + 1} (score: {scores[i]:.3f})")
        ax_p.axis("off")

        # Location highlight on thumbnail
        loc = slide_thumb.copy()
        draw = ImageDraw.Draw(loc)
        ps = int(patch_size_level0 * scale)
        cx, cy = int(x * scale), int(y * scale)
        draw.rectangle([cx, cy, cx + ps, cy + ps], outline="lime", width=4)

        ax_l.imshow(np.array(loc))
        ax_l.set_title("Location on slide")
        ax_l.axis("off")

        path = patch_dir / f"{slide_id}_patch_{rank + 1}.png"
        fig.savefig(str(path), dpi=save_dpi, bbox_inches="tight")
        plt.close(fig)
