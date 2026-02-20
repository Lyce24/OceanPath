"""
Stage 7: Interpretability & post-training analysis.

Makes model behavior inspectable:
  - Attention heatmaps on individual slides (WSI overlays + top-k patches)
  - UMAP of slide-level embeddings (class-colored, decision boundaries)

Reads:
  - outputs/train/{exp_name}/ → checkpoints, embeddings, predictions
  - {feature_h5_dir}          → H5 files for coords + coords_attrs
  - {slide_dir}               → WSI files (scanned to discover slides)
  - {mmap_dir}                → features (+ fallback coords)
  - {csv_path}                → cohort metadata + labels

Writes:
  - outputs/analyze/{exp_name}/
    ├── analyze_config.yaml
    ├── analyze_summary.json
    ├── attention/           → per-slide heatmaps + top-k patches
    └── umap/                → scatter plots + umap.parquet

Model handling:
  - Reads eval/recommended_model.json to find the winning strategy
  - If ensemble: loads ALL K fold checkpoints, averages attention + embeddings
  - If best_fold/refit: loads single checkpoint
  - Override with analysis.model_strategy=ensemble|best_fold|refit

Coordinate + patch_size source (for attention overlays):
  Primary: H5 files at {feature_h5_dir}/{slide_id}.h5
    → coords dataset + coords.attrs["patch_size_level0"]
    This matches the original notebook workflow exactly.
  Fallback: memmap coords (same data, but no attrs metadata).

Slide selection for attention (analysis.attention.slide_ids):
  - Explicit: set slide_ids=[GEJ_001,GEJ_042] in config or CLI
  - Auto (default): picks top errors (most confident wrong predictions)
    + random correct predictions for contrast (from OOF)
  - slide_dir scan: when OOF unavailable, discovers slides from slide_dir

WSI path discovery:
  Scans cfg.data.slide_dir for files matching cfg.data.wsi_extensions,
  just like the original notebook scanned a flat directory.

Usage:
    python scripts/analyze.py platform=local data=gej encoder=univ1 \\
           splits=kfold5 model=abmil training=gej

    # Specific slides
    python scripts/analyze.py ... \\
           analysis.attention.slide_ids=[GEJ_001,GEJ_042,GEJ_103]

    # UMAP only
    python scripts/analyze.py ... analysis.run_attention=false

    # Force ensemble model
    python scripts/analyze.py ... analysis.model_strategy=ensemble
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _setup_logging(cfg: DictConfig) -> None:
    """Configure logging to match train.py / evaluate.py style."""
    level = logging.DEBUG if cfg.get("verbose", False) else logging.INFO
    exp_name = OmegaConf.select(cfg, "exp_name", default="analyze")
    fmt = (
        "%(asctime)s | %(levelname)-7s | "
        f"exp={exp_name} | "
        "%(message)s"
    )
    logging.basicConfig(level=level, format=fmt, force=True)


# ═════════════════════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════════════════════


def resolve_model_strategy(
    train_dir: Path,
    eval_dir: Path,
    override: Optional[str] = None,
) -> tuple[str, Path]:
    """
    Determine which final model to use.

    Priority:
      1. Explicit override from config (analysis.model_strategy)
      2. eval/recommended_model.json from Stage 6
      3. Fallback: best_fold

    Returns (strategy_name, model_directory).
    """
    if override and override != "null":
        strategy = override
    else:
        rec_path = eval_dir / "recommended_model.json"
        if rec_path.is_file():
            rec = json.loads(rec_path.read_text())
            strategy = rec.get("recommended_model", "best_fold")
            logger.info(f"Stage 6 recommended model: {strategy}")
        else:
            strategy = "best_fold"
            logger.warning(
                f"No recommended_model.json at {rec_path}. "
                f"Falling back to '{strategy}'."
            )

    model_dir = train_dir / "final" / strategy
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. "
            f"Did Stage 5 finalization complete?"
        )

    logger.info(f"Using model strategy: {strategy} → {model_dir}")
    return strategy, model_dir


def load_model_wrapper(model_dir: Path, device: str):
    """Load the EnsembleWrapper (works for single models too)."""
    from oceanpath.eval.ensemble import EnsembleWrapper

    wrapper = EnsembleWrapper(model_dir=model_dir, device=device)
    logger.info(
        f"Loaded {wrapper.n_models} model(s) "
        f"(strategy: {wrapper.strategy}) on {device}"
    )
    return wrapper


# ═════════════════════════════════════════════════════════════════════════════
# Slide discovery & path resolution
# ═════════════════════════════════════════════════════════════════════════════


def build_slide_lookup(cfg: DictConfig) -> dict[str, str]:
    """
    Build {slide_id: wsi_path} by scanning cfg.data.slide_dir.

    Mirrors the original notebook which loaded slides from a flat directory:
        slide = SDPCWSI(slide_path=f"{data_path}/{de_id}.sdpc", ...)

    Scans slide_dir for files matching cfg.data.wsi_extensions, then
    supplements with CSV entries if any slides have non-standard locations.

    Returns
    -------
    dict mapping slide_id (stem) → absolute file path.
    """
    from oceanpath.eval.attention import discover_slides_in_dir

    wsi_extensions = list(
        cfg.data.get("wsi_extensions", [".svs", ".tiff", ".sdpc"])
    )

    # Primary: scan slide_dir directly
    slide_dir = str(cfg.data.slide_dir)
    slide_to_file = discover_slides_in_dir(slide_dir, wsi_extensions)

    # Supplement: check CSV for any additional slides
    try:
        csv_df = pd.read_csv(cfg.data.csv_path)
        filename_col = cfg.data.filename_column
        for _, row in csv_df.iterrows():
            sid = Path(str(row[filename_col])).stem
            if sid in slide_to_file:
                continue
            # Check if this slide exists with any extension
            for ext in wsi_extensions:
                candidate = Path(slide_dir) / f"{sid}{ext}"
                if candidate.is_file():
                    slide_to_file[sid] = str(candidate)
                    break
    except Exception as e:
        logger.debug(f"CSV supplement scan skipped: {e}")

    return slide_to_file


def build_label_map(cfg: DictConfig) -> dict[str, int]:
    """
    Build {slide_id: int_label} from the labels CSV.

    Mirrors the original notebook:
        labels[i] = full_df.loc[full_df['filename'] == i, 'Diagnosis'].values[0]
    """
    csv_df = pd.read_csv(cfg.data.csv_path)
    filename_col = cfg.data.filename_column
    label_col = (
        cfg.data.label_columns[0] if cfg.data.label_columns else "label"
    )

    label_map = {}
    for _, row in csv_df.iterrows():
        sid = Path(str(row[filename_col])).stem
        label_map[sid] = int(row[label_col])

    return label_map


# ═════════════════════════════════════════════════════════════════════════════
# Slide selection for attention
# ═════════════════════════════════════════════════════════════════════════════


def select_slides_for_attention(
    cfg_attn: DictConfig,
    train_dir: Path,
    slide_to_file: dict[str, str],
) -> list[str]:
    """
    Select which slides to visualize.

    Modes (in priority order):
      1. Explicit: cfg_attn.slide_ids is a list → use directly
      2. Auto from OOF: pick top errors + random correct from OOF predictions
      3. All from OOF: max_slides == -1 → every OOF slide
      4. Fallback: scan slide_dir → pick up to max_slides available WSIs

    Parameters
    ----------
    cfg_attn : attention config block.
    train_dir : training output directory (contains oof_predictions.parquet).
    slide_to_file : {slide_id: wsi_path} from build_slide_lookup.
    """
    # Mode 1: explicit slide IDs from config
    explicit = cfg_attn.get("slide_ids")
    if explicit is not None and len(explicit) > 0:
        slide_ids = list(explicit)
        logger.info(
            f"Attention: using {len(slide_ids)} explicit slide IDs from config"
        )
        return slide_ids

    max_slides = cfg_attn.get("max_slides", 20)

    # Mode 2/3: auto-select from OOF predictions
    oof_path = train_dir / "oof_predictions.parquet"
    if oof_path.is_file():
        slide_ids = _select_from_oof(oof_path, cfg_attn, max_slides)
        if slide_ids:
            return slide_ids

    # Mode 4: fallback — discover from slide_dir
    if slide_to_file:
        all_ids = sorted(slide_to_file.keys())
        if max_slides == -1:
            logger.info(
                f"Attention: using ALL {len(all_ids)} slides from slide_dir"
            )
            return all_ids
        selected = all_ids[:max_slides]
        logger.info(
            f"Attention: selected {len(selected)} slides from slide_dir "
            f"(OOF unavailable, using directory scan)"
        )
        return selected

    logger.warning(
        "No slides found: no OOF predictions and no WSIs in slide_dir. "
        "Set analysis.attention.slide_ids explicitly."
    )
    return []


def _select_from_oof(
    oof_path: Path,
    cfg_attn: DictConfig,
    max_slides: int,
) -> list[str]:
    """Select slides from OOF predictions (top errors + random correct)."""
    oof_df = pd.read_parquet(str(oof_path))
    prob_cols = sorted([c for c in oof_df.columns if c.startswith("prob_")])
    if not prob_cols:
        logger.warning("No probability columns in OOF predictions")
        return []

    # Compute predicted class + confidence
    if len(prob_cols) == 1:
        oof_df["pred_class"] = (oof_df[prob_cols[0]] > 0.5).astype(int)
        oof_df["pred_confidence"] = (oof_df[prob_cols[0]] - 0.5).abs() * 2
    else:
        probs = oof_df[prob_cols].values
        oof_df["pred_class"] = probs.argmax(axis=1)
        oof_df["pred_confidence"] = probs.max(axis=1)

    oof_df["is_error"] = oof_df["label"] != oof_df["pred_class"]

    n_errors = cfg_attn.get("auto_n_errors", 10)
    n_correct = cfg_attn.get("auto_n_correct", 10)

    # All slides
    if max_slides == -1:
        slide_ids = oof_df["slide_id"].tolist()
        logger.info(f"Attention: using ALL {len(slide_ids)} OOF slides")
        return slide_ids

    # Auto: top errors (most confident wrong) + random correct
    errors = oof_df[oof_df["is_error"]].sort_values(
        "pred_confidence", ascending=False,
    )
    error_ids = errors["slide_id"].head(min(n_errors, len(errors))).tolist()

    correct = oof_df[~oof_df["is_error"]]
    if len(correct) > n_correct:
        correct_ids = (
            correct.sample(n=n_correct, random_state=42)["slide_id"].tolist()
        )
    else:
        correct_ids = correct["slide_id"].tolist()

    # Combine, deduplicate, cap
    slide_ids = list(dict.fromkeys(error_ids + correct_ids))[:max_slides]

    logger.info(
        f"Attention: auto-selected {len(slide_ids)} slides "
        f"({len(error_ids)} errors + {len(correct_ids)} correct)"
    )
    return slide_ids


# ═════════════════════════════════════════════════════════════════════════════
# Part 1: Attention heatmaps
# ═════════════════════════════════════════════════════════════════════════════


def run_attention_analysis(
    cfg: DictConfig,
    wrapper,
    slide_ids: list[str],
    slide_to_file: dict[str, str],
    label_map: dict[str, int],
    output_dir: Path,
) -> dict:
    """
    Produce attention heatmaps for selected slides.

    Coordinate loading strategy (matching original notebook):
      1. H5 files at {feature_h5_dir}/{slide_id}.h5  (primary)
         → coords dataset  +  coords.attrs["patch_size_level0"]
         Original notebook:
           coords = f['coords'][:]
           coords_attrs = dict(f['coords'].attrs)
           patch_size = coords_attrs["patch_size_level0"]
      2. Memmap dataset with return_coords=True      (fallback)

    For each slide:
      1. Load features from memmap
      2. Load coords from H5 (with patch_size_level0 from attrs)
      3. Forward through model(s) with return_attention=True
      4. Render overlay on slide thumbnail + top-k patches
    """
    from oceanpath.eval.attention import (
        visualize_slide_attention,
        load_h5_coords,
        get_patch_size_from_h5,
    )
    from oceanpath.data.dataset import MmapDataset

    a = cfg.analysis.attention
    attn_dir = output_dir / "attention"
    attn_dir.mkdir(parents=True, exist_ok=True)

    class_names = _build_class_names(cfg)

    # ── Resolve H5 directory for coords ───────────────────────────────────
    feature_h5_dir = Path(cfg.data.feature_h5_dir)
    h5_available = feature_h5_dir.is_dir()
    if h5_available:
        logger.info(f"H5 coord source: {feature_h5_dir}")
    else:
        logger.warning(
            f"H5 directory not found: {feature_h5_dir}. "
            f"Falling back to memmap coords."
        )

    # ── Build memmap dataset for features (+ fallback coords) ─────────────
    dataset = MmapDataset(
        mmap_dir=str(cfg.data.mmap_dir),
        slide_ids=slide_ids,
        labels=label_map,
        max_instances=None,  # no subsampling for analysis
        is_train=False,
        return_coords=True,  # enable fallback coords from memmap
    )

    # ── Resolve default patch_size_level0 ─────────────────────────────────
    #
    # This is the global default. Per-slide values from H5 attrs take
    # priority inside the loop (the original notebook used per-slide
    # coords_attrs["patch_size_level0"] for each slide).
    #
    default_patch_size = _resolve_patch_size(
        cfg, a, feature_h5_dir, slide_ids,
    )

    # ── Run per-slide attention ───────────────────────────────────────────
    results = []
    skipped = []

    logger.info(f"Running attention analysis on {len(dataset)} slides...")

    for idx in range(len(dataset)):
        item = dataset[idx]
        sid = item["slide_id"]
        features = item["features"]  # [N, D]

        # ── Load coords: H5 (primary) → memmap (fallback) ────────────────
        #
        # Mirrors original notebook:
        #   coords = preloaded_coords[de_id]["coords"]
        #   coords_attrs = preloaded_coords[de_id]["coords_attrs"]
        #   patch_size_level0 = coords_attrs["patch_size_level0"]
        #
        coords = None
        slide_patch_size = default_patch_size

        h5_path = feature_h5_dir / f"{sid}.h5"
        if h5_available and h5_path.is_file():
            try:
                h5_coords, coords_attrs = load_h5_coords(str(h5_path))
                coords = h5_coords

                # Per-slide patch_size_level0 from H5 attrs (most precise)
                ps_from_attrs = coords_attrs.get("patch_size_level0")
                if ps_from_attrs is not None:
                    slide_patch_size = int(ps_from_attrs)
            except Exception as e:
                logger.warning(f"Failed to load H5 coords for {sid}: {e}")

        # Fallback to memmap coords
        if coords is None:
            if "coords" in item:
                coords = item["coords"].numpy()
            else:
                logger.warning(f"No coords for {sid} — skipping attention")
                skipped.append(sid)
                continue

        # Run model(s) with attention
        output = wrapper.predict_slide(features)

        if output.attention is None:
            logger.warning(
                f"No attention returned for {sid} — "
                f"architecture may not support attention extraction"
            )
            skipped.append(sid)
            continue

        # Render
        try:
            result = visualize_slide_attention(
                slide_id=sid,
                attention=output.attention,
                coords=coords,
                pred_class=output.pred_class,
                pred_prob=output.pred_prob,
                true_label=label_map.get(sid, -1),
                patch_size_level0=slide_patch_size,
                output_dir=str(attn_dir / sid),
                slide_path=slide_to_file.get(sid),
                class_names=class_names,
                vis_level=a.vis_level,
                top_k=a.top_k,
                cmap_name=a.cmap,
                blur_sigma_mult=a.blur_sigma_mult,
                transparent_bins=a.transparent_bins,
                alpha_cap=a.alpha_cap,
                save_dpi=a.save_dpi,
                n_models=wrapper.n_models,
            )
            results.append(result)
            logger.info(
                f"  [{idx + 1}/{len(dataset)}] {sid}: "
                f"pred={output.pred_class} ({output.pred_prob:.3f}), "
                f"true={label_map.get(sid, '?')}, "
                f"patch_size={slide_patch_size}"
            )
        except Exception as e:
            logger.error(f"Failed to visualize {sid}: {e}", exc_info=True)
            skipped.append(sid)

    # ── Save summary ──────────────────────────────────────────────────────
    summary = {
        "n_visualized": len(results),
        "n_skipped": len(skipped),
        "skipped_ids": skipped,
        "model_strategy": wrapper.strategy,
        "n_models": wrapper.n_models,
        "default_patch_size_level0": default_patch_size,
        "coord_source": "h5" if h5_available else "memmap",
        "feature_h5_dir": str(feature_h5_dir),
        "slides": [
            {
                "slide_id": r["slide_id"],
                "annotation": r["annotation"],
                "n_files": len(r.get("files", [])),
            }
            for r in results
        ],
    }
    summary_path = attn_dir / "attention_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        f"Attention analysis complete: {len(results)} slides, "
        f"{len(skipped)} skipped → {attn_dir}"
    )
    return summary


def _resolve_patch_size(
    cfg: DictConfig,
    attn_cfg: DictConfig,
    feature_h5_dir: Path,
    slide_ids: list[str],
) -> int:
    """
    Resolve the default patch_size_level0.

    This is the global fallback — individual slides may override it from
    their H5 coords attrs inside the per-slide loop.

    Priority:
      1. Explicit config: analysis.attention.patch_size_level0
      2. H5 coords attrs: coords.attrs["patch_size_level0"]
         (matching original notebook behaviour)
      3. TRIDENT manifest.json in the feature_job_dir
      4. Fallback: extraction.patch_size
    """
    from oceanpath.eval.attention import get_patch_size_from_h5

    # Priority 1: explicit config
    ps = attn_cfg.get("patch_size_level0")
    if ps is not None and ps != "null":
        logger.info(f"patch_size_level0 from config: {ps}")
        return int(ps)

    # Priority 2: H5 coords attrs (original notebook method)
    ps = get_patch_size_from_h5(str(feature_h5_dir), slide_ids)
    if ps is not None:
        return ps

    # Priority 3: TRIDENT manifest
    manifest_path = Path(cfg.data.feature_job_dir) / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
            val = manifest.get("patch_size_level0")
            if val:
                logger.info(f"patch_size_level0 from manifest: {val}")
                return int(val)
        except Exception:
            pass

    # Priority 4: fallback
    fallback = int(cfg.extraction.patch_size)
    logger.warning(
        f"patch_size_level0 not set — using extraction.patch_size={fallback}. "
        f"If your slides are 40x native and extraction is at 20x, set "
        f"analysis.attention.patch_size_level0={fallback * 2}"
    )
    return fallback


# ═════════════════════════════════════════════════════════════════════════════
# Part 2: UMAP
# ═════════════════════════════════════════════════════════════════════════════


def run_umap_analysis(
    cfg: DictConfig,
    wrapper,
    train_dir: Path,
    label_map: dict[str, int],
    output_dir: Path,
) -> dict:
    """
    Compute and plot UMAP of slide embeddings.

    Two embedding sources:
      "oof" → load pre-computed OOF embeddings from training (default)
      "live" → recompute embeddings through the current model wrapper
    """
    from oceanpath.eval.umap_viz import (
        load_oof_embeddings,
        load_test_embeddings,
        compute_umap,
        plot_umap,
        plot_train_test_umap,
        save_umap_data,
    )

    u = cfg.analysis.umap
    umap_dir = output_dir / "umap"
    umap_dir.mkdir(parents=True, exist_ok=True)

    class_names = _build_class_names(cfg)
    n_folds = cfg.splits.get("n_folds", cfg.splits.get("k", 5))

    # ── Load or compute embeddings ────────────────────────────────────────
    if u.source == "oof":
        emb_df = load_oof_embeddings(str(train_dir), n_folds)
        if emb_df is None or emb_df.empty:
            logger.warning(
                "No OOF embeddings found. Falling back to live computation."
            )
            emb_df = _compute_live_embeddings(cfg, wrapper, label_map)
    elif u.source == "live":
        emb_df = _compute_live_embeddings(cfg, wrapper, label_map)
    else:
        raise ValueError(f"Unknown UMAP source: {u.source}")

    if emb_df is None or emb_df.empty:
        logger.error("Cannot compute embeddings — skipping UMAP")
        return {"error": "no embeddings"}

    # ── Extract arrays ────────────────────────────────────────────────────
    embeddings = np.stack(emb_df["embedding"].values)
    slide_ids = emb_df["slide_id"].tolist()
    labels = np.array([label_map.get(sid, -1) for sid in slide_ids])

    logger.info(
        f"UMAP: {len(slide_ids)} slides, "
        f"embedding dim={embeddings.shape[1]}"
    )

    # ── Compute UMAP ─────────────────────────────────────────────────────
    coords_2d = compute_umap(
        embeddings,
        n_pca=u.n_pca,
        n_neighbors=u.n_neighbors,
        min_dist=u.min_dist,
        metric=u.metric,
        seed=u.seed,
    )

    # Highlight slides from config
    highlight_ids = list(u.highlight_slide_ids or [])

    # ── Plot: basic UMAP ─────────────────────────────────────────────────
    plot_umap(
        coords_2d, labels,
        output_path=str(umap_dir / "umap.png"),
        class_names=class_names,
        title=f"UMAP — {cfg.exp_name} ({u.source} embeddings)",
        highlight_slide_ids=highlight_ids,
        all_slide_ids=slide_ids,
        save_dpi=u.save_dpi,
        show_decision_boundary=False,
    )

    # ── Plot: with SVM decision boundary ─────────────────────────────────
    if u.show_decision_boundary and len(np.unique(labels)) >= 2:
        plot_umap(
            coords_2d, labels,
            output_path=str(umap_dir / "umap_decision_boundary.png"),
            class_names=class_names,
            title=f"UMAP + SVM Boundaries — {cfg.exp_name}",
            highlight_slide_ids=highlight_ids,
            all_slide_ids=slide_ids,
            save_dpi=u.save_dpi,
            show_decision_boundary=True,
        )

    # ── Save UMAP coordinates as parquet ─────────────────────────────────
    csv_df = pd.read_csv(cfg.data.csv_path)
    filename_col = cfg.data.filename_column
    metadata_df = csv_df.rename(columns={filename_col: "filename"})
    metadata_df["slide_id"] = metadata_df["filename"].apply(
        lambda x: Path(str(x)).stem
    )
    save_umap_data(
        coords_2d, slide_ids, labels,
        output_path=str(umap_dir / "umap.parquet"),
        metadata_df=metadata_df,
    )

    # ── Combined train+test UMAP (if test embeddings exist) ──────────────
    test_emb_df = load_test_embeddings(str(train_dir), n_folds)
    if test_emb_df is not None and not test_emb_df.empty:
        test_embeddings = np.stack(test_emb_df["embedding"].values)
        test_ids = test_emb_df["slide_id"].tolist()
        test_labels = np.array([label_map.get(sid, -1) for sid in test_ids])

        try:
            plot_train_test_umap(
                train_embeddings=embeddings,
                train_labels=labels,
                train_ids=slide_ids,
                test_embeddings=test_embeddings,
                test_labels=test_labels,
                test_ids=test_ids,
                output_path=str(umap_dir / "umap_train_test.png"),
                class_names=class_names,
                n_pca=u.n_pca,
                n_neighbors=u.n_neighbors,
                min_dist=u.min_dist,
                metric=u.metric,
                seed=u.seed,
                save_dpi=u.save_dpi,
                title=f"UMAP Train + Test — {cfg.exp_name}",
            )
        except Exception as e:
            logger.warning(f"Could not produce train+test UMAP: {e}")

    summary = {
        "n_slides": len(slide_ids),
        "embedding_dim": int(embeddings.shape[1]),
        "source": u.source,
        "n_folds": n_folds,
    }
    logger.info(f"UMAP analysis complete → {umap_dir}")
    return summary


def _compute_live_embeddings(cfg, wrapper, label_map):
    """Compute embeddings through the model wrapper on all available slides."""
    from oceanpath.data.dataset import MmapDataset

    dataset = MmapDataset(
        mmap_dir=str(cfg.data.mmap_dir),
        slide_ids=list(label_map.keys()),
        labels=label_map,
        max_instances=cfg.training.get("max_instances", 8000),
        is_train=False,
        return_coords=False,
    )

    embeddings, slide_ids, labels = wrapper.get_all_embeddings(dataset)
    records = [
        {"slide_id": sid, "fold": -1, "embedding": emb}
        for sid, emb in zip(slide_ids, embeddings)
    ]
    return pd.DataFrame(records)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _build_class_names(cfg: DictConfig) -> Optional[dict]:
    """
    Build {int: str} class name mapping from config.

    Returns None when no class names are configured, so downstream
    functions (plot_umap, etc.) auto-generate names from labels.
    """
    names = cfg.analysis.get("class_names")
    if names is None or (hasattr(names, "__len__") and len(names) == 0):
        return None
    return {i: str(n) for i, n in enumerate(names)}


def _json_default(o):
    """JSON serialiser fallback for numpy types."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _print_attention_report(summary: dict) -> None:
    """Print a structured attention analysis report."""
    n_vis = summary.get("n_visualized", 0)
    n_skip = summary.get("n_skipped", 0)

    print(f"\n{'─' * 55}")
    print(f"  Attention Heatmaps")
    print(f"{'─' * 55}")
    print(f"  Visualized:       {n_vis}")
    print(f"  Skipped:          {n_skip}")
    print(f"  Coord source:     {summary.get('coord_source', '?')}")
    print(f"  patch_size_lvl0:  {summary.get('default_patch_size_level0', '?')}")

    slides = summary.get("slides", [])
    if slides:
        for s in slides[:10]:
            sid = s.get("slide_id", "?")
            ann = s.get("annotation", "")
            print(f"    {sid}: {ann}")
        if len(slides) > 10:
            print(f"    ... and {len(slides) - 10} more")

    print(f"{'─' * 55}")


def _print_umap_report(summary: dict) -> None:
    """Print a structured UMAP analysis report."""
    print(f"\n{'─' * 55}")
    print(f"  UMAP Embeddings")
    print(f"{'─' * 55}")
    print(f"  Slides:        {summary.get('n_slides', '?')}")
    print(f"  Embedding dim: {summary.get('embedding_dim', '?')}")
    print(f"  Source:        {summary.get('source', '?')}")
    print(f"  N folds:       {summary.get('n_folds', '?')}")
    print(f"{'─' * 55}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


@hydra.main(config_path="../configs", config_name="analyze", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Stage 7: Interpretability & post-training analysis."""

    _setup_logging(cfg)

    start_time = time.monotonic()
    a = cfg.analysis

    # Log banner + resolved config (matches evaluate.py style)
    logger.info("=" * 60)
    logger.info("  Stage 7: Interpretability & Post-Training Analysis")
    logger.info("=" * 60)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # Resolve paths
    train_dir = Path(cfg.train_dir)
    eval_dir = Path(cfg.eval_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    # ── Build shared lookups ──────────────────────────────────────────────
    #
    # Scan slide_dir once (like the original notebook scanning data_path)
    # and load labels from CSV (like the original loading from full_df).
    #
    slide_to_file = build_slide_lookup(cfg)
    label_map = build_label_map(cfg)

    logger.info(
        f"Discovered {len(slide_to_file)} WSIs, "
        f"{len(label_map)} labelled slides"
    )

    # ── Dry run ───────────────────────────────────────────────────────────
    if cfg.dry_run:
        enabled = []
        if a.run_attention:
            enabled.append("attention")
        if a.run_umap:
            enabled.append("UMAP")

        print(f"\n{'=' * 60}")
        print(f"  DRY RUN — analyze.py")
        print(f"{'=' * 60}")
        print(f"  Experiment:  {cfg.exp_name}")
        print(f"  Train dir:   {train_dir}")
        print(f"  Output dir:  {output_dir}")
        print(f"  Slide dir:   {cfg.data.slide_dir}")
        print(f"  H5 dir:      {cfg.data.feature_h5_dir}")
        print(f"  WSIs found:  {len(slide_to_file)}")
        print(f"  Labelled:    {len(label_map)}")
        print(f"  Analyses:    {', '.join(enabled) if enabled else '(none)'}")
        print(f"  Model:       {a.get('model_strategy', 'auto')}")
        print(f"{'=' * 60}\n")
        return

    # Save resolved config
    config_path = output_dir / "analyze_config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))

    # ── Resolve model strategy ────────────────────────────────────────────
    strategy_override = a.get("model_strategy")
    strategy, model_dir = resolve_model_strategy(
        train_dir, eval_dir, strategy_override,
    )

    # ── Load model wrapper ────────────────────────────────────────────────
    wrapper = None
    need_model = a.run_attention or (a.run_umap and a.umap.source == "live")

    if need_model:
        device = a.get("device", "cuda:0")
        if not torch.cuda.is_available() and "cuda" in device:
            device = "cpu"
            logger.warning("CUDA not available — falling back to CPU")
        wrapper = load_model_wrapper(model_dir, device)

    results = {}

    # ── Part 1: Attention ─────────────────────────────────────────────────
    if a.run_attention:
        logger.info("=" * 50)
        logger.info("  Part 1: Attention Heatmaps")
        logger.info("=" * 50)

        slide_ids = select_slides_for_attention(
            a.attention, train_dir, slide_to_file,
        )

        if slide_ids:
            results["attention"] = run_attention_analysis(
                cfg, wrapper, slide_ids, slide_to_file, label_map, output_dir,
            )
            _print_attention_report(results["attention"])
        else:
            logger.warning("No slides selected for attention analysis")
            results["attention"] = {"skipped": True}

    # ── Part 2: UMAP ─────────────────────────────────────────────────────
    if a.run_umap:
        logger.info("=" * 50)
        logger.info("  Part 2: UMAP Embeddings")
        logger.info("=" * 50)

        results["umap"] = run_umap_analysis(
            cfg, wrapper, train_dir, label_map, output_dir,
        )
        if "error" not in results["umap"]:
            _print_umap_report(results["umap"])

    # ── Cleanup ───────────────────────────────────────────────────────────
    n_models = wrapper.n_models if wrapper else 0
    if wrapper is not None:
        wrapper.cleanup()

    # ── Save summary ──────────────────────────────────────────────────────
    results_path = output_dir / "analyze_summary.json"
    results_path.write_text(
        json.dumps(results, indent=2, default=_json_default)
    )

    elapsed = time.monotonic() - start_time

    # ── Final summary (print block, matches train.py / evaluate.py) ──────
    print(f"\n{'=' * 60}")
    print(f"  Stage 7 Complete")
    print(f"{'=' * 60}")
    print(f"  Output:       {output_dir}")
    print(f"  Time:         {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"  Model:        {strategy} ({n_models} model(s))")
    print(f"  WSIs:         {len(slide_to_file)} discovered")

    for name, r in results.items():
        if isinstance(r, dict) and "error" not in r and not r.get("skipped"):
            detail = ""
            if name == "attention":
                detail = f" ({r.get('n_visualized', '?')} slides)"
            elif name == "umap":
                detail = f" ({r.get('n_slides', '?')} slides)"
            print(f"  {name:15s}: ✓{detail}")
        else:
            print(f"  {name:15s}: skipped")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()