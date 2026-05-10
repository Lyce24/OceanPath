"""
End-to-end linear probing pipeline: extract slide embeddings + sklearn probe.

Pipeline stages:
  1) Extract slide-level embeddings from the patch-level mmap.
     - mean_pool: mask-aware mean-pooling (baseline, no model)
     - aggregator: forward through a pretrained MIL aggregator
     Embeddings are written to an OceanPath mmap directory and cached.
  2) Run a StandardScaler + LogisticRegression linear probe on those
     embeddings, respecting patient-level grouping to prevent leakage.

Caching:
  - If the embedding mmap already exists, extraction is skipped.
  - Set force_extract=true to rebuild, or set embedding.mmap_dir to
    point at a pre-extracted mmap and bypass extraction entirely.

Usage:
    # Mean-pool + 5-fold CV
    python scripts/linear_probing.py platform=local data=gej encoder=univ2

    # Pretrained aggregator + 5-fold CV
    python scripts/linear_probing.py platform=local data=gej encoder=univ2 \
        model=bimamba embedding.mode=aggregator \
        embedding.ckpt=outputs/pretrain/vicreg/model.ckpt

    # Skip extraction — use pre-extracted embeddings
    python scripts/linear_probing.py platform=local data=gej encoder=univ2 \
        embedding.mmap_dir=mmap/gej_custom_embeddings

    # External transfer: train TCGA -> test CPTAC
    python scripts/linear_probing.py platform=local data=tcga_nsclc encoder=univ2 \
        protocol=external_test \
        test.mmap_dir=mmap/cptac_nsclc_patch_features \
        test.manifest_csv=manifests/cptac_nsclc.csv

    # Predefined split
    python scripts/linear_probing.py platform=local data=gej encoder=univ2 \
        protocol=predefined_split split_column=split

    # Custom C grid + metric
    python scripts/linear_probing.py platform=local data=gej encoder=univ2 \
        'c_grid=[0.01,0.1,1.0,10.0]' primary_metric=auprc
"""

import argparse
import logging
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

POOLING_MODES = {"mean_pool", "max_pool", "mean_max_pool"}

logger = logging.getLogger(__name__)


def _setup_logging(cfg: DictConfig) -> None:
    level = logging.DEBUG if cfg.get("verbose", False) else logging.INFO
    data_name = OmegaConf.select(cfg, "data.name", default="unknown")
    encoder_name = OmegaConf.select(cfg, "encoder.name", default="unknown")
    fmt = (
        "%(asctime)s | %(levelname)-7s | "
        f"stage=linear_probing | data={data_name} | encoder={encoder_name} | "
        "%(message)s"
    )
    logging.basicConfig(level=level, format=fmt, force=True)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1: Embedding extraction
# ═════════════════════════════════════════════════════════════════════════════


def _file_content_hash(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of file contents, read in 1 MB chunks. Returns 12-char hex prefix."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:12]


def _model_config_hash(cfg: DictConfig) -> str:
    """Hash the full model config (arch + all hyperparameters) into a short tag.

    This ensures that two runs with the same arch but different hyperparameters
    (e.g. embed_dim=512 vs 384, num_layers=2 vs 4) get separate cache dirs.
    """
    import hashlib

    model_cfg = OmegaConf.select(cfg, "model", default={})
    # OmegaConf.to_yaml gives a stable, sorted representation
    model_str = OmegaConf.to_yaml(model_cfg, resolve=True)
    return hashlib.sha256(model_str.encode()).hexdigest()[:8]


def _embedding_cache_tag(cfg: DictConfig) -> str:
    """Build a deterministic cache tag from extraction-relevant config fields.

    Encodes mode, arch, model config hash, checkpoint content hash, and
    max_instances so that different extraction configs never silently share
    the same cached mmap.
    """
    mode = cfg.embedding.mode
    parts = [mode]

    if mode == "aggregator":
        arch = OmegaConf.select(cfg, "model.arch", default="unknown")
        mcfg_hash = _model_config_hash(cfg)
        parts.append(f"{arch}_{mcfg_hash}")

        ckpt = OmegaConf.select(cfg, "embedding.ckpt", default=None)
        if ckpt:
            ckpt_path = Path(str(ckpt))
            ckpt_stem = ckpt_path.stem
            if ckpt_path.is_file():
                ckpt_hash = _file_content_hash(ckpt_path)
            else:
                logger.warning("Checkpoint file not found for hashing: %s — using path hash.", ckpt)
                import hashlib

                ckpt_hash = hashlib.sha256(str(ckpt).encode()).hexdigest()[:12]
            parts.append(f"{ckpt_stem}_{ckpt_hash}")
        else:
            parts.append("random_init")

    max_inst = OmegaConf.select(cfg, "embedding.max_instances", default=None)
    if max_inst is not None:
        parts.append(f"cap{max_inst}")

    # Sampling mode only affects the cached embeddings when a cap is active.
    # Including it in the tag prevents silent reuse across different sampling
    # regimes (e.g. switching from contiguous → spatial_stratified).
    sampling_mode = OmegaConf.select(cfg, "embedding.sampling_mode", default="contiguous")
    if max_inst is not None and sampling_mode != "contiguous":
        sampling_seed = OmegaConf.select(cfg, "embedding.sampling_seed", default=42)
        parts.append(f"{sampling_mode}-s{sampling_seed}")

    return "_".join(parts)


def _resolve_embedding_mmap_dir(cfg: DictConfig) -> str:
    """Resolve the embedding mmap directory from config.

    If embedding.mmap_dir is set explicitly, use it (user-managed cache).
    Otherwise auto-derive: {embedding.base_dir}/{data.name}_{cache_tag}_embeddings
    """
    explicit = OmegaConf.select(cfg, "embedding.mmap_dir", default=None)
    if explicit is not None:
        return str(explicit)
    base_dir = cfg.embedding.base_dir
    tag = _embedding_cache_tag(cfg)
    return str(Path(base_dir) / f"{cfg.data.name}_{tag}_embeddings")


def _format_label_fraction_tag(label_fraction: float) -> str:
    pct = float(label_fraction) * 100.0
    text = f"{pct:g}".replace(".", "p")
    return f"labels{text}pct"


def _probe_output_tag(cfg: DictConfig, cache_tag: str) -> str:
    label_fraction = float(OmegaConf.select(cfg, "label_fraction", default=1.0))
    if label_fraction < 1.0:
        return f"{cache_tag}_{_format_label_fraction_tag(label_fraction)}"
    return cache_tag


def _build_extract_args(cfg: DictConfig, output_dir: str) -> argparse.Namespace:
    """Build the argparse.Namespace expected by the extraction module."""
    return argparse.Namespace(
        # Core paths
        mmap_dir=cfg.data.mmap_dir,
        manifest_csv=cfg.data.csv_path,
        output_path=output_dir,
        output_format="mmap",
        # Manifest columns
        slide_id_column=OmegaConf.select(cfg, "slide_id_column", default=None),
        filename_column=cfg.filename_column,
        # Mode + aggregator
        mode=cfg.embedding.mode,
        arch=OmegaConf.select(cfg, "model.arch", default="mamba2mil"),
        ckpt=OmegaConf.select(cfg, "embedding.ckpt", default=None),
        device=OmegaConf.select(cfg, "embedding.device", default=None),
        batch_size=cfg.embedding.batch_size,
        num_workers=cfg.embedding.num_workers,
        max_instances=OmegaConf.select(cfg, "embedding.max_instances", default=None),
        sampling_mode=OmegaConf.select(cfg, "embedding.sampling_mode", default="contiguous"),
        sampling_seed=int(OmegaConf.select(cfg, "embedding.sampling_seed", default=42)),
        # Aggregator hyperparameters from model config
        in_dim=OmegaConf.select(cfg, "encoder.feature_dim", default=1536),
        embed_dim=OmegaConf.select(cfg, "model.embed_dim", default=384),
        num_layers=OmegaConf.select(cfg, "model.num_layers", default=2),
        d_state=OmegaConf.select(cfg, "model.d_state", default=64),
        d_conv=OmegaConf.select(cfg, "model.d_conv", default=4),
        expand=OmegaConf.select(cfg, "model.expand", default=2),
        headdim=OmegaConf.select(cfg, "model.headdim", default=64),
        attn_dim=OmegaConf.select(cfg, "model.attn_dim", default=256),
        dropout=OmegaConf.select(cfg, "model.dropout", default=0.1),
        use_hilbert=OmegaConf.select(cfg, "model.use_hilbert", default=True),
        hilbert_order=OmegaConf.select(cfg, "model.hilbert_order", default=16),
        use_cobra_inference=OmegaConf.select(cfg, "model.use_cobra_inference", default=False),
        gradient_checkpointing=OmegaConf.select(cfg, "model.gradient_checkpointing", default=False),
        # Output mmap precision
        feat_precision=cfg.embedding.feat_precision,
        coord_dtype=cfg.embedding.coord_dtype,
        max_chunk_gb=cfg.embedding.max_chunk_gb,
        # Flags
        verbose=cfg.get("verbose", False),
    )


def _embedding_mmap_is_ready(mmap_dir: str) -> bool:
    """Check if an embedding mmap directory exists and contains a valid index."""
    p = Path(mmap_dir)
    return (p / "index_arrays.npz").is_file() and (p / ".schema_version").is_file()


def _validate_slide_overlap(mmap_dir: str, slide_ids: list[str]) -> None:
    """Pre-flight check: verify manifest slide_ids overlap with mmap index."""
    import numpy as np

    idx_path = Path(mmap_dir) / "index_arrays.npz"
    if not idx_path.is_file():
        raise FileNotFoundError(f"Mmap index not found: {idx_path}")

    idx = np.load(str(idx_path), allow_pickle=True)
    mmap_ids = set(idx["slide_ids"].tolist())
    manifest_ids = set(slide_ids)
    overlap = mmap_ids & manifest_ids
    n_missing = len(manifest_ids) - len(overlap)

    if not overlap:
        raise RuntimeError(
            f"Zero slide_id overlap between manifest ({len(manifest_ids)} IDs) "
            f"and mmap ({len(mmap_ids)} IDs).\n"
            f"  Manifest examples: {sorted(manifest_ids)[:3]}\n"
            f"  Mmap examples:     {sorted(mmap_ids)[:3]}\n"
            f"The slide_id convention must match: mmap uses the relative path from "
            f"h5_dir without .h5 extension (e.g. 'subdir/slide_001').\n"
            f"Check data.csv_path and filename_column / slide_id_column."
        )

    if n_missing > 0:
        logger.warning(
            "%d / %d manifest slide_ids not found in mmap (proceeding with %d matched).",
            n_missing,
            len(manifest_ids),
            len(overlap),
        )


def _run_extraction(cfg: DictConfig, embedding_mmap_dir: str) -> dict:
    """Run slide-level embedding extraction. Returns summary dict."""
    import torch

    from oceanpath.modules.extract_slide_embeddings import (
        build_aggregator_from_args,
        build_loader,
        extract_aggregator,
        extract_pooling,
        load_slide_ids,
        write_embedding_mmap,
    )

    args = _build_extract_args(cfg, embedding_mmap_dir)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Using device: %s", device)

    slide_ids = load_slide_ids(args.manifest_csv, args.slide_id_column, args.filename_column)
    logger.info("Resolved %d slide IDs from %s", len(slide_ids), args.manifest_csv)

    # Pre-flight: verify slide_ids match the mmap index before building loader
    _validate_slide_overlap(args.mmap_dir, slide_ids)

    loader = build_loader(
        mmap_dir=args.mmap_dir,
        slide_ids=slide_ids,
        mode=args.mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_instances=args.max_instances,
        sampling_mode=getattr(args, "sampling_mode", "contiguous"),
        sampling_seed=int(getattr(args, "sampling_seed", 42)),
    )

    logger.info(
        "Extraction config | mode=%s | batch_size=%d | max_instances=%s | sampling_mode=%s",
        args.mode,
        args.batch_size,
        args.max_instances,
        getattr(args, "sampling_mode", "contiguous"),
    )

    t0 = time.monotonic()
    if args.mode in POOLING_MODES:
        embeddings, kept_slide_ids = extract_pooling(loader, device, args.mode)
    else:
        torch.manual_seed(42)
        model = build_aggregator_from_args(args)
        embeddings, kept_slide_ids = extract_aggregator(loader, model, device)
    extract_elapsed = time.monotonic() - t0

    logger.info(
        "Extracted %d slide embeddings with dim=%d in %.1fs",
        embeddings.shape[0],
        embeddings.shape[1],
        extract_elapsed,
    )

    write_embedding_mmap(embedding_mmap_dir, kept_slide_ids, embeddings, args)
    logger.info("Saved embedding mmap -> %s", embedding_mmap_dir)

    return {
        "mode": args.mode,
        "n_slides": len(kept_slide_ids),
        "embed_dim": int(embeddings.shape[1]),
        "extract_seconds": round(extract_elapsed, 1),
        "arch": args.arch if args.mode == "aggregator" else None,
        "ckpt": args.ckpt,
    }


def _run_extraction_for_test(cfg: DictConfig, test_mmap_dir: str, test_manifest_csv: str) -> str:
    """Extract embeddings for an external test cohort. Returns embedding mmap path."""
    import torch

    from oceanpath.modules.extract_slide_embeddings import (
        build_aggregator_from_args,
        build_loader,
        extract_aggregator,
        extract_pooling,
        load_slide_ids,
        write_embedding_mmap,
    )

    # Derive output path under same base_dir as primary embeddings
    base_dir = cfg.embedding.base_dir
    tag = _embedding_cache_tag(cfg)
    test_name = Path(test_mmap_dir).name  # e.g. "cptac_nsclc"
    embedding_dir = str(Path(base_dir) / f"{test_name}_{tag}_embeddings")

    if _embedding_mmap_is_ready(embedding_dir) and not cfg.get("force_extract", False):
        logger.info("Test embedding mmap exists, reusing: %s", embedding_dir)
        return embedding_dir

    logger.info("Extracting test cohort embeddings -> %s", embedding_dir)
    args = _build_extract_args(cfg, embedding_dir)
    # Override paths for test cohort
    args.mmap_dir = test_mmap_dir
    args.manifest_csv = test_manifest_csv
    # Override column names for test cohort if specified
    args.slide_id_column = (
        OmegaConf.select(cfg, "test.slide_id_column", default=None) or args.slide_id_column
    )
    args.filename_column = (
        OmegaConf.select(cfg, "test.filename_column", default=None) or args.filename_column
    )

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    slide_ids = load_slide_ids(args.manifest_csv, args.slide_id_column, args.filename_column)
    logger.info("Test cohort: %d slide IDs from %s", len(slide_ids), args.manifest_csv)

    _validate_slide_overlap(args.mmap_dir, slide_ids)

    loader = build_loader(
        mmap_dir=args.mmap_dir,
        slide_ids=slide_ids,
        mode=args.mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_instances=args.max_instances,
        sampling_mode=getattr(args, "sampling_mode", "contiguous"),
        sampling_seed=int(getattr(args, "sampling_seed", 42)),
    )

    if args.mode in POOLING_MODES:
        embeddings, kept_slide_ids = extract_pooling(loader, device, args.mode)
    else:
        torch.manual_seed(42)
        model = build_aggregator_from_args(args)
        embeddings, kept_slide_ids = extract_aggregator(loader, model, device)

    write_embedding_mmap(embedding_dir, kept_slide_ids, embeddings, args)
    logger.info("Saved test embedding mmap -> %s", embedding_dir)

    return embedding_dir


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2: Linear probing
# ═════════════════════════════════════════════════════════════════════════════


def _build_probe_args(cfg: DictConfig, embedding_mmap_dir: str) -> argparse.Namespace:
    """Build the args Namespace from a Hydra config — thin wrapper around
    `oceanpath.modules.linear_probing_sklearn.make_probe_args` so this script,
    the LP callback, and the CLI all share one shape.
    """
    from oceanpath.modules.linear_probing_sklearn import make_probe_args

    return make_probe_args(
        # Core paths
        mmap_dir=embedding_mmap_dir,
        manifest_csv=cfg.data.csv_path,
        output_dir=cfg.output_dir,
        # External test cohort (set later if needed)
        test_mmap_dir=None,
        test_manifest_csv=None,
        # Manifest columns
        slide_id_column=OmegaConf.select(cfg, "slide_id_column", default=None),
        filename_column=cfg.filename_column,
        label_column=cfg.label_column,
        patient_column=cfg.patient_column,
        split_column=OmegaConf.select(cfg, "split_column", default=None),
        # Protocol
        protocol=cfg.protocol,
        n_splits=cfg.n_splits,
        inner_splits=cfg.inner_splits,
        primary_metric=OmegaConf.select(cfg, "primary_metric", default=None),
        # Probe hyperparameters
        c_grid=list(cfg.c_grid) if cfg.c_grid else None,
        class_weight=cfg.class_weight,
        max_iter=cfg.max_iter,
        multi_class_mode=cfg.multi_class_mode,
        # Robustness
        non_singleton_policy=cfg.non_singleton_policy,
        allow_missing_mmap=cfg.allow_missing_mmap,
        allow_missing_manifest=cfg.allow_missing_manifest,
        aggregate_to_patient=OmegaConf.select(cfg, "aggregate_to_patient", default=True),
        # Predefined-split partition names
        train_split_names=list(cfg.train_split_names) if cfg.get("train_split_names") else None,
        eval_split_names=list(cfg.eval_split_names) if cfg.get("eval_split_names") else None,
        # Bootstrap / LDE / output
        n_boot=OmegaConf.select(cfg, "n_boot", default=0),
        label_fraction=OmegaConf.select(cfg, "label_fraction", default=1.0),
        label_fraction_seed=OmegaConf.select(cfg, "label_fraction_seed", default=None),
        seed=cfg.seed,
        save_model=cfg.save_model,
        verbose=cfg.get("verbose", False),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


@hydra.main(config_path="../configs", config_name="linear_probing", version_base=None)
def main(cfg: DictConfig) -> None:
    _setup_logging(cfg)
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    from oceanpath.modules.linear_probing_sklearn import (
        load_embedding_mmap,
        load_manifest,
        merge_manifest_and_embeddings,
        run_external_test,
        run_grouped_cv,
        run_predefined_split,
        save_summary,
    )
    from oceanpath.modules.linear_probing_sklearn import (
        setup_logging as setup_probe_logging,
    )

    # ── Pre-flight: fail fast with a readable message before extraction ──────
    primary_manifest = Path(str(cfg.data.csv_path))
    if not primary_manifest.is_file():
        raise FileNotFoundError(
            f"data.csv_path does not exist: {primary_manifest} "
            f"(data={cfg.data.name}). Check platform.project_root and the manifest path."
        )

    primary_mmap = Path(str(cfg.data.mmap_dir))
    if not (primary_mmap / "index_arrays.npz").is_file():
        raise FileNotFoundError(
            f"data.mmap_dir lacks index_arrays.npz: {primary_mmap} "
            f"(data={cfg.data.name}). Build the patch mmap with scripts/build_mmap.py first."
        )

    if str(cfg.protocol) == "external_test":
        test_mmap = OmegaConf.select(cfg, "test.mmap_dir", default=None)
        test_manifest = OmegaConf.select(cfg, "test.manifest_csv", default=None)
        if not test_mmap or not test_manifest:
            raise ValueError(
                "protocol=external_test requires both test.mmap_dir and test.manifest_csv."
            )
        if not Path(str(test_manifest)).is_file():
            raise FileNotFoundError(f"test.manifest_csv does not exist: {test_manifest}")
        if not (Path(str(test_mmap)) / "index_arrays.npz").is_file():
            raise FileNotFoundError(f"test.mmap_dir lacks index_arrays.npz: {test_mmap}")

    # ── Compute cache tag once — used for both embedding dir and output dir ──
    cache_tag = _embedding_cache_tag(cfg)

    # Append probe tag to output_dir so different modes / ckpts / caps / label
    # fractions don't collide. The embedding cache itself is still keyed only by
    # extraction-relevant fields.
    output_dir = Path(cfg.output_dir) / _probe_output_tag(cfg, cache_tag)

    # ═════════════════════════════════════════════════════════════════════════
    # Stage 1: Embedding extraction (or reuse)
    # ═════════════════════════════════════════════════════════════════════════
    embedding_mmap_dir = _resolve_embedding_mmap_dir(cfg)
    force_extract = cfg.get("force_extract", False)
    extract_summary = None

    if _embedding_mmap_is_ready(embedding_mmap_dir) and not force_extract:
        logger.info("Embedding mmap exists, reusing: %s", embedding_mmap_dir)
    else:
        logger.info(
            "Extracting slide embeddings | mode=%s | -> %s",
            cfg.embedding.mode,
            embedding_mmap_dir,
        )
        extract_summary = _run_extraction(cfg, embedding_mmap_dir)

    # ═════════════════════════════════════════════════════════════════════════
    # Stage 2: Linear probing
    # ═════════════════════════════════════════════════════════════════════════
    args = _build_probe_args(cfg, embedding_mmap_dir)
    args.output_dir = str(output_dir)  # override with tag-aware path
    setup_probe_logging(output_dir, args.verbose)

    # ── Load primary cohort ──────────────────────────────────────────────────
    logger.info("Loading primary manifest: %s", args.manifest_csv)
    manifest_df = load_manifest(
        csv_path=args.manifest_csv,
        slide_id_column=args.slide_id_column,
        filename_column=args.filename_column,
        label_column=args.label_column,
        patient_column=args.patient_column,
        split_column=args.split_column if args.protocol == "predefined_split" else None,
    )

    logger.info("Loading primary embedding mmap: %s", embedding_mmap_dir)
    store = load_embedding_mmap(embedding_mmap_dir, non_singleton_policy=args.non_singleton_policy)

    df = merge_manifest_and_embeddings(
        manifest_df=manifest_df,
        store=store,
        allow_missing_mmap=args.allow_missing_mmap,
        allow_missing_manifest=args.allow_missing_manifest,
    )
    logger.info(
        "Primary cohort: %d slides | %d patients | %d classes",
        len(df),
        df["patient_id"].nunique(),
        df["label"].nunique(),
    )

    # ── Run protocol ─────────────────────────────────────────────────────────
    if args.protocol == "grouped_cv":
        summary = run_grouped_cv(args, df)

    elif args.protocol == "external_test":
        test_patch_mmap = OmegaConf.select(cfg, "test.mmap_dir", default=None)
        test_manifest = OmegaConf.select(cfg, "test.manifest_csv", default=None)
        if not test_patch_mmap or not test_manifest:
            raise ValueError("protocol=external_test requires test.mmap_dir and test.manifest_csv.")

        # Extract test cohort embeddings (same mode/model as primary)
        test_embedding_dir = _run_extraction_for_test(cfg, test_patch_mmap, test_manifest)

        # Test manifest may use different column names; fall back to primary
        test_label_col = (
            OmegaConf.select(cfg, "test.label_column", default=None) or args.label_column
        )
        test_patient_col = (
            OmegaConf.select(cfg, "test.patient_column", default=None) or args.patient_column
        )
        test_filename_col = (
            OmegaConf.select(cfg, "test.filename_column", default=None) or args.filename_column
        )
        test_slide_id_col = (
            OmegaConf.select(cfg, "test.slide_id_column", default=None) or args.slide_id_column
        )

        logger.info("Loading external test manifest: %s", test_manifest)
        test_manifest_df = load_manifest(
            csv_path=test_manifest,
            slide_id_column=test_slide_id_col,
            filename_column=test_filename_col,
            label_column=test_label_col,
            patient_column=test_patient_col,
            split_column=None,
        )
        logger.info("Loading external test embedding mmap: %s", test_embedding_dir)
        test_store = load_embedding_mmap(
            test_embedding_dir,
            non_singleton_policy=args.non_singleton_policy,
        )
        test_df = merge_manifest_and_embeddings(
            manifest_df=test_manifest_df,
            store=test_store,
            allow_missing_mmap=args.allow_missing_mmap,
            allow_missing_manifest=args.allow_missing_manifest,
        )
        logger.info(
            "External cohort: %d slides | %d patients | %d classes",
            len(test_df),
            test_df["patient_id"].nunique(),
            test_df["label"].nunique(),
        )
        # Set test paths on args so run_external_test can save them
        args.test_mmap_dir = test_embedding_dir
        args.test_manifest_csv = test_manifest
        summary = run_external_test(args, df, test_df)

    elif args.protocol == "predefined_split":
        summary = run_predefined_split(args, df)

    else:
        raise ValueError(f"Unknown protocol: {args.protocol}")

    # ── Attach extraction metadata to summary ────────────────────────────────
    if extract_summary is not None:
        summary["extraction"] = extract_summary
    summary["embedding_mmap_dir"] = embedding_mmap_dir

    # ── Save & print ─────────────────────────────────────────────────────────
    save_summary(output_dir, summary)
    logger.info("Summary saved -> %s", output_dir / "summary.json")

    _print_summary(cfg, args, summary, embedding_mmap_dir, extract_summary)
    logger.info("Done.")


def _print_summary(
    cfg: DictConfig,
    args: argparse.Namespace,
    summary: dict,
    embedding_mmap_dir: str,
    extract_summary: dict | None,
) -> None:
    print(f"\n{'=' * 64}")
    print("  linear_probing — complete")
    print(f"{'=' * 64}")

    # Extraction info
    mode = cfg.embedding.mode
    if extract_summary:
        print(
            f"  Embedding mode:   {mode} (extracted {extract_summary['n_slides']} slides, "
            f"dim={extract_summary['embed_dim']}, {extract_summary['extract_seconds']}s)"
        )
        if mode == "aggregator":
            print(f"  Architecture:     {extract_summary.get('arch', '?')}")
            print(f"  Checkpoint:       {extract_summary.get('ckpt') or '(random init)'}")
    else:
        print(f"  Embedding mode:   {mode} (cached)")
    print(f"  Embedding mmap:   {embedding_mmap_dir}")

    # Probing info
    print(f"  Manifest:         {args.manifest_csv}")
    print(f"  Output:           {args.output_dir}")

    task_type = summary.get("task_type", "unknown")
    print(f"  Task type:        {task_type}")
    print(f"  Protocol:         {args.protocol}")
    label_fraction = summary.get("label_fraction", getattr(args, "label_fraction", 1.0))
    if label_fraction is not None and float(label_fraction) < 1.0:
        actual = summary.get("mean_train_labeled_patient_fraction")
        if actual is None:
            actual = summary.get("train_labeled_patient_fraction")
        suffix = f" (actual patients {actual:.3f})" if actual is not None else ""
        print(f"  Label fraction:   {float(label_fraction):.3f}{suffix}")

    if args.protocol == "grouped_cv":
        pm = summary.get("primary_metric", "?")
        mean_key = f"mean_{pm}"
        std_key = f"std_{pm}"
        if mean_key in summary:
            print(f"  {pm}:    {summary[mean_key]:.4f} +/- {summary.get(std_key, 0):.4f}")
        print(f"  Folds:            {summary.get('n_folds', '?')}")
    else:
        pm = summary.get("primary_metric", "?")
        if pm in summary:
            print(f"  {pm}:    {summary[pm]:.4f}")
        print(f"  Best C:           {summary.get('best_C', '?')}")

    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
