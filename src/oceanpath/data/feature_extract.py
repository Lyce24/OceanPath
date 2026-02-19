"""
Wrapper around TRIDENT's Processor for config-driven feature extraction.

This module does NOT reimplement TRIDENT logic. It provides a clean,
config-driven interface that:
  1. Replaces argparse with structured configs
  2. Adds provenance tracking (manifest.json + encoder fingerprint)
  3. Makes extraction resumable at the slide level with stale-detection
  4. Separates "what to extract" from "how to run it"
  5. Supports Slurm array sharding for parallel extraction
  6. Validates all inputs before committing compute
"""

import json
import hashlib
import datetime
import logging
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ── Schema version ────────────────────────────────────────────────────────────
H5_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1


# ── Config dataclass ──────────────────────────────────────────────────────────


@dataclass
class TridentExtractionConfig:
    """All parameters needed to run TRIDENT feature extraction."""

    # Slide source
    wsi_dir: str
    job_dir: str
    wsi_ext: Optional[list[str]] = None
    custom_list_of_wsis: Optional[str] = None
    reader_type: Optional[str] = None
    search_nested: bool = False

    # Segmentation
    segmenter: str = "hest"
    seg_conf_thresh: float = 0.5
    remove_holes: bool = False
    remove_artifacts: bool = False
    remove_penmarks: bool = False

    # Patching
    mag: int = 20
    patch_size: int = 256
    overlap: int = 0
    min_tissue_proportion: float = 0.0
    coords_dir: Optional[str] = None

    # Feature extraction
    patch_encoder: str = "uni_v2"
    patch_encoder_ckpt_path: Optional[str] = None
    slide_encoder: Optional[str] = None

    # Runtime
    gpu: int = 0
    batch_size: int = 64
    seg_batch_size: Optional[int] = None
    feat_batch_size: Optional[int] = None
    max_workers: Optional[int] = None
    skip_errors: bool = True

    # Caching
    wsi_cache: Optional[str] = None
    cache_batch_size: int = 32

    @property
    def device(self) -> str:
        if torch.cuda.is_available():
            return f"cuda:{self.gpu}"
        return "cpu"

    @property
    def coords_subdir(self) -> str:
        return self.coords_dir or f"{self.mag}x_{self.patch_size}px_{self.overlap}px_overlap"

    @property
    def encoder_name(self) -> str:
        return self.slide_encoder or self.patch_encoder

    @property
    def encoder_type(self) -> str:
        return "slide" if self.slide_encoder else "patch"


# ── Encoder fingerprinting ────────────────────────────────────────────────────


def compute_encoder_fingerprint(cfg: TridentExtractionConfig) -> str:
    ckpt_path = cfg.patch_encoder_ckpt_path
    if ckpt_path and Path(ckpt_path).is_file():
        return _hash_file(Path(ckpt_path))
    id_string = f"{cfg.encoder_name}:{ckpt_path or 'default'}"
    return hashlib.sha256(id_string.encode()).hexdigest()[:16]


def _hash_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()[:16]


# ── Input validation ──────────────────────────────────────────────────────────


class ValidationError(Exception):
    pass


def validate_inputs(cfg: TridentExtractionConfig, tasks: list[str]) -> dict:
    errors = []

    wsi_dir = Path(cfg.wsi_dir)
    if not wsi_dir.is_dir():
        errors.append(f"Slide directory does not exist: {wsi_dir}")

    slide_count = 0
    if wsi_dir.is_dir():
        if cfg.custom_list_of_wsis:
            wsi_list_path = Path(cfg.custom_list_of_wsis)
            if not wsi_list_path.is_file():
                errors.append(f"Custom WSI list not found: {wsi_list_path}")
            else:
                slide_count = sum(
                    1 for line in wsi_list_path.read_text().splitlines() if line.strip()
                )
        else:
            extensions = set(cfg.wsi_ext or [".svs", ".tiff", ".ndpi", ".sdpc"])
            if cfg.search_nested:
                slide_count = sum(
                    1 for f in wsi_dir.rglob("*") if f.suffix.lower() in extensions
                )
            else:
                slide_count = sum(
                    1 for f in wsi_dir.iterdir() if f.suffix.lower() in extensions
                )

        if slide_count == 0:
            errors.append(f"No slides found in {wsi_dir} with extensions {cfg.wsi_ext}")

    if "cuda" in cfg.device:
        if not torch.cuda.is_available():
            errors.append("CUDA requested but torch.cuda.is_available() is False")
        elif cfg.gpu >= torch.cuda.device_count():
            errors.append(
                f"GPU {cfg.gpu} requested but only {torch.cuda.device_count()} available"
            )

    job_dir = Path(cfg.job_dir)
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        errors.append(f"Cannot create job directory {job_dir}: {e}")

    valid_tasks = {"seg", "coords", "feat"}
    for t in tasks:
        if t not in valid_tasks:
            errors.append(f"Unknown task '{t}'. Must be one of {valid_tasks}")

    if errors:
        raise ValidationError(
            "Input validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        )

    return {
        "slide_count": slide_count,
        "slide_dir": str(wsi_dir),
        "encoder": cfg.encoder_name,
        "encoder_type": cfg.encoder_type,
        "device": cfg.device,
        "tasks": tasks,
        "job_dir": str(cfg.job_dir),
        "mag": cfg.mag,
        "patch_size": cfg.patch_size,
    }


# ── Diff mode ─────────────────────────────────────────────────────────────────


def compute_extraction_diff(
    cfg: TridentExtractionConfig,
    encoder_fingerprint: str,
) -> Optional[Path]:
    wsi_dir = Path(cfg.wsi_dir)

    # Resolve H5 output dir based on encoder type
    if cfg.encoder_type == "slide":
        h5_dir = Path(cfg.job_dir) / cfg.coords_subdir / f"slide_features_{cfg.encoder_name}"
    else:
        h5_dir = Path(cfg.job_dir) / cfg.coords_subdir / f"features_{cfg.encoder_name}"

    manifest_path = Path(cfg.job_dir) / "manifest.json"

    extensions = set(cfg.wsi_ext or [".svs", ".tiff", ".ndpi", ".sdpc"])
    if cfg.custom_list_of_wsis:
        slide_names = [
            line.strip()
            for line in Path(cfg.custom_list_of_wsis).read_text().splitlines()
            if line.strip()
        ]
    elif cfg.search_nested:
        slide_names = [f.name for f in wsi_dir.rglob("*") if f.suffix.lower() in extensions]
    else:
        slide_names = [f.name for f in wsi_dir.iterdir() if f.suffix.lower() in extensions]

    if not slide_names:
        logger.warning("No slides found for diff computation")
        return None

    prev_fingerprint = None
    if manifest_path.is_file():
        try:
            prev_manifest = json.loads(manifest_path.read_text())
            prev_fingerprint = prev_manifest.get("encoder_fingerprint")
        except (json.JSONDecodeError, KeyError):
            pass

    needs_extraction = []
    for name in slide_names:
        slide_id = Path(name).stem
        h5_path = h5_dir / f"{slide_id}.h5"

        if not h5_path.is_file():
            needs_extraction.append(name)
        elif prev_fingerprint and prev_fingerprint != encoder_fingerprint:
            needs_extraction.append(name)

    if not needs_extraction:
        logger.info(
            f"All {len(slide_names)} slides already extracted "
            f"with encoder fingerprint {encoder_fingerprint[:8]}..."
        )
        return None

    logger.info(
        f"Diff mode: {len(needs_extraction)}/{len(slide_names)} slides need extraction"
    )

    diff_list_path = Path(cfg.job_dir) / ".diff_wsi_list.txt"
    diff_list_path.parent.mkdir(parents=True, exist_ok=True)
    diff_list_path.write_text("\n".join(needs_extraction) + "\n")
    return diff_list_path


# ── Slurm sharding ────────────────────────────────────────────────────────────


def shard_wsi_list(
    cfg: TridentExtractionConfig,
    shard_id: int,
    total_shards: int,
) -> Path:
    wsi_dir = Path(cfg.wsi_dir)
    extensions = set(cfg.wsi_ext or [".svs", ".tiff", ".ndpi", ".sdpc"])

    if cfg.custom_list_of_wsis:
        all_slides = sorted(
            line.strip()
            for line in Path(cfg.custom_list_of_wsis).read_text().splitlines()
            if line.strip()
        )
    elif cfg.search_nested:
        all_slides = sorted(f.name for f in wsi_dir.rglob("*") if f.suffix.lower() in extensions)
    else:
        all_slides = sorted(f.name for f in wsi_dir.iterdir() if f.suffix.lower() in extensions)

    shard_slides = all_slides[shard_id::total_shards]
    logger.info(f"Slurm shard {shard_id}/{total_shards}: {len(shard_slides)}/{len(all_slides)} slides")

    shard_dir = Path(cfg.job_dir) / ".shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"shard_{shard_id:04d}.txt"
    shard_path.write_text("\n".join(shard_slides) + "\n")
    return shard_path


# ── Core functions (lazy TRIDENT import) ──────────────────────────────────────

_trident_imported = False


def _lazy_import_trident():
    global _trident_imported, Processor
    if _trident_imported:
        return
    from trident import Processor as _Proc
    Processor = _Proc
    _trident_imported = True


def create_processor(cfg: TridentExtractionConfig):
    _lazy_import_trident()
    return Processor(
        job_dir=cfg.job_dir,
        wsi_source=cfg.wsi_dir,
        wsi_ext=cfg.wsi_ext,
        wsi_cache=cfg.wsi_cache,
        skip_errors=cfg.skip_errors,
        custom_mpp_keys=None,
        custom_list_of_wsis=cfg.custom_list_of_wsis,
        max_workers=cfg.max_workers,
        reader_type=cfg.reader_type,
        search_nested=cfg.search_nested,
    )


def run_segmentation(processor, cfg: TridentExtractionConfig) -> None:
    from trident.segmentation_models.load import segmentation_model_factory

    seg_model = segmentation_model_factory(
        cfg.segmenter, confidence_thresh=cfg.seg_conf_thresh,
    )
    artifact_model = None
    if cfg.remove_artifacts or cfg.remove_penmarks:
        artifact_model = segmentation_model_factory(
            "grandqc_artifact",
            remove_penmarks_only=cfg.remove_penmarks and not cfg.remove_artifacts,
        )
    processor.run_segmentation_job(
        seg_model,
        seg_mag=seg_model.target_mag,
        holes_are_tissue=not cfg.remove_holes,
        artifact_remover_model=artifact_model,
        batch_size=cfg.seg_batch_size or cfg.batch_size,
        device=cfg.device,
    )


def run_patching(processor, cfg: TridentExtractionConfig) -> None:
    processor.run_patching_job(
        target_magnification=cfg.mag,
        patch_size=cfg.patch_size,
        overlap=cfg.overlap,
        saveto=cfg.coords_dir,
        min_tissue_proportion=cfg.min_tissue_proportion,
    )


def run_feature_extraction(processor, cfg: TridentExtractionConfig) -> None:
    if cfg.slide_encoder is None:
        from trident.patch_encoder_models.load import encoder_factory
        encoder = encoder_factory(cfg.patch_encoder, weights_path=cfg.patch_encoder_ckpt_path)
        processor.run_patch_feature_extraction_job(
            coords_dir=cfg.coords_subdir,
            patch_encoder=encoder,
            device=cfg.device,
            saveas="h5",
            batch_limit=cfg.feat_batch_size or cfg.batch_size,
        )
    else:
        from trident.slide_encoder_models.load import encoder_factory
        encoder = encoder_factory(cfg.slide_encoder)
        processor.run_slide_feature_extraction_job(
            slide_encoder=encoder,
            coords_dir=cfg.coords_subdir,
            device=cfg.device,
            saveas="h5",
            batch_limit=cfg.feat_batch_size or cfg.batch_size,
        )


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run_pipeline(
    cfg: TridentExtractionConfig,
    tasks: Optional[list[str]] = None,
    diff_mode: bool = False,
    shard_id: Optional[int] = None,
    total_shards: Optional[int] = None,
    dry_run: bool = False,
    force: bool = False,
) -> Optional[Path]:
    if tasks is None:
        tasks = ["seg", "coords", "feat"]

    start_time = time.monotonic()

    # Step 1: Validate
    logger.info("Validating inputs...")
    summary = validate_inputs(cfg, tasks)
    encoder_fp = compute_encoder_fingerprint(cfg)
    summary["encoder_fingerprint"] = encoder_fp

    logger.info(
        f"Validation passed: {summary['slide_count']} slides, "
        f"encoder={summary['encoder']} (fp={encoder_fp[:8]}...), "
        f"device={summary['device']}, tasks={summary['tasks']}"
    )

    if dry_run:
        logger.info("DRY RUN — validation passed, exiting before compute")
        _print_dry_run_summary(summary)
        return None

    # Step 2: Slurm sharding
    if shard_id is not None and total_shards is not None:
        shard_list = shard_wsi_list(cfg, shard_id, total_shards)
        cfg = TridentExtractionConfig(**{**asdict(cfg), "custom_list_of_wsis": str(shard_list)})

    # Step 3: Diff mode
    if diff_mode and "feat" in tasks and not force:
        diff_list = compute_extraction_diff(cfg, encoder_fp)
        if diff_list is None:
            return Path(cfg.job_dir)
        cfg = TridentExtractionConfig(**{**asdict(cfg), "custom_list_of_wsis": str(diff_list)})

    # Step 4: Run tasks
    task_fn = {
        "seg": run_segmentation,
        "coords": run_patching,
        "feat": run_feature_extraction,
    }

    processor = create_processor(cfg)

    for task_name in tasks:
        if task_name not in task_fn:
            raise ValueError(f"Unknown task: {task_name}. Must be one of {list(task_fn)}")
        task_start = time.monotonic()
        logger.info(f"{'=' * 60}")
        logger.info(f"  Starting task: {task_name}")
        logger.info(f"{'=' * 60}")
        task_fn[task_name](processor, cfg)
        logger.info(f"  Task '{task_name}' completed in {time.monotonic() - task_start:.1f}s")

    # Step 5: Provenance
    total_elapsed = time.monotonic() - start_time
    _write_manifest(cfg, encoder_fp, tasks, total_elapsed, shard_id, total_shards)
    logger.info(f"Pipeline completed in {total_elapsed:.1f}s → {cfg.job_dir}")
    return Path(cfg.job_dir)


# ── Provenance ────────────────────────────────────────────────────────────────


def _write_manifest(
    cfg: TridentExtractionConfig,
    encoder_fingerprint: str,
    tasks: list[str],
    elapsed_seconds: float,
    shard_id: Optional[int] = None,
    total_shards: Optional[int] = None,
) -> None:
    git_sha = _get_git_sha()

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "encoder": cfg.encoder_name,
        "encoder_type": cfg.encoder_type,
        "encoder_fingerprint": encoder_fingerprint,
        "magnification": cfg.mag,
        "patch_size": cfg.patch_size,
        "overlap": cfg.overlap,
        "segmenter": cfg.segmenter,
        "seg_conf_thresh": cfg.seg_conf_thresh,
        "wsi_dir": cfg.wsi_dir,
        "tasks_run": tasks,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "extracted_at": datetime.datetime.now().isoformat(),
        "device": cfg.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "git_sha": git_sha,
        "h5_schema_version": H5_SCHEMA_VERSION,
        "full_config": asdict(cfg),
    }

    if shard_id is not None:
        manifest["shard"] = {"id": shard_id, "total": total_shards}
        manifest_name = f"manifest_shard_{shard_id:04d}.json"
    else:
        manifest_name = "manifest.json"

    manifest_path = Path(cfg.job_dir) / manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2, default=str))
    tmp_path.rename(manifest_path)
    logger.info(f"Manifest saved: {manifest_path}")


# ── Output validation ─────────────────────────────────────────────────────────


def validate_outputs(cfg: TridentExtractionConfig, tasks: list[str]) -> dict:
    if "feat" not in tasks:
        return {"total": 0, "found": 0, "missing": []}

    wsi_dir = Path(cfg.wsi_dir)

    # Resolve H5 dir based on encoder type (matches TRIDENT output structure)
    if cfg.encoder_type == "slide":
        h5_dir = Path(cfg.job_dir) / cfg.coords_subdir / f"slide_features_{cfg.encoder_name}"
    else:
        h5_dir = Path(cfg.job_dir) / cfg.coords_subdir / f"features_{cfg.encoder_name}"

    extensions = set(cfg.wsi_ext or [".svs", ".tiff", ".ndpi", ".sdpc"])

    if cfg.custom_list_of_wsis:
        expected = {
            Path(line.strip()).stem
            for line in Path(cfg.custom_list_of_wsis).read_text().splitlines()
            if line.strip()
        }
    elif cfg.search_nested:
        expected = {f.stem for f in wsi_dir.rglob("*") if f.suffix.lower() in extensions}
    else:
        expected = {f.stem for f in wsi_dir.iterdir() if f.suffix.lower() in extensions}

    found = {f.stem for f in h5_dir.glob("*.h5")} if h5_dir.is_dir() else set()
    missing = sorted(expected - found)

    result = {"total": len(expected), "found": len(found & expected), "missing": missing}

    if missing:
        logger.warning(f"Output validation: {len(missing)}/{len(expected)} slides missing")
    else:
        logger.info(f"Output validation: all {len(expected)} slides have H5 files")

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_git_sha() -> Optional[str]:
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _print_dry_run_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("  DRY RUN SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key:>25s}: {value}")
    print("=" * 60 + "\n")