"""
MIL training orchestrator with fold-level resume.

Runs all folds of a CV scheme, with:
  - Fold-level resume backed by config, prediction, metric, and checkpoint evidence
  - Explicit memory cleanup between folds
  - W&B logging with fold grouping
  - OOF prediction + embedding collection
  - Split integrity verification
  - Deterministic replay (batch index logging)
  - Phase 2 finalization: best_fold, ensemble, refit

Usage:
    # Standard 5-fold ABMIL
    python scripts/train.py platform=local data=blca encoder=conch_v15 splits=kfold5 model=abmil

    # TransMIL with holdout split
    python scripts/train.py platform=local data=blca encoder=virchow2 splits=kfold5 model=transmil

    # Override training hyperparams
    python scripts/train.py ... training.lr=1e-4 training.max_epochs=50

    # Force rerun of completed folds
    python scripts/train.py ... training.force_rerun=true

    # Dry run: validate configs and data, don't train
    python scripts/train.py ... dry_run=true
"""

import gc
import hashlib
import json
import logging
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from omegaconf import DictConfig, OmegaConf

from oceanpath.contracts import PipelineStage, StageResult

logger = logging.getLogger(__name__)

_TRAINING_IDENTITY_SCHEMA_VERSION = 1
_FOLD_COMPLETION_SCHEMA_VERSION = 1
_RUN_COMPLETION_SCHEMA_VERSION = 1
_TRAINING_IDENTITY_FILE = "training_identity.json"
_FOLD_COMPLETION_FILE = "completion.json"
_RUN_COMPLETION_FILE = "training_completion.json"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _file_sha256(path: Path) -> str:
    """Hash a material run artifact without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_file_sha256(path: Path) -> str:
    return _file_sha256(path) if path.is_file() else "missing"


def _plain_config_section(cfg: DictConfig, key: str) -> Any:
    value = OmegaConf.select(cfg, key, default={})
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def training_identity_payload(cfg: DictConfig) -> dict[str, Any]:
    """Return values that can materially change a trained model.

    Operational switches such as ``dry_run``, verbosity, W&B mode, and
    ``force_rerun`` are deliberately excluded. Input evidence is included so a
    manifest, split assignment, mmap source, or preload checkpoint cannot be
    replaced in place and then silently reuse completed folds.
    """
    from oceanpath.config import FoundationPaths

    paths = FoundationPaths.from_config(cfg)
    training = _plain_config_section(cfg, "training")
    if isinstance(training, dict):
        training = dict(training)
        training.pop("force_rerun", None)

    platform_keys = ("accelerator", "devices", "strategy", "precision", "num_nodes")
    platform = {key: _plain_config_section(cfg, f"platform.{key}") for key in platform_keys}

    encoder_checkpoint = OmegaConf.select(cfg, "encoder.checkpoint_path", default=None)
    aggregator_checkpoint = OmegaConf.select(cfg, "training.aggregator_weights_path", default=None)

    return {
        "schema_version": _TRAINING_IDENTITY_SCHEMA_VERSION,
        "material_config": {
            "data": _plain_config_section(cfg, "data"),
            "encoder": _plain_config_section(cfg, "encoder"),
            "splits": _plain_config_section(cfg, "splits"),
            "model": _plain_config_section(cfg, "model"),
            "training": training,
            "platform": platform,
        },
        "input_evidence": {
            "manifest_sha256": _optional_file_sha256(paths.manifest_path),
            "split_integrity_sha256": _optional_file_sha256(paths.splits_dir / ".integrity_hash"),
            "mmap_source_sha256": _optional_file_sha256(paths.mmap_dir / ".source_hash"),
            "encoder_checkpoint_sha256": (
                _optional_file_sha256(Path(str(encoder_checkpoint)))
                if encoder_checkpoint not in (None, "null")
                else "default"
            ),
            "aggregator_checkpoint_sha256": (
                _optional_file_sha256(Path(str(aggregator_checkpoint)))
                if aggregator_checkpoint not in (None, "null")
                else "none"
            ),
        },
    }


def training_run_fingerprint(cfg: DictConfig) -> str:
    """Fingerprint the semantic training config and its material inputs."""
    canonical = json.dumps(
        training_identity_payload(cfg),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    temporary.replace(path)


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_training_identity(cfg: DictConfig, output_dir: Path) -> str:
    payload = training_identity_payload(cfg)
    fingerprint = training_run_fingerprint(cfg)
    _write_json_atomic(
        output_dir / _TRAINING_IDENTITY_FILE,
        {
            "schema_version": _TRAINING_IDENTITY_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "payload": payload,
        },
    )
    return fingerprint


def _assert_training_output_identity(cfg: DictConfig, output_dir: Path) -> str:
    """Refuse to mix a new material config into an existing train directory."""
    expected = training_run_fingerprint(cfg)
    identity_path = output_dir / _TRAINING_IDENTITY_FILE
    if identity_path.is_file():
        stored = _load_json_object(identity_path)
        actual = stored.get("fingerprint") if stored else None
        if actual != expected:
            raise RuntimeError(
                "Training output identity collision. The existing directory was produced "
                "by a different material training configuration or input snapshot.\n"
                f"  output_dir: {output_dir}\n"
                f"  existing:   {actual or 'invalid identity record'}\n"
                f"  requested:  {expected}\n"
                "Choose a new experiment.name (recommended) instead of reusing this directory."
            )
        return expected

    if output_dir.is_dir():
        legacy_entries = [entry for entry in output_dir.iterdir() if entry.name != "_run"]
        if legacy_entries:
            raise RuntimeError(
                "Training output directory contains artifacts but no training_identity.json; "
                "it cannot be resumed safely. Choose a new experiment.name or migrate the "
                f"legacy run explicitly: {output_dir}"
            )
    return expected


def _artifact_evidence(root: Path, path: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise RuntimeError(f"Run artifact is outside its output directory: {path}") from error
    if not resolved_path.is_file():
        raise RuntimeError(f"Required run artifact is missing: {resolved_path}")
    return {
        "path": relative.as_posix(),
        "size_bytes": resolved_path.stat().st_size,
        "sha256": _file_sha256(resolved_path),
    }


def _artifact_evidence_matches(root: Path, evidence: dict[str, Any]) -> bool:
    relative = evidence.get("path")
    if not isinstance(relative, str):
        return False
    path = root / relative
    if not path.is_file() or path.stat().st_size != evidence.get("size_bytes"):
        return False
    return _file_sha256(path) == evidence.get("sha256")


def _write_fold_completion(
    fold_dir: Path,
    *,
    fold_idx: int,
    training_fingerprint: str,
    best_checkpoint: str,
) -> None:
    checkpoint = Path(best_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (Path.cwd() / checkpoint).resolve()
    artifacts = {
        "config": _artifact_evidence(fold_dir, fold_dir / "config.yaml"),
        "metrics": _artifact_evidence(fold_dir, fold_dir / "fold_metrics.json"),
        "val_predictions": _artifact_evidence(fold_dir, fold_dir / "preds_val.parquet"),
        "best_checkpoint": _artifact_evidence(fold_dir, checkpoint),
    }
    _write_json_atomic(
        fold_dir / _FOLD_COMPLETION_FILE,
        {
            "schema_version": _FOLD_COMPLETION_SCHEMA_VERSION,
            "status": "completed",
            "fold": fold_idx,
            "training_fingerprint": training_fingerprint,
            "artifacts": artifacts,
        },
    )


def _fold_complete(
    fold_dir: Path,
    *,
    fold_idx: int,
    training_fingerprint: str,
) -> bool:
    """Validate fold completion metadata and every recorded artifact hash."""
    record = _load_json_object(fold_dir / _FOLD_COMPLETION_FILE)
    if not record:
        return False
    if (
        record.get("schema_version") != _FOLD_COMPLETION_SCHEMA_VERSION
        or record.get("status") != "completed"
        or record.get("fold") != fold_idx
        or record.get("training_fingerprint") != training_fingerprint
    ):
        return False
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    required = {"config", "metrics", "val_predictions", "best_checkpoint"}
    return required <= artifacts.keys() and all(
        isinstance(artifacts[name], dict) and _artifact_evidence_matches(fold_dir, artifacts[name])
        for name in required
    )


def _write_training_completion(
    output_dir: Path,
    *,
    training_fingerprint: str,
    n_folds: int,
    skip_finalize: bool,
) -> None:
    for fold_idx in range(n_folds):
        if not _fold_complete(
            output_dir / f"fold_{fold_idx}",
            fold_idx=fold_idx,
            training_fingerprint=training_fingerprint,
        ):
            raise RuntimeError(f"Fold {fold_idx} lacks valid completion evidence")

    artifacts: dict[str, Any] = {
        "cv_summary": _artifact_evidence(output_dir, output_dir / "cv_summary.json"),
    }
    if not skip_finalize:
        artifacts["best_fold_checkpoint"] = _artifact_evidence(
            output_dir,
            output_dir / "final" / "best_fold" / "model.ckpt",
        )
    fold_completions = [
        _artifact_evidence(output_dir, output_dir / f"fold_{fold_idx}" / _FOLD_COMPLETION_FILE)
        for fold_idx in range(n_folds)
    ]
    _write_json_atomic(
        output_dir / _RUN_COMPLETION_FILE,
        {
            "schema_version": _RUN_COMPLETION_SCHEMA_VERSION,
            "status": "completed",
            "training_fingerprint": training_fingerprint,
            "n_folds": n_folds,
            "skip_finalize": skip_finalize,
            "artifacts": artifacts,
            "fold_completions": fold_completions,
        },
    )


def validate_training_run_dir(
    output_dir: str | Path,
    *,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Validate root and fold completion evidence for a finished training run."""
    output_dir = Path(output_dir)
    record = _load_json_object(output_dir / _RUN_COMPLETION_FILE)
    if not record:
        raise RuntimeError(f"Missing or invalid {output_dir / _RUN_COMPLETION_FILE}")
    fingerprint = record.get("training_fingerprint")
    if (
        record.get("schema_version") != _RUN_COMPLETION_SCHEMA_VERSION
        or record.get("status") != "completed"
        or not isinstance(fingerprint, str)
        or (expected_fingerprint is not None and fingerprint != expected_fingerprint)
    ):
        raise RuntimeError(f"Training completion identity is invalid: {output_dir}")

    artifacts = record.get("artifacts")
    fold_completions = record.get("fold_completions")
    n_folds = record.get("n_folds")
    if not isinstance(artifacts, dict) or not isinstance(fold_completions, list):
        raise RuntimeError(f"Training completion evidence is malformed: {output_dir}")
    if not isinstance(n_folds, int) or len(fold_completions) != n_folds:
        raise RuntimeError(f"Training completion fold count is invalid: {output_dir}")
    if not all(
        isinstance(evidence, dict) and _artifact_evidence_matches(output_dir, evidence)
        for evidence in [*artifacts.values(), *fold_completions]
    ):
        raise RuntimeError(f"Training completion artifact hash mismatch: {output_dir}")
    for fold_idx in range(n_folds):
        if not _fold_complete(
            output_dir / f"fold_{fold_idx}",
            fold_idx=fold_idx,
            training_fingerprint=fingerprint,
        ):
            raise RuntimeError(f"Fold {fold_idx} completion evidence is invalid")
    return record


def _training_run_complete(
    output_dir: Path,
    *,
    training_fingerprint: str,
) -> bool:
    try:
        validate_training_run_dir(
            output_dir,
            expected_fingerprint=training_fingerprint,
        )
    except RuntimeError:
        return False
    return True


def _setup_logging(cfg: DictConfig) -> None:
    level = logging.DEBUG if cfg.get("verbose", False) else logging.INFO
    exp_name = OmegaConf.select(cfg, "exp_name", default="train")
    fmt = f"%(asctime)s | %(levelname)-7s | exp={exp_name} | %(message)s"
    logging.basicConfig(level=level, format=fmt, force=True)


def _get_n_folds(cfg: DictConfig) -> int:
    """Determine fold count through the canonical planner."""
    from oceanpath.training import FoldPlanConfig, plan_folds

    plan = FoldPlanConfig(
        scheme=str(cfg.splits.scheme),
        n_folds=int(OmegaConf.select(cfg, "splits.n_folds", default=0) or 0),
        n_repeats=int(OmegaConf.select(cfg, "splits.n_repeats", default=0) or 0),
    )
    return len(plan_folds(plan))


def _get_output_dir(cfg: DictConfig) -> Path:
    """Resolve the output directory through the canonical path contract."""
    from oceanpath.config import FoundationPaths

    return FoundationPaths.from_config(cfg).train_dir


@contextmanager
def fold_context(fold_idx: int):
    """Context manager for fold execution with explicit cleanup."""
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"Fold {fold_idx} cleanup: {allocated:.2f}GB still allocated")


def _extract_metrics(trainer) -> dict:
    """
    Snapshot trainer.callback_metrics into a plain dict.

    Called IMMEDIATELY after validate/test to capture metrics before
    subsequent operations (e.g. trainer.test) overwrite them.
    """
    return {
        k: v.item() if isinstance(v, torch.Tensor) else v
        for k, v in trainer.callback_metrics.items()
    }


def _run_multi_crop_eval(trainer, module, datamodule, stage, n_crops, fold_dir, t):
    """Run K-crop evaluation and average slide-level probabilities.

    For each crop, we change the eval_crop_seed on the dataset so that
    spatial-stratified (or random) subsampling produces a different
    deterministic view.  After all crops, the averaged predictions are
    stored on the module so that save_predictions() writes the final
    averaged result.

    Note: Lightning callback_metrics after this function reflect the *last*
    crop only, not the averaged predictions.  For model selection during
    training (early stopping / checkpointing), use eval_n_crops=1.  Use
    K>1 only for final reporting after training is complete.
    """
    import pandas as pd

    ds = datamodule.val_dataset if stage == "val" else datamodule.test_dataset
    if ds is None:
        return

    pred_attr = "_val_predictions" if stage == "val" else "_test_predictions"

    all_crop_dfs: list[pd.DataFrame] = []
    for crop_idx in range(n_crops):
        datamodule.set_eval_crop_seed(ds, seed=crop_idx * 1000)
        if stage == "val":
            trainer.validate(module, datamodule=datamodule)
        else:
            trainer.test(module, datamodule=datamodule)
        # Grab the per-slide predictions from this crop
        preds = getattr(module, pred_attr)
        if preds:
            crop_df = pd.DataFrame(preds)
            all_crop_dfs.append(crop_df)
            preds.clear()

    if not all_crop_dfs:
        return

    # Average probabilities across crops, grouped by slide_id
    combined = pd.concat(all_crop_dfs, ignore_index=True)
    prob_cols = [c for c in combined.columns if c.startswith("prob_")]
    agg = dict.fromkeys(prob_cols, "mean")
    agg["label"] = "first"
    averaged = combined.groupby("slide_id", sort=False).agg(agg).reset_index()

    # Re-populate module predictions with the averaged values
    setattr(module, pred_attr, averaged.to_dict("records"))

    # Reset seed back to 0 for consistency
    datamodule.set_eval_crop_seed(ds, seed=0)

    logger.info(
        "Multi-crop eval (%s): %d crops x %d slides -> averaged predictions",
        stage,
        n_crops,
        len(averaged),
    )


def _print_fold_report(fold_idx: int, fold_metrics: dict) -> None:
    """Print a structured fold report with key val and test metrics."""
    print(f"\n{'─' * 60}")
    print(f"  Fold {fold_idx} Report")
    print(f"{'─' * 60}")

    for prefix, label in [("val", "Val"), ("test", "Test")]:
        metrics_present = [k for k in fold_metrics if k.startswith(f"{prefix}/")]
        if not metrics_present:
            continue

        parts = []
        for key in ["loss", "auroc", "acc", "precision", "recall", "f1", "balanced_acc"]:
            full = f"{prefix}/{key}"
            val = fold_metrics.get(full)
            if val is not None:
                parts.append(f"{key}={val:.4f}")

        if parts:
            print(f"  {label:5s}: {', '.join(parts)}")

    best_epoch = fold_metrics.get("best_epoch", "?")
    print(f"  Best epoch: {best_epoch}")
    print(f"{'─' * 60}")


# ── Fold runner ───────────────────────────────────────────────────────────────


def run_fold(
    cfg: DictConfig,
    fold_idx: int,
    output_dir: Path,
) -> dict:
    """
    Train a single fold.

    Returns fold-level metrics dict containing BOTH val/ and test/ keys.
    """
    from oceanpath.datasets import MILDataModule
    from oceanpath.training.callbacks import (
        BagCurriculumCallback,
        BatchIndexLogger,
        FoldTimingCallback,
        MetadataWriter,
        WandbFoldSummary,
    )
    from oceanpath.training.lightning import MILTrainModule

    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    t = cfg.training

    # Save resolved config for this fold
    config_path = fold_dir / "config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))

    # ── Seed ──────────────────────────────────────────────────────────────
    L.seed_everything(t.seed + fold_idx, workers=True)

    # ── DataModule ────────────────────────────────────────────────────────
    from oceanpath.config import FoundationPaths

    splits_dir = str(FoundationPaths.from_config(cfg).splits_dir)

    datamodule = MILDataModule(
        mmap_dir=cfg.data.mmap_dir,
        splits_dir=splits_dir,
        csv_path=cfg.data.csv_path,
        label_column=cfg.data.label_columns[0],
        filename_column=cfg.data.filename_column,
        scheme=cfg.splits.scheme,
        fold=fold_idx,
        batch_size=t.batch_size,
        max_instances=t.max_instances,
        dataset_max_instances=OmegaConf.select(cfg, "training.dataset_max_instances", default=None),
        num_workers=cfg.platform.num_workers,
        class_weighted_sampling=t.class_weighted_sampling,
        instance_dropout=t.instance_dropout,
        feature_noise_std=t.feature_noise_std,
        cache_size_mb=t.cache_size_mb,
        return_coords=t.return_coords,
        verify_splits=t.verify_splits,
        use_preallocated_collator=t.use_preallocated_collator,
        cap_strategy=OmegaConf.select(cfg, "training.cap_strategy", default="random"),
        cap_grid_size=OmegaConf.select(cfg, "training.cap_grid_size", default=32),
        eval_n_crops=OmegaConf.select(cfg, "training.eval_n_crops", default=1),
        length_bucket=OmegaConf.select(cfg, "training.length_bucket", default=False),
        length_bucket_size=OmegaConf.select(cfg, "training.length_bucket_size", default=64),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    module = MILTrainModule(
        arch=cfg.model.arch,
        in_dim=cfg.encoder.feature_dim,
        num_classes=len(cfg.data.label_columns)
        if len(cfg.data.label_columns) > 1
        else datamodule.num_classes,
        model_cfg=model_cfg,
        lr=t.lr,
        weight_decay=t.weight_decay,
        lr_scheduler=t.lr_scheduler,
        warmup_epochs=t.warmup_epochs,
        max_epochs=t.max_epochs,
        loss_type=t.loss_type,
        class_weights=OmegaConf.select(cfg, "training.class_weights", default=None),
        focal_gamma=t.focal_gamma,
        monitor_metric=t.monitor_metric,
        monitor_mode=t.monitor_mode,
        canary_interval=t.canary_interval,
        compile_model=t.compile_model,
        freeze_aggregator=t.freeze_aggregator,
        collect_embeddings=t.collect_embeddings,
        aggregator_weights_path=OmegaConf.select(
            cfg, "training.aggregator_weights_path", default=None
        ),
        aggregator_lr=OmegaConf.select(cfg, "training.aggregator_lr", default=None),
        head_lr=OmegaConf.select(cfg, "training.head_lr", default=None),
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    ckpt_filename = "best-{epoch}-{" + t.monitor_metric + ":.4f}"

    es_min_delta = t.get("early_stopping_min_delta", 0.0)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(fold_dir / "checkpoints"),
            filename=ckpt_filename,
            monitor=t.monitor_metric,
            mode=t.monitor_mode,
            save_top_k=t.save_top_k,
            save_last=t.save_last,
            verbose=True,
        ),
        EarlyStopping(
            monitor=t.monitor_metric,
            mode=t.monitor_mode,
            patience=t.early_stopping_patience,
            min_delta=es_min_delta,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        FoldTimingCallback(),
        BatchIndexLogger(output_dir=str(fold_dir)),
        WandbFoldSummary(
            fold=fold_idx, monitor_metric=t.monitor_metric, monitor_mode=t.monitor_mode
        ),
        MetadataWriter(
            output_dir=str(fold_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        ),
    ]

    logger.info(
        f"Monitor: {t.monitor_metric} (mode={t.monitor_mode}), "
        f"early_stopping_patience={t.early_stopping_patience}, "
        f"min_delta={es_min_delta}"
    )

    if t.bag_curriculum:
        callbacks.append(
            BagCurriculumCallback(
                start_instances=t.bag_curriculum_start,
                end_instances=t.bag_curriculum_end,
                warmup_epochs=t.bag_curriculum_warmup,
            )
        )

    with suppress(Exception):
        callbacks.append(RichProgressBar())

    # ── Logger ────────────────────────────────────────────────────────────
    from oceanpath.runtime import build_lightning_logger

    wandb_logger = build_lightning_logger(
        cfg,
        save_dir=fold_dir,
        name=f"{cfg.exp_name}/fold_{fold_idx}",
        extra_tags=(f"fold_{fold_idx}",),
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=t.max_epochs,
        accelerator=cfg.platform.accelerator,
        devices=cfg.platform.devices,
        strategy=cfg.platform.strategy,
        precision=cfg.platform.precision,
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=t.gradient_clip_val,
        accumulate_grad_batches=t.accumulate_grad_batches,
        deterministic=t.deterministic,
        default_root_dir=str(fold_dir),
        enable_checkpointing=True,
        log_every_n_steps=1,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info(f"{'=' * 60}")
    logger.info(f"  FOLD {fold_idx} — training")
    logger.info(f"{'=' * 60}")

    trainer.fit(module, datamodule=datamodule)

    # ── Load best checkpoint ──────────────────────────────────────────────
    best_ckpt = trainer.checkpoint_callback.best_model_path
    if not best_ckpt:
        last_ckpt = fold_dir / "checkpoints" / "last.ckpt"
        if last_ckpt.is_file():
            best_ckpt = str(last_ckpt)
            logger.warning(f"No best checkpoint found, falling back to {best_ckpt}")

    # ── Collect val metrics and predictions ───────────────────────────────
    #
    # CRITICAL: we snapshot val metrics BEFORE running test, because
    # trainer.test() OVERWRITES callback_metrics — wiping all val/ keys.
    #
    val_metrics = {}
    if best_ckpt:
        logger.info(f"Loading best checkpoint: {best_ckpt}")
        try:
            best_module = MILTrainModule.load_from_checkpoint(
                best_ckpt,
                weights_only=False,
            )
        except TypeError:
            best_module = MILTrainModule.load_from_checkpoint(best_ckpt)
        best_module.eval()

        # Run validation → populates callback_metrics with val/ keys
        eval_n_crops = OmegaConf.select(cfg, "training.eval_n_crops", default=1)
        if eval_n_crops > 1 and datamodule.val_dataset is not None:
            # Multi-crop evaluation: run K passes with different seeds, average
            _run_multi_crop_eval(trainer, best_module, datamodule, "val", eval_n_crops, fold_dir, t)
        else:
            trainer.validate(best_module, datamodule=datamodule)

        # ═══ SNAPSHOT val metrics NOW ═══
        val_metrics = _extract_metrics(trainer)

        # Save val predictions + embeddings
        best_module.save_predictions(str(fold_dir), prefix="val")
        if t.collect_embeddings:
            best_module.save_embeddings(str(fold_dir), prefix="val")

        # ── Run test (this OVERWRITES callback_metrics) ──────────────
        test_metrics = {}
        try:
            datamodule.setup(stage="test")
            if datamodule.test_dataset is not None and len(datamodule.test_dataset) > 0:
                trainer.test(best_module, datamodule=datamodule)
                # Snapshot test metrics
                test_metrics = _extract_metrics(trainer)
                best_module.save_predictions(str(fold_dir), prefix="test")
                if t.collect_embeddings:
                    best_module.save_embeddings(str(fold_dir), prefix="test")
        except RuntimeError as e:
            logger.info(f"No test set for this scheme/fold: {e}")

        del best_module
    else:
        logger.warning("No checkpoint found — skipping OOF prediction collection")
        test_metrics = {}

    # ── Assemble fold metrics (val + test merged) ─────────────────────────
    #
    # Start from val_metrics (which has val/loss, val/auroc, etc.)
    # then add test metrics (test/loss, test/auroc, etc.)
    # This ensures val/ keys are ALWAYS present for finalize_models.
    #
    fold_metrics = {}
    fold_metrics.update(val_metrics)  # val/loss, val/auroc, ...
    fold_metrics.update(test_metrics)  # test/loss, test/auroc, ...
    fold_metrics["fold"] = fold_idx
    fold_metrics["best_checkpoint"] = best_ckpt

    # ── Extract best_epoch ────────────────────────────────────────────────
    from oceanpath.workflows.finalize import parse_best_epoch_from_checkpoint

    best_epoch = parse_best_epoch_from_checkpoint(best_ckpt)
    if best_epoch is None:
        es_cb = trainer.early_stopping_callback
        if es_cb is not None and hasattr(es_cb, "stopped_epoch") and es_cb.stopped_epoch > 0:
            best_epoch = max(1, es_cb.stopped_epoch - es_cb.patience + 1)
        else:
            best_epoch = trainer.current_epoch
    fold_metrics["best_epoch"] = best_epoch

    # Also store the best checkpoint's monitored score explicitly
    fold_metrics["best_model_score"] = (
        trainer.checkpoint_callback.best_model_score.item()
        if trainer.checkpoint_callback.best_model_score is not None
        else None
    )

    # ── Save fold metrics ─────────────────────────────────────────────────
    metrics_path = fold_dir / "fold_metrics.json"
    metrics_path.write_text(json.dumps(fold_metrics, indent=2, default=str))

    # Write the completion signal last. A prediction file by itself is not
    # sufficient evidence that the matching config and checkpoint survived.
    _write_fold_completion(
        fold_dir,
        fold_idx=fold_idx,
        training_fingerprint=training_run_fingerprint(cfg),
        best_checkpoint=str(best_ckpt),
    )

    # ── Print fold report ─────────────────────────────────────────────────
    _print_fold_report(fold_idx, fold_metrics)

    # ── Close W&B ─────────────────────────────────────────────────────────
    if wandb_logger is not None:
        from oceanpath.runtime import finish_run

        finish_run()

    return fold_metrics


# ── Aggregate cross-fold results ──────────────────────────────────────────────


def aggregate_cv_results(output_dir: Path, n_folds: int) -> dict:
    """
    Aggregate per-fold metrics into a CV summary.

    Computes mean ± std for all metrics across folds.
    Concatenates OOF predictions into a single parquet.
    """
    import pandas as pd

    all_metrics = []
    all_val_preds = []
    all_test_preds = []

    for fold_idx in range(n_folds):
        fold_dir = output_dir / f"fold_{fold_idx}"

        metrics_path = fold_dir / "fold_metrics.json"
        if metrics_path.is_file():
            all_metrics.append(json.loads(metrics_path.read_text()))

        val_path = fold_dir / "preds_val.parquet"
        if val_path.is_file():
            df = pd.read_parquet(val_path)
            df["fold"] = fold_idx
            all_val_preds.append(df)

        test_path = fold_dir / "preds_test.parquet"
        if test_path.is_file():
            df = pd.read_parquet(test_path)
            df["fold"] = fold_idx
            all_test_preds.append(df)

    # Aggregate metrics
    summary = {}
    if all_metrics:
        import numpy as np

        numeric_keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))]
        for key in numeric_keys:
            values = [m[key] for m in all_metrics if key in m and isinstance(m[key], (int, float))]
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))

    # Concatenate OOF predictions
    if all_val_preds:
        oof_df = pd.concat(all_val_preds, ignore_index=True)
        oof_path = output_dir / "oof_predictions.parquet"
        oof_df.to_parquet(str(oof_path), index=False, engine="pyarrow")
        logger.info(f"OOF predictions: {len(oof_df)} slides → {oof_path}")
        summary["n_oof_slides"] = len(oof_df)

    if all_test_preds:
        test_df = pd.concat(all_test_preds, ignore_index=True)
        test_path = output_dir / "test_predictions.parquet"
        test_df.to_parquet(str(test_path), index=False, engine="pyarrow")
        logger.info(f"Test predictions: {len(test_df)} slides → {test_path}")

    # Save summary
    summary_path = output_dir / "cv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"CV summary → {summary_path}")

    return summary


# ── Main ──────────────────────────────────────────────────────────────────────


def _run_training(cfg: DictConfig) -> None:
    _setup_logging(cfg)

    # Silence the "Tensor Cores" warning and enable TF32 for faster matmuls
    torch.set_float32_matmul_precision("high")

    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    output_dir = _get_output_dir(cfg)
    n_folds = _get_n_folds(cfg)
    scheme = cfg.splits.scheme
    training_fingerprint = training_run_fingerprint(cfg)

    # ── Dry run ───────────────────────────────────────────────────────────
    if cfg.get("dry_run", False):
        from oceanpath.config import FoundationPaths
        from oceanpath.datasets import MILDataModule

        splits_dir = str(FoundationPaths.from_config(cfg).splits_dir)
        dm = MILDataModule(
            mmap_dir=cfg.data.mmap_dir,
            splits_dir=splits_dir,
            csv_path=cfg.data.csv_path,
            label_column=cfg.data.label_columns[0],
            filename_column=cfg.data.filename_column,
            scheme=scheme,
            fold=0,
            batch_size=cfg.training.batch_size,
            max_instances=cfg.training.max_instances,
            num_workers=0,
            verify_splits=cfg.training.verify_splits,
        )
        dm.setup(stage="fit")

        print(f"\n{'=' * 60}")
        print("  DRY RUN — train.py")
        print(f"{'=' * 60}")
        print(f"  Experiment:  {cfg.exp_name}")
        print(f"  Output:      {output_dir}")
        print(f"  Scheme:      {scheme}")
        print(f"  N folds:     {n_folds}")
        print(f"  Model:       {cfg.model.arch}")
        print(f"  Encoder:     {cfg.encoder.name} (D={cfg.encoder.feature_dim})")
        print(f"  Num classes: {dm.num_classes}")
        print(f"  Train size:  {len(dm.train_dataset)} (fold 0)")
        print(f"  Val size:    {len(dm.val_dataset)} (fold 0)")
        print(f"  Max epochs:  {cfg.training.max_epochs}")
        print(f"  LR:          {cfg.training.lr}")
        print(f"  Batch size:  {cfg.training.batch_size}")
        print(f"{'=' * 60}\n")
        return

    if not cfg.training.force_rerun and _training_run_complete(
        output_dir,
        training_fingerprint=training_fingerprint,
    ):
        logger.info(
            "Training run is already complete with verified evidence (fp=%s); skipping",
            training_fingerprint,
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_training_identity(cfg, output_dir)
    (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    # ── Train all folds ───────────────────────────────────────────────────
    logger.info(f"Starting {n_folds}-fold training: {cfg.exp_name}")
    start_time = time.monotonic()

    all_fold_metrics = []

    for fold_idx in range(n_folds):
        fold_dir = output_dir / f"fold_{fold_idx}"

        # Fold-level resume
        if not cfg.training.force_rerun and _fold_complete(
            fold_dir,
            fold_idx=fold_idx,
            training_fingerprint=training_fingerprint,
        ):
            logger.info(
                "Fold %d already complete with verified config/checkpoint evidence — skipping",
                fold_idx,
            )
            metrics_path = fold_dir / "fold_metrics.json"
            if metrics_path.is_file():
                all_fold_metrics.append(json.loads(metrics_path.read_text()))
            continue

        # ── Clean up old fold outputs on force_rerun ──────────────────
        if cfg.training.force_rerun and fold_dir.exists():
            import shutil

            logger.info(f"force_rerun: cleaning {fold_dir}")
            shutil.rmtree(fold_dir)

        with fold_context(fold_idx):
            fold_metrics = run_fold(
                cfg=cfg,
                fold_idx=fold_idx,
                output_dir=output_dir,
            )
            all_fold_metrics.append(fold_metrics)

    # ── Aggregate results ─────────────────────────────────────────────────
    total_time = time.monotonic() - start_time
    summary = aggregate_cv_results(output_dir, n_folds)

    # ── Phase 2: Finalize models ──────────────────────────────────────────
    skip_finalize = OmegaConf.select(cfg, "training.skip_finalize", default=False)
    finalize_results: dict[str, Any] | None = None
    if not skip_finalize and len(all_fold_metrics) == n_folds:
        from oceanpath.workflows.finalize import finalize_models

        try:
            finalize_results = finalize_models(
                cfg=cfg,
                output_dir=output_dir,
                n_folds=n_folds,
                all_fold_metrics=all_fold_metrics,
            )
            summary["finalize"] = {
                k: v.get("model_path", v.get("error", "unknown"))
                for k, v in finalize_results.items()
            }
        except Exception as e:
            logger.error(f"Finalization failed: {e}", exc_info=True)
            summary["finalize_error"] = str(e)
    elif skip_finalize:
        logger.info("Skipping finalization (skip_finalize=true)")
    else:
        logger.warning(
            f"Skipping finalization: only {len(all_fold_metrics)}/{n_folds} folds completed"
        )

    # Aggregate writes an initial summary before finalization. Rewrite it with
    # finalization status, then publish the root completion record only after
    # all required fold and best-model evidence validates.
    summary_path = output_dir / "cv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    if len(all_fold_metrics) != n_folds:
        raise RuntimeError(
            f"Training completed only {len(all_fold_metrics)}/{n_folds} folds; "
            "no completion signal was written"
        )
    if not skip_finalize:
        best_fold = (finalize_results or {}).get("best_fold", {})
        best_model = output_dir / "final" / "best_fold" / "model.ckpt"
        if "error" in best_fold or not best_model.is_file():
            raise RuntimeError(
                "Best-fold finalization failed; no training completion signal was written"
            )
    _write_training_completion(
        output_dir,
        training_fingerprint=training_fingerprint,
        n_folds=n_folds,
        skip_finalize=bool(skip_finalize),
    )

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Training complete: {cfg.exp_name}")
    print(f"{'=' * 60}")
    print(f"  Output:         {output_dir}")
    print(f"  Folds:          {n_folds}")
    print(f"  Total time:     {total_time:.0f}s ({total_time / 60:.1f}min)")

    # Print mean ± std for all val and test metrics
    for key in [
        "val/loss",
        "val/auroc",
        "val/acc",
        "val/precision",
        "val/recall",
        "val/f1",
        "val/balanced_acc",
        "test/loss",
        "test/auroc",
        "test/acc",
        "test/precision",
        "test/recall",
        "test/f1",
        "test/balanced_acc",
    ]:
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        if mean_key in summary:
            print(f"  {key:22s}: {summary[mean_key]:.4f} ± {summary[std_key]:.4f}")

    print(f"{'=' * 60}\n")


def run_training(cfg: DictConfig) -> StageResult:
    """Run the canonical supervised training workflow."""
    from oceanpath.config import FoundationPaths
    from oceanpath.runtime import run_context

    paths = FoundationPaths.from_config(cfg)
    started = time.monotonic()
    dry_run = bool(cfg.get("dry_run", False))
    training_fingerprint = _assert_training_output_identity(cfg, paths.train_dir)
    with run_context(
        cfg,
        stage="train_model",
        output_dir=paths.train_dir,
        persist=not dry_run,
    ):
        _run_training(cfg)
    return StageResult(
        stage=PipelineStage.TRAIN_MODEL,
        status="dry_run" if dry_run else "completed",
        output_dir=paths.train_dir,
        elapsed_seconds=time.monotonic() - started,
        details={
            "experiment_name": paths.experiment_name,
            "training_fingerprint": training_fingerprint,
        },
    )
