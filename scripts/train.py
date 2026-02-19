"""
MIL training orchestrator with fold-level resume.

Runs all folds of a CV scheme, with:
  - Fold-level resume: skip completed folds (checks preds_val.parquet)
  - Explicit memory cleanup between folds
  - W&B logging with fold grouping
  - OOF prediction + embedding collection
  - Split integrity verification
  - Deterministic replay (batch index logging)
  - Phase 2 finalization: best_fold, ensemble, refit

Usage:
    # Standard 5-fold ABMIL
    python scripts/train.py platform=local data=blca encoder=uni2 splits=kfold5 model=abmil

    # TransMIL with holdout split
    python scripts/train.py platform=local data=blca encoder=uni2 splits=holdout_80_10_10 model=transmil

    # Override training hyperparams
    python scripts/train.py ... training.lr=1e-4 training.max_epochs=50

    # Force rerun of completed folds
    python scripts/train.py ... training.force_rerun=true

    # Dry run: validate configs and data, don't train
    python scripts/train.py ... dry_run=true
"""

import gc
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _setup_logging(cfg: DictConfig) -> None:
    level = logging.DEBUG if cfg.get("verbose", False) else logging.INFO
    exp_name = OmegaConf.select(cfg, "exp_name", default="train")
    fmt = (
        "%(asctime)s | %(levelname)-7s | "
        f"exp={exp_name} | "
        "%(message)s"
    )
    logging.basicConfig(level=level, format=fmt, force=True)


def _get_n_folds(cfg: DictConfig) -> int:
    """Determine number of folds/repeats from split scheme."""
    scheme = cfg.splits.scheme
    if scheme in ("holdout", "custom_holdout"):
        return 1
    elif scheme in ("kfold", "custom_kfold"):
        return cfg.splits.n_folds
    elif scheme == "monte_carlo":
        return cfg.splits.n_repeats
    elif scheme == "nested_cv":
        return cfg.splits.n_folds
    else:
        raise ValueError(f"Unknown scheme: {scheme}")


def _get_output_dir(cfg: DictConfig) -> Path:
    """Compute the output directory for this experiment."""
    return Path(cfg.platform.output_root) / "train" / cfg.exp_name


def _fold_complete(fold_dir: Path) -> bool:
    """Check if a fold has completed (preds_val.parquet exists)."""
    return (fold_dir / "preds_val.parquet").is_file()


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
    outer_fold: Optional[int] = None,
) -> dict:
    """
    Train a single fold.

    Returns fold-level metrics dict containing BOTH val/ and test/ keys.
    """
    from oceanpath.data.datamodule import MILDataModule
    from oceanpath.modules.train_module import MILTrainModule
    from oceanpath.modules.callbacks import (
        BagCurriculumCallback,
        BatchIndexLogger,
        FoldTimingCallback,
        MetadataWriter,
        WandbFoldSummary,
    )

    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    t = cfg.training

    # Save resolved config for this fold
    config_path = fold_dir / "config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))

    # ── Seed ──────────────────────────────────────────────────────────────
    L.seed_everything(t.seed + fold_idx, workers=True)

    # ── DataModule ────────────────────────────────────────────────────────
    splits_dir = str(Path(cfg.platform.splits_root) / cfg.data.name / cfg.splits.name)

    datamodule = MILDataModule(
        mmap_dir=cfg.data.mmap_dir,
        splits_dir=splits_dir,
        csv_path=cfg.data.csv_path,
        label_column=cfg.data.label_columns[0],
        filename_column=cfg.data.filename_column,
        scheme=cfg.splits.scheme,
        fold=fold_idx,
        outer_fold=outer_fold,
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
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    module = MILTrainModule(
        arch=cfg.model.arch,
        in_dim=cfg.encoder.feature_dim,
        num_classes=len(cfg.data.label_columns) if len(cfg.data.label_columns) > 1 else datamodule.num_classes,
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
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    ckpt_filename = "best-{epoch}-{" + t.monitor_metric + ":.4f}"

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
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        FoldTimingCallback(),
        BatchIndexLogger(output_dir=str(fold_dir)),
        WandbFoldSummary(fold=fold_idx, monitor_metric=t.monitor_metric, monitor_mode=t.monitor_mode),
        MetadataWriter(
            output_dir=str(fold_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        ),
    ]

    logger.info(
        f"Monitor: {t.monitor_metric} (mode={t.monitor_mode}), "
        f"early_stopping_patience={t.early_stopping_patience}"
    )

    if t.bag_curriculum:
        callbacks.append(BagCurriculumCallback(
            start_instances=t.bag_curriculum_start,
            end_instances=t.bag_curriculum_end,
            warmup_epochs=t.bag_curriculum_warmup,
        ))

    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass

    # ── Logger ────────────────────────────────────────────────────────────
    wandb_logger = None
    if not cfg.wandb.get("offline", False):
        try:
            wandb_logger = WandbLogger(
                project=cfg.wandb.project,
                entity=OmegaConf.select(cfg, "wandb.entity", default=None),
                group=cfg.wandb.group,
                name=f"{cfg.exp_name}/fold_{fold_idx}",
                tags=list(cfg.wandb.tags) + [f"fold_{fold_idx}"],
                save_dir=str(fold_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception as e:
            logger.warning(f"W&B init failed: {e}. Training without W&B.")

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
                best_ckpt, weights_only=False,
            )
        except TypeError:
            best_module = MILTrainModule.load_from_checkpoint(best_ckpt)
        best_module.eval()

        # Run validation → populates callback_metrics with val/ keys
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
    fold_metrics.update(val_metrics)       # val/loss, val/auroc, ...
    fold_metrics.update(test_metrics)      # test/loss, test/auroc, ...
    fold_metrics["fold"] = fold_idx
    fold_metrics["best_checkpoint"] = best_ckpt

    # ── Extract best_epoch ────────────────────────────────────────────────
    from oceanpath.modules.finalize import parse_best_epoch_from_checkpoint
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

    # ── Print fold report ─────────────────────────────────────────────────
    _print_fold_report(fold_idx, fold_metrics)

    # ── Close W&B ─────────────────────────────────────────────────────────
    if wandb_logger:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

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
        numeric_keys = [
            k for k in all_metrics[0]
            if isinstance(all_metrics[0][k], (int, float))
        ]
        for key in numeric_keys:
            values = [
                m[key] for m in all_metrics
                if key in m and isinstance(m[key], (int, float))
            ]
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


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    _setup_logging(cfg)

    # Silence the "Tensor Cores" warning and enable TF32 for faster matmuls
    torch.set_float32_matmul_precision("high")

    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    output_dir = _get_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full config
    (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    n_folds = _get_n_folds(cfg)
    scheme = cfg.splits.scheme

    # ── Dry run ───────────────────────────────────────────────────────────
    if cfg.get("dry_run", False):
        from oceanpath.data.datamodule import MILDataModule

        splits_dir = str(Path(cfg.platform.splits_root) / cfg.data.name / cfg.splits.name)
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
        print(f"  DRY RUN — train.py")
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

    # ── Train all folds ───────────────────────────────────────────────────
    logger.info(f"Starting {n_folds}-fold training: {cfg.exp_name}")
    start_time = time.monotonic()

    all_fold_metrics = []

    for fold_idx in range(n_folds):
        fold_dir = output_dir / f"fold_{fold_idx}"

        # Fold-level resume
        if _fold_complete(fold_dir) and not cfg.training.force_rerun:
            logger.info(
                f"Fold {fold_idx} already complete — skipping. "
                f"(found {fold_dir / 'preds_val.parquet'})"
            )
            metrics_path = fold_dir / "fold_metrics.json"
            if metrics_path.is_file():
                all_fold_metrics.append(json.loads(metrics_path.read_text()))
            continue

        with fold_context(fold_idx):
            fold_metrics = run_fold(
                cfg=cfg,
                fold_idx=fold_idx,
                output_dir=output_dir,
                outer_fold=None,
            )
            all_fold_metrics.append(fold_metrics)

    # ── Aggregate results ─────────────────────────────────────────────────
    total_time = time.monotonic() - start_time
    summary = aggregate_cv_results(output_dir, n_folds)

    # ── Phase 2: Finalize models ──────────────────────────────────────────
    skip_finalize = OmegaConf.select(cfg, "training.skip_finalize", default=False)
    if not skip_finalize and len(all_fold_metrics) == n_folds:
        from oceanpath.modules.finalize import finalize_models
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

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Training complete: {cfg.exp_name}")
    print(f"{'=' * 60}")
    print(f"  Output:         {output_dir}")
    print(f"  Folds:          {n_folds}")
    print(f"  Total time:     {total_time:.0f}s ({total_time / 60:.1f}min)")

    # Print mean ± std for all val and test metrics
    for key in [
        "val/loss", "val/auroc", "val/acc", "val/precision", "val/recall",
        "val/f1", "val/balanced_acc",
        "test/loss", "test/auroc", "test/acc", "test/precision", "test/recall",
        "test/f1", "test/balanced_acc",
    ]:
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        if mean_key in summary:
            print(f"  {key:22s}: {summary[mean_key]:.4f} ± {summary[std_key]:.4f}")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()