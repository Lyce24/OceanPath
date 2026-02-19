"""
Model finalization after cross-validation (Phase 2 of Stage 5).

After all CV folds complete, produces THREE final model artifacts:

  1. best_fold — single fold checkpoint with the best validation score
  2. ensemble  — all K fold checkpoints packaged for ensemble inference
  3. refit     — fresh model retrained on ALL training data (no val split)

All three are saved so Stage 6 (evaluate.py) can load and compare them on a
held-out test set and pick the winner.

Output structure
════════════════════════════════════════════════════════════════════════════
  output_dir/final/
  ├── best_fold/
  │   ├── model.ckpt          # Lightning checkpoint
  │   └── info.json           # selection metadata
  ├── ensemble/
  │   ├── fold_0.ckpt ... fold_{K-1}.ckpt
  │   └── info.json           # model config + fold scores
  ├── refit/
  │   ├── model.ckpt
  │   └── info.json           # epoch rule, training stats
  └── finalize_summary.json   # top-level summary of all three

Stage 6 contract
════════════════════════════════════════════════════════════════════════════
  Each info.json has a 'strategy' field (best_fold | ensemble | refit)
  and a 'model_path' (or list of paths for ensemble).
  Stage 6 reads info.json to know HOW to load and run inference:
    best_fold / refit → MILTrainModule.load_from_checkpoint(model_path)
    ensemble          → load each fold_*.ckpt, average softmax probs
"""

import gc
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, RichProgressBar
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════


def finalize_models(
    cfg: DictConfig,
    output_dir: Path,
    n_folds: int,
    all_fold_metrics: list[dict],
) -> dict:
    """
    Phase 2 entry point: produce best_fold, ensemble, and refit artifacts.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config (same as used for fold training).
    output_dir : Path
        Experiment output directory (contains fold_0/, fold_1/, ...).
    n_folds : int
        Number of CV folds.
    all_fold_metrics : list[dict]
        Per-fold metrics dicts from Phase 1. Each must have
        'best_checkpoint' (str path) and 'best_epoch' (int) keys.

    Returns
    -------
    dict with keys 'best_fold', 'ensemble', 'refit' → info dicts.
    Each info dict has 'error' key if that strategy failed.
    """
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Phase 2: Finalizing models")
    logger.info("=" * 60)

    results = {}

    # ── 1. Best fold (fast — just copies a checkpoint) ───────────────────
    results["best_fold"] = _safe_run(
        "best_fold", _save_best_fold, cfg, final_dir, all_fold_metrics,
    )

    # ── 2. Ensemble (fast — copies K checkpoints) ────────────────────────
    results["ensemble"] = _safe_run(
        "ensemble", _save_ensemble, cfg, final_dir, n_folds, all_fold_metrics,
    )

    # ── 3. Refit (slow — full training run) ──────────────────────────────
    results["refit"] = _safe_run(
        "refit", _run_refit, cfg, final_dir, all_fold_metrics,
    )

    # ── Save summary ─────────────────────────────────────────────────────
    summary_path = final_dir / "finalize_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Finalization summary → {summary_path}")

    _print_summary(results)

    return results


def _safe_run(name: str, fn, *args) -> dict:
    """Run a finalization step, catching and logging errors."""
    try:
        return fn(*args)
    except Exception as e:
        logger.error(f"{name} failed: {e}", exc_info=True)
        return {"strategy": name, "error": str(e)}


# ═════════════════════════════════════════════════════════════════════════════
# 1. Best fold
# ═════════════════════════════════════════════════════════════════════════════


def _save_best_fold(
    cfg: DictConfig,
    final_dir: Path,
    all_fold_metrics: list[dict],
) -> dict:
    """
    Copy the single best fold checkpoint.

    Selection uses the same monitor_metric / monitor_mode as training.
    """
    t = cfg.training
    metric_key = "test/loss"
    mode = t.monitor_mode

    # ── Find best fold ───────────────────────────────────────────────────
    best_idx = None
    best_score = float("inf") if mode == "min" else float("-inf")

    for i, fm in enumerate(all_fold_metrics):
        score = fm.get(metric_key)
        if score is None:
            logger.warning(f"Fold {i} missing metric '{metric_key}' — skipping")
            continue
        is_better = (score < best_score) if mode == "min" else (score > best_score)
        if is_better:
            best_score = score
            best_idx = i

    if best_idx is None:
        raise ValueError(
            f"No fold has metric '{metric_key}'. "
            f"Available keys: {list(all_fold_metrics[0].keys()) if all_fold_metrics else '(empty)'}"
        )

    # ── Copy checkpoint ──────────────────────────────────────────────────
    src_ckpt = all_fold_metrics[best_idx].get("best_checkpoint", "")
    if not src_ckpt or not Path(src_ckpt).is_file():
        raise FileNotFoundError(
            f"Fold {best_idx} checkpoint not found: '{src_ckpt}'"
        )

    dest_dir = final_dir / "best_fold"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_ckpt = dest_dir / "model.ckpt"
    shutil.copy2(src_ckpt, dest_ckpt)

    info = {
        "strategy": "best_fold",
        "source_fold": best_idx,
        "monitor_metric": metric_key,
        "monitor_mode": mode,
        "best_score": float(best_score),
        "best_epoch": all_fold_metrics[best_idx].get("best_epoch"),
        "source_checkpoint": str(src_ckpt),
        "model_path": str(dest_ckpt),
    }
    _write_info(dest_dir, info)
    logger.info(
        f"best_fold: fold {best_idx} "
        f"({metric_key}={best_score:.4f}, epoch={info['best_epoch']}) → {dest_ckpt}"
    )
    return info


# ═════════════════════════════════════════════════════════════════════════════
# 2. Ensemble
# ═════════════════════════════════════════════════════════════════════════════


def _save_ensemble(
    cfg: DictConfig,
    final_dir: Path,
    n_folds: int,
    all_fold_metrics: list[dict],
) -> dict:
    """
    Copy all fold checkpoints into an ensemble directory.

    Stage 6 loads each, runs inference, and averages softmax probabilities.
    """
    dest_dir = final_dir / "ensemble"
    dest_dir.mkdir(parents=True, exist_ok=True)

    fold_details = []
    copied = 0

    for i, fm in enumerate(all_fold_metrics):
        src_ckpt = fm.get("best_checkpoint", "")
        if not src_ckpt or not Path(src_ckpt).is_file():
            logger.warning(f"Fold {i} checkpoint missing — skipping in ensemble")
            fold_details.append({"fold": i, "error": "checkpoint not found"})
            continue

        dest_ckpt = dest_dir / f"fold_{i}.ckpt"
        shutil.copy2(src_ckpt, dest_ckpt)
        copied += 1

        fold_details.append({
            "fold": i,
            "model_path": str(dest_ckpt),
            "best_epoch": fm.get("best_epoch"),
            "monitor_score": fm.get("test/loss", float("inf")),
        })

    if copied == 0:
        raise FileNotFoundError("No fold checkpoints found — cannot create ensemble")

    info = {
        "strategy": "ensemble",
        "n_folds": n_folds,
        "n_checkpoints": copied,
        "monitor_metric": "test/loss",
        "ensemble_method": "mean_prob",
        "folds": fold_details,
        # Stage 6 needs these to reconstruct model architecture
        "model_arch": cfg.model.arch,
        "model_cfg": OmegaConf.to_container(cfg.model, resolve=True),
        "in_dim": cfg.encoder.feature_dim,
    }
    _write_info(dest_dir, info)
    logger.info(f"ensemble: {copied}/{n_folds} fold checkpoints → {dest_dir}")
    return info


# ═════════════════════════════════════════════════════════════════════════════
# 3. Refit on full training data
# ═════════════════════════════════════════════════════════════════════════════


def _run_refit(
    cfg: DictConfig,
    final_dir: Path,
    all_fold_metrics: list[dict],
) -> dict:
    """
    Train a fresh model on ALL training data for a fixed number of epochs.

    Key design choices:
      - NO validation set → no early stopping, no model selection
      - Epoch count derived from CV fold stopping points (p75 by default)
      - plateau scheduler falls back to cosine (plateau needs val loss)
      - Final model = last epoch state (not "best" — there's nothing to select on)
    """
    from oceanpath.data.datamodule import MILDataModule
    from oceanpath.modules.train_module import MILTrainModule

    t = cfg.training
    start_time = time.monotonic()

    # ── Compute refit epochs ─────────────────────────────────────────────
    refit_epoch_rule = OmegaConf.select(
        cfg, "training.refit_epoch_rule", default="p75",
    )
    refit_epochs = _compute_refit_epochs(
        all_fold_metrics, rule=refit_epoch_rule, fallback_epochs=t.max_epochs,
    )

    # ── Build DataModule in refit mode ───────────────────────────────────
    splits_dir = str(
        Path(cfg.platform.splits_root) / cfg.data.name / cfg.splits.name
    )

    datamodule = MILDataModule(
        mmap_dir=cfg.data.mmap_dir,
        splits_dir=splits_dir,
        csv_path=cfg.data.csv_path,
        label_column=cfg.data.label_columns[0],
        filename_column=cfg.data.filename_column,
        scheme=cfg.splits.scheme,
        fold=0,                    # ignored in refit_mode
        batch_size=t.batch_size,
        max_instances=t.max_instances,
        dataset_max_instances=OmegaConf.select(
            cfg, "training.dataset_max_instances", default=None,
        ),
        num_workers=cfg.platform.num_workers,
        class_weighted_sampling=t.class_weighted_sampling,
        instance_dropout=t.instance_dropout,
        feature_noise_std=t.feature_noise_std,
        cache_size_mb=0,           # never cache training data
        return_coords=t.return_coords,
        verify_splits=False,       # already verified during CV
        use_preallocated_collator=t.use_preallocated_collator,
        refit_mode=True,           # <-- ALL slides → train, no val
    )
    datamodule.setup(stage="fit")
    logger.info(
        f"Refit dataset: {len(datamodule.train_dataset)} slides "
        f"({datamodule.train_dataset.get_label_counts()})"
    )

    # ── Build fresh model ────────────────────────────────────────────────
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    num_classes = (
        len(cfg.data.label_columns)
        if len(cfg.data.label_columns) > 1
        else datamodule.num_classes
    )

    # plateau needs val loss → fall back to cosine
    refit_scheduler = t.lr_scheduler
    if refit_scheduler == "plateau":
        refit_scheduler = "cosine"
        logger.info("Refit: plateau scheduler → cosine (no validation set)")

    module = MILTrainModule(
        arch=cfg.model.arch,
        in_dim=cfg.encoder.feature_dim,
        num_classes=num_classes,
        model_cfg=model_cfg,
        lr=t.lr,
        weight_decay=t.weight_decay,
        lr_scheduler=refit_scheduler,
        warmup_epochs=t.warmup_epochs,
        max_epochs=refit_epochs,      # <-- NOT cfg.training.max_epochs
        loss_type=t.loss_type,
        class_weights=OmegaConf.select(cfg, "training.class_weights", default=None),
        focal_gamma=t.focal_gamma,
        monitor_metric=t.monitor_metric,
        monitor_mode=t.monitor_mode,
        canary_interval=t.canary_interval,
        compile_model=t.compile_model,
        freeze_aggregator=t.freeze_aggregator,
        collect_embeddings=False,     # nothing to collect
    )

    L.seed_everything(t.seed, workers=True)

    # ── Trainer — NO early stopping, NO val monitoring ───────────────────
    refit_dir = final_dir / "refit"
    refit_dir.mkdir(parents=True, exist_ok=True)

    callbacks = []
    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass

    trainer = L.Trainer(
        max_epochs=refit_epochs,
        accelerator=cfg.platform.accelerator,
        devices=cfg.platform.devices,
        strategy=cfg.platform.strategy,
        precision=cfg.platform.precision,
        callbacks=callbacks,
        logger=False,                 # no W&B for refit
        gradient_clip_val=t.gradient_clip_val,
        accumulate_grad_batches=t.accumulate_grad_batches,
        deterministic=t.deterministic,
        default_root_dir=str(refit_dir),
        enable_checkpointing=False,   # we save manually at the end
        log_every_n_steps=1,
    )

    # ── Train ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(
        f"  REFIT — {len(datamodule.train_dataset)} slides, "
        f"{refit_epochs} epochs (rule={refit_epoch_rule})"
    )
    logger.info("=" * 60)

    trainer.fit(module, train_dataloaders=datamodule.train_dataloader())

    # ── Save final-epoch checkpoint ──────────────────────────────────────
    final_ckpt = refit_dir / "model.ckpt"
    trainer.save_checkpoint(str(final_ckpt))

    elapsed = time.monotonic() - start_time

    info = {
        "strategy": "refit",
        "refit_epochs": refit_epochs,
        "refit_epoch_rule": refit_epoch_rule,
        "fold_best_epochs": [fm.get("best_epoch") for fm in all_fold_metrics],
        "n_train_slides": len(datamodule.train_dataset),
        "label_counts": datamodule.train_dataset.get_label_counts(),
        "lr_scheduler": refit_scheduler,
        "elapsed_seconds": round(elapsed, 1),
        "model_path": str(final_ckpt),
    }
    _write_info(refit_dir, info)
    logger.info(f"refit: {refit_epochs} epochs, {elapsed:.0f}s → {final_ckpt}")

    # Cleanup
    del module, trainer, datamodule
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return info


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _compute_refit_epochs(
    all_fold_metrics: list[dict],
    rule: str = "p75",
    fallback_epochs: int = 30,
) -> int:
    """
    Determine how many epochs to train the refit model.

    Uses the best_epoch from each fold (epoch at which the best checkpoint
    was saved — NOT the stopped_epoch).

    Rules:
      p75    — 75th percentile of fold best_epochs (default, matches original)
      median — median
      max    — maximum across folds
      mean   — mean (rounded up)
    """
    epochs = [
        fm["best_epoch"]
        for fm in all_fold_metrics
        if fm.get("best_epoch") is not None
        and isinstance(fm["best_epoch"], (int, float))
        and fm["best_epoch"] > 0
    ]
    if not epochs:
        logger.warning(
            f"No valid best_epoch in fold metrics — "
            f"falling back to {fallback_epochs} epochs"
        )
        return fallback_epochs

    rules = {
        "p75": lambda e: int(np.ceil(np.percentile(e, 75))),
        "median": lambda e: int(np.ceil(np.median(e))),
        "max": lambda e: max(e),
        "mean": lambda e: int(np.ceil(np.mean(e))),
    }
    if rule not in rules:
        raise ValueError(f"Unknown refit_epoch_rule '{rule}'. Choose from {list(rules)}")

    n = max(1, rules[rule](epochs))
    logger.info(f"Refit epochs: {n} (rule={rule}, fold best_epochs={epochs})")
    return n


def _get_all_train_slide_ids(cfg: DictConfig) -> list[str]:
    """
    Collect ALL non-test slide IDs from the splits (union across all folds).

    For kfold:         all slides
    For holdout:       train + val (excludes test)
    For custom_kfold:  fold >= 0 (fold == -1 is test)
    For monte_carlo:   union of train+val from repeat 0
    For nested_cv:     inner_fold >= 0 from outer_fold 0
    """
    from oceanpath.data.splits import load_splits

    splits_dir = Path(cfg.platform.splits_root) / cfg.data.name / cfg.splits.name
    splits_df = load_splits(str(splits_dir), verify=False)
    scheme = cfg.splits.scheme

    if scheme in ("kfold",):
        return splits_df["slide_id"].tolist()

    elif scheme in ("holdout", "custom_holdout"):
        return splits_df.loc[
            splits_df["split"].isin(["train", "val"]), "slide_id"
        ].tolist()

    elif scheme in ("custom_kfold",):
        return splits_df.loc[splits_df["fold"] >= 0, "slide_id"].tolist()

    elif scheme == "monte_carlo":
        r0 = splits_df[splits_df["repeat"] == 0]
        return r0.loc[r0["split"].isin(["train", "val"]), "slide_id"].tolist()

    elif scheme == "nested_cv":
        outer = splits_df[splits_df["outer_fold"] == 0]
        return outer.loc[outer["inner_fold"] >= 0, "slide_id"].tolist()

    else:
        logger.warning(f"Unknown scheme '{scheme}' — using all slides for refit")
        return splits_df["slide_id"].tolist()


def parse_best_epoch_from_checkpoint(ckpt_path: str) -> Optional[int]:
    """
    Extract epoch number from a Lightning checkpoint filename.

    Handles:  best-epoch=5-val/loss=0.1234.ckpt  →  5
              epoch=12-step=500.ckpt              →  12
    """
    if not ckpt_path:
        return None
    match = re.search(r"epoch=(\d+)", Path(ckpt_path).stem)
    return int(match.group(1)) + 1 if match else None


def _write_info(dest_dir: Path, info: dict) -> None:
    """Write info.json atomically."""
    path = dest_dir / "info.json"
    path.write_text(json.dumps(info, indent=2, default=str))


def _print_summary(results: dict) -> None:
    """Print a human-readable summary of finalization results."""
    print(f"\n{'=' * 60}")
    print("  Finalization Summary")
    print(f"{'=' * 60}")

    for name in ("best_fold", "ensemble", "refit"):
        info = results.get(name, {})
        if "error" in info:
            print(f"  {name:12s}: FAILED — {info['error']}")
        elif name == "best_fold":
            print(
                f"  {name:12s}: fold {info.get('source_fold')} "
                f"({info.get('monitor_metric')}={info.get('best_score', 0):.4f}, "
                f"epoch={info.get('best_epoch')})"
            )
        elif name == "ensemble":
            print(
                f"  {name:12s}: {info.get('n_checkpoints')}/{info.get('n_folds')} folds"
            )
        elif name == "refit":
            print(
                f"  {name:12s}: {info.get('refit_epochs')} epochs, "
                f"{info.get('n_train_slides')} slides, "
                f"{info.get('elapsed_seconds', 0):.0f}s"
            )

    print(f"{'=' * 60}\n")