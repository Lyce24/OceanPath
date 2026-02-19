"""
Inference engine for Stage 6 evaluation.

Loads final model artifacts from Stage 5 (best_fold, ensemble, refit)
and runs prediction on a held-out test set.

Each strategy has its own loading/inference logic:
  best_fold / refit → single checkpoint → MILTrainModule.load_from_checkpoint
  ensemble         → K checkpoints → average softmax probabilities
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

import lightning as L
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def run_inference(
    model_dir: Path,
    mmap_dir: str,
    splits_dir: str,
    csv_path: str,
    label_column: str,
    filename_column: str,
    scheme: str,
    test_slide_ids: Optional[list] = None,
    batch_size: int = 1,
    max_instances: Optional[int] = 8000,
    num_workers: int = 0,
    accelerator: str = "auto",
    devices: int = 1,
    precision: str = "16-mixed",
) -> pd.DataFrame:
    """
    Run inference for a final model (best_fold, ensemble, or refit).

    Parameters
    ----------
    model_dir : Path to final model directory (e.g. final/best_fold/)
                Must contain info.json and model checkpoint(s).
    mmap_dir : path to memmap features directory
    splits_dir : path to splits directory
    csv_path : path to CSV with labels
    label_column : column name for labels
    filename_column : column name for slide filenames
    scheme : split scheme name
    test_slide_ids : explicit list of test slide IDs (overrides splits)
    batch_size : inference batch size
    max_instances : collator cap
    num_workers : dataloader workers
    accelerator / devices / precision : Lightning trainer args

    Returns
    -------
    DataFrame with columns: slide_id, label, prob_0, ..., prob_C
    """
    info_path = model_dir / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"No info.json in {model_dir}")

    info = json.loads(info_path.read_text())
    strategy = info["strategy"]

    logger.info(f"Running inference: strategy={strategy}, model_dir={model_dir}")

    if strategy == "ensemble":
        return _ensemble_inference(
            info, model_dir,
            mmap_dir=mmap_dir, splits_dir=splits_dir, csv_path=csv_path,
            label_column=label_column, filename_column=filename_column,
            scheme=scheme, test_slide_ids=test_slide_ids,
            batch_size=batch_size, max_instances=max_instances,
            num_workers=num_workers, accelerator=accelerator,
            devices=devices, precision=precision,
        )
    else:
        # best_fold or refit — single checkpoint
        ckpt_path = model_dir / "model.ckpt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        return _single_model_inference(
            str(ckpt_path),
            mmap_dir=mmap_dir, splits_dir=splits_dir, csv_path=csv_path,
            label_column=label_column, filename_column=filename_column,
            scheme=scheme, test_slide_ids=test_slide_ids,
            batch_size=batch_size, max_instances=max_instances,
            num_workers=num_workers, accelerator=accelerator,
            devices=devices, precision=precision,
        )


def _build_test_datamodule(
    mmap_dir: str,
    splits_dir: str,
    csv_path: str,
    label_column: str,
    filename_column: str,
    scheme: str,
    test_slide_ids: Optional[list],
    batch_size: int,
    max_instances: Optional[int],
    num_workers: int,
):
    """Build a DataModule configured for test-only inference."""
    from oceanpath.data.datamodule import MILDataModule

    dm = MILDataModule(
        mmap_dir=mmap_dir,
        splits_dir=splits_dir,
        csv_path=csv_path,
        label_column=label_column,
        filename_column=filename_column,
        scheme=scheme,
        fold=0,
        batch_size=batch_size,
        max_instances=max_instances,
        num_workers=num_workers,
        class_weighted_sampling=False,
        instance_dropout=0.0,
        feature_noise_std=0.0,
        cache_size_mb=0,
        return_coords=False,
        verify_splits=False,
        use_preallocated_collator=False,  # dynamic for test
    )

    # If explicit test IDs provided, override
    if test_slide_ids is not None:
        dm._override_test_ids = test_slide_ids

    dm.setup(stage="test")
    return dm


def _single_model_inference(
    ckpt_path: str,
    mmap_dir: str,
    splits_dir: str,
    csv_path: str,
    label_column: str,
    filename_column: str,
    scheme: str,
    test_slide_ids: Optional[list],
    batch_size: int,
    max_instances: Optional[int],
    num_workers: int,
    accelerator: str,
    devices: int,
    precision: str,
) -> pd.DataFrame:
    """Run inference with a single checkpoint."""
    from oceanpath.modules.train_module import MILTrainModule

    # Load model
    try:
        module = MILTrainModule.load_from_checkpoint(
            ckpt_path, weights_only=False,
        )
    except TypeError:
        module = MILTrainModule.load_from_checkpoint(ckpt_path)
    module.eval()

    # Build test data
    dm = _build_test_datamodule(
        mmap_dir=mmap_dir, splits_dir=splits_dir, csv_path=csv_path,
        label_column=label_column, filename_column=filename_column,
        scheme=scheme, test_slide_ids=test_slide_ids,
        batch_size=batch_size, max_instances=max_instances,
        num_workers=num_workers,
    )

    if dm.test_dataset is None or len(dm.test_dataset) == 0:
        logger.warning("No test data available")
        return pd.DataFrame()

    # Run test
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.test(module, datamodule=dm)

    # Extract predictions
    preds_df = _predictions_to_df(module._test_predictions)
    logger.info(f"Inference complete: {len(preds_df)} slides")
    return preds_df


def _ensemble_inference(
    info: dict,
    model_dir: Path,
    mmap_dir: str,
    splits_dir: str,
    csv_path: str,
    label_column: str,
    filename_column: str,
    scheme: str,
    test_slide_ids: Optional[list],
    batch_size: int,
    max_instances: Optional[int],
    num_workers: int,
    accelerator: str,
    devices: int,
    precision: str,
) -> pd.DataFrame:
    """
    Run ensemble inference: load each fold model, predict, average softmax.
    """
    from oceanpath.modules.train_module import MILTrainModule

    fold_details = info.get("folds", [])
    valid_folds = [f for f in fold_details if "error" not in f]

    if not valid_folds:
        raise ValueError("No valid fold checkpoints in ensemble")

    # Build test data once
    dm = _build_test_datamodule(
        mmap_dir=mmap_dir, splits_dir=splits_dir, csv_path=csv_path,
        label_column=label_column, filename_column=filename_column,
        scheme=scheme, test_slide_ids=test_slide_ids,
        batch_size=batch_size, max_instances=max_instances,
        num_workers=num_workers,
    )

    if dm.test_dataset is None or len(dm.test_dataset) == 0:
        logger.warning("No test data available")
        return pd.DataFrame()

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
    )

    all_fold_preds = []

    for fold_info in valid_folds:
        ckpt_path = model_dir / f"fold_{fold_info['fold']}.ckpt"
        if not ckpt_path.is_file():
            logger.warning(f"Fold {fold_info['fold']} checkpoint not found — skipping")
            continue

        try:
            module = MILTrainModule.load_from_checkpoint(
                str(ckpt_path), weights_only=False,
            )
        except TypeError:
            module = MILTrainModule.load_from_checkpoint(str(ckpt_path))
        module.eval()

        # Reset predictions accumulator
        module._test_predictions = []

        trainer.test(module, datamodule=dm)
        fold_df = _predictions_to_df(module._test_predictions)
        fold_df = fold_df.set_index("slide_id")
        all_fold_preds.append(fold_df)

        del module

    if not all_fold_preds:
        raise ValueError("No successful fold predictions in ensemble")

    # Average probabilities across folds
    prob_cols = [c for c in all_fold_preds[0].columns if c.startswith("prob_")]

    # Stack and average
    combined = all_fold_preds[0][["label"]].copy()
    for col in prob_cols:
        values = np.stack([fp[col].values for fp in all_fold_preds], axis=0)
        combined[col] = values.mean(axis=0)

    combined = combined.reset_index()
    logger.info(
        f"Ensemble inference: {len(combined)} slides, "
        f"{len(all_fold_preds)} fold models averaged"
    )
    return combined


def _predictions_to_df(predictions: list) -> pd.DataFrame:
    """Convert accumulated predictions list to DataFrame."""
    if not predictions:
        return pd.DataFrame()
    return pd.DataFrame(predictions)


def get_test_slide_ids(
    splits_dir: str,
    scheme: str,
    fold: int = 0,
) -> list:
    """Extract test slide IDs from splits."""
    from oceanpath.data.splits import load_splits, get_slide_ids_for_fold

    splits_df = load_splits(splits_dir, verify=False)
    fold_ids = get_slide_ids_for_fold(splits_df, fold, scheme=scheme)
    return fold_ids.get("test", [])