from __future__ import annotations

import copy
import os
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WANDB_AVAILABLE = False

from modules.cv.exp_config import ExperimentConfig
from modules.cv.wsi_module import WSIClassificationModule
from modules.cv.utils.callbacks import BestStateTracker, FoldMetricLogger
from modules.cv.utils.eval_utils import (
    logits_to_probs,
    compute_metrics,
    bootstrap_ci,
    operating_point_metrics,
    collect_predictions,
)
from modules.cv.utils.train_utils import (
    set_seed, 
    cleanup_memory, 
    load_data, 
    FullTrainDataModule, 
    TestOnlyDataModule
)

from data.data_module import WSIMILDataModule
from models.MIL.wsi_model import WSIModel

# =============================================================================
# Main Experiment Class
# =============================================================================
class KFoldExperimentLIT:
    """
    K-Fold Cross-Validation Experiment with proper train/test separation.
    
    Protocol:
    1. Hold out test set (k_fold == -1) - NEVER touched during training
    2. K-Fold CV on training data (k_fold >= 0):
       - For each fold k: train on folds != k, early stop on fold k
       - Save best checkpoint for each fold
    3. Evaluate all K models on held-out test set
    4. Report per-model metrics AND ensemble metrics
    5. Optionally refit on full training data OR use ensemble for final model
    
    Attributes:
        config: Experiment configuration
        device: Torch device for inference
        wandb_run: Optional W&B run for logging
    """

    def __init__(
        self,
        config: ExperimentConfig,
        device: Optional[torch.device] = None,
        wandb_run: Optional[Any] = None,
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wandb_run = wandb_run
        
        # Internal state
        self._fold_results: List[Dict[str, Any]] = []
        self._refit_model: Optional[WSIClassificationModule] = None
        self._refit_epoch: Optional[int] = None
        self._n_classes: Optional[int] = None
        self._feature_dim: Optional[int] = None
        
        mode_str = "FINE-TUNE" if config.mode == "ft" else "LINEAR PROBE"
        print(f"\n[KFoldExperimentLIT] Mode: {mode_str}")
        print(f"[KFoldExperimentLIT] LR encoder/head: {config.encoder_lr} / {config.head_lr}")
        print(f"[KFoldExperimentLIT] WD encoder/head: {config.encoder_wd} / {config.head_wd}")
        print(f"[KFoldExperimentLIT] Scheduler: {config.scheduler}")

    # =========================================================================
    # Logging Utilities
    # =========================================================================
    def _log_summary(self, key: str, value: Any) -> None:
        """Log a value to wandb summary."""
        if self.wandb_run is not None:
            self.wandb_run.summary[key] = value

    def _log_table(self, name: str, columns: List[str], data: List[List[Any]]) -> None:
        """Log a table to wandb."""
        if self.wandb_run is not None and WANDB_AVAILABLE:
            self.wandb_run.log({name: wandb.Table(columns=columns, data=data)})

    def _log_metric(self, key: str, value: Any, step: Optional[int] = None) -> None:
        """Log a metric to wandb."""
        if self.wandb_run is not None:
            if step is not None:
                self.wandb_run.log({key: value}, step=step)
            else:
                self.wandb_run.log({key: value})

    def _reset_state(self) -> None:
        """Reset internal state."""
        self._fold_results = []
        self._refit_model = None
        self._refit_epoch = None
        self._n_classes = None
        self._feature_dim = None
        cleanup_memory()

    # -------------------------------------------------------------------------
    # Model & Trainer Creation
    # -------------------------------------------------------------------------
    def _create_model(
        self,
        n_classes: int,
        feature_dim: int,
        class_weights: Optional[torch.Tensor] = None,
        max_epochs: Optional[int] = None,
    ) -> WSIClassificationModule:
        return WSIClassificationModule(
            mil=self.config.mil,
            n_classes=n_classes,
            feature_dim=feature_dim,
            mil_attrs=self.config.mil_attrs,
            mode=self.config.mode,
            encoder_weights_path=self.config.encoder_weights_path,
            encoder_lr=float(self.config.encoder_lr),
            head_lr=float(self.config.head_lr),
            encoder_wd=float(self.config.encoder_wd),
            head_wd=float(self.config.head_wd),
            scheduler=self.config.scheduler,
            min_lr=self.config.min_lr,
            warmup_epochs=self.config.warmup_epochs,
            max_epochs=int(max_epochs or self.config.max_epochs),
            class_weights=class_weights if self.config.use_class_weights else None,
            label_smoothing=self.config.label_smoothing,
        )

    def _create_trainer(
        self,
        max_epochs: int,
        callbacks: List[Callback],
        fold: Optional[int] = None,
    ) -> pl.Trainer:
        if self.wandb_run is not None and fold is not None:
            callbacks = callbacks + [
                FoldMetricLogger(fold=fold, wandb_run=self.wandb_run, log_lr=True)
            ]

        return pl.Trainer(
            max_epochs=max_epochs,
            precision=self.config.precision,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            gradient_clip_val=self.config.gradient_clip_val,
            callbacks=callbacks,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            deterministic=False,
            check_val_every_n_epoch=self.config.check_val_every_n_epoch,
            log_every_n_steps=self.config.log_every_n_steps,
        )

    # -------------------------------------------------------------------------
    # Fold Training (NO evaluation on validation - just early stopping)
    # -------------------------------------------------------------------------
    def _train_fold(self, fold: int) -> Dict[str, Any]:
        """
        Train a single fold.
        
        IMPORTANT: We do NOT evaluate metrics on the validation fold here.
        The validation fold is ONLY used for early stopping.
        All metrics are computed on the held-out test set later.
        
        Returns:
            Dictionary containing:
            - fold: Fold index
            - state_dict: Best model weights
            - best_epoch: Epoch where best model was saved
            - best_val_score: Best validation score (for early stopping, not reporting)
            - n_classes: Number of classes
            - feature_dim: Feature dimension
        """
        print(f"\n{'='*20} FOLD {fold} {'='*20}")
        set_seed(self.config.seed + fold)

        # Create data module for this fold
        dm = WSIMILDataModule(
            csv_path=self.config.csv_path,
            feature_dir=self.config.feature_dir,
            id_col=self.config.id_col,
            target_col=self.config.target_col,
            current_fold=fold,
            batch_size=self.config.batch_size,
            bag_size=self.config.bag_size,
            replacement=self.config.replacement,
            num_workers=self.config.num_workers,
            precision=16,
            return_key=False,
            verbose=False,
        )
        dm.setup("fit")

        # Store dimensions for later use
        self._n_classes = dm.num_classes
        self._feature_dim = dm.dim_features

        # Create model
        model = self._create_model(
            n_classes=dm.num_classes,
            feature_dim=dm.dim_features,
            class_weights=dm.class_weight_tensor,
        )

        # Setup callbacks - early stopping monitors validation metric
        monitor = "val/loss"  # Use val loss for early stopping
        
        tracker = BestStateTracker(
            monitor=monitor,
            mode="min",
            min_delta=self.config.min_delta,
        )
        
        early_stop = EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=self.config.early_stopping_patience,
            min_delta=self.config.min_delta,
            verbose=False,
        )

        # Train
        trainer = self._create_trainer(
            max_epochs=self.config.max_epochs,
            callbacks=[tracker, early_stop],
            fold=fold,
        )
        trainer.fit(model, datamodule=dm)

        # Get best state
        if tracker.best_state is None:
            warnings.warn(f"Fold {fold}: No best state tracked, using final state")
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = trainer.current_epoch
            best_score = None
        else:
            best_state = copy.deepcopy(tracker.best_state)
            best_epoch = tracker.best_epoch
            best_score = tracker.best_score

        # Log fold training summary
        epoch_str = f"{best_epoch + 1}" if best_epoch is not None else "N/A"
        score_str = f"{best_score:.4f}" if best_score is not None else "N/A"
        print(f"[Fold {fold}] Training complete")
        print(f"[Fold {fold}] Best epoch: {epoch_str}, Val score (for early stop): {score_str}")
        print(f"[Fold {fold}] Train samples: {len(dm.train_ids)}, Val samples: {len(dm.val_ids)}")

        result = {
            "fold": fold,
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_score": best_score,  # For reference only, not for reporting
            "n_classes": dm.num_classes,
            "feature_dim": dm.dim_features,
            "n_train_samples": len(dm.train_ids),
            "n_val_samples": len(dm.val_ids),
        }
        
        # Log to wandb
        self._log_summary(f"fold_{fold}/best_epoch", best_epoch + 1 if best_epoch is not None else None)
        self._log_summary(f"fold_{fold}/best_val_score", best_score)

        # Cleanup
        del model, trainer, dm
        cleanup_memory()
        
        return result

    # -------------------------------------------------------------------------
    # Run K-Fold CV (training only)
    # -------------------------------------------------------------------------
    def run_cv(self) -> Dict[str, Any]:
        """
        Run K-Fold cross-validation (training only).
        
        This method:
        1. Trains K models, one for each fold
        2. Uses validation fold ONLY for early stopping
        3. Saves best checkpoint for each fold
        4. Does NOT compute any metrics (that's done in test_cv)
        
        Returns:
            Dictionary containing:
            - fold_results: List of per-fold results
            - n_folds: Number of folds
            - best_epochs: List of best epochs per fold
        """
        df = load_data(self.config.csv_path)
        
        if "k_fold" not in df.columns:
            raise ValueError("CSV must have 'k_fold' column")

        # Get fold indices (exclude test set which has k_fold == -1)
        folds = sorted([int(f) for f in df["k_fold"].unique() if int(f) >= 0])
        
        if not folds:
            raise ValueError("No valid folds found (k_fold >= 0)")

        print(f"\n{'='*60}")
        print(f"K-FOLD CROSS-VALIDATION TRAINING")
        print(f"{'='*60}")
        print(f"Number of folds: {len(folds)}")
        print(f"Mode: {self.config.mode.upper()}")
        print(f"Effective batch size: {self.config.effective_batch_size}")
        print(f"Max epochs: {self.config.max_epochs}")
        print(f"Early stopping patience: {self.config.early_stopping_patience}")
        print(f"{'='*60}")

        if self.wandb_run is not None:
            self.wandb_run.config.update(self.config.to_dict())

        # Train each fold
        self._fold_results = []
        for k in folds:
            result = self._train_fold(k)
            self._fold_results.append(result)

        if not self._fold_results:
            raise RuntimeError("All folds failed to train")

        # Summary
        best_epochs = [
            r["best_epoch"] + 1 
            for r in self._fold_results 
            if r.get("best_epoch") is not None
        ]
        
        best_val_scores = [
            r["best_val_score"] 
            for r in self._fold_results 
            if r.get("best_val_score") is not None
        ]
        
        print(f"\n{'='*60}")
        print(f"CV TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Successfully trained {len(self._fold_results)} folds")
        print(f"Best epochs per fold: {best_epochs}")
        print(f"Median best epoch: {int(np.median(best_epochs)) if best_epochs else 'N/A'}")
        print(f"Mean best val score (for early stopping): {np.mean(best_val_scores) if best_val_scores else 'N/A'}")

        # Log summary
        self._log_summary("cv/median_best_epoch", int(np.median(best_epochs)) if best_epochs else None)
        self._log_summary("cv/mean_best_val_score", np.mean(best_val_scores) if best_val_scores else None)

        return {
            "fold_results": self._fold_results,
            "n_folds": len(self._fold_results),
            "best_epochs": best_epochs,
            "best_val_scores": best_val_scores,
        }
        
    # -------------------------------------------------------------------------
    # Build Models from Fold Results
    # -------------------------------------------------------------------------
    def _load_fold_model(self, fold_result: Dict[str, Any]) -> WSIModel:
        """Load a single model from fold results."""
        module = self._create_model(
            n_classes=fold_result["n_classes"],
            feature_dim=fold_result["feature_dim"],
            class_weights=None,
        )
        module.load_state_dict(fold_result["state_dict"])
        return module.model  # Return the underlying WSIModel
    
    
    # -------------------------------------------------------------------------
    # Refit on Full Training Data
    # -------------------------------------------------------------------------
    def _train_refit_model(self, df_train: pd.DataFrame) -> WSIClassificationModule:
        """
        Train final model on ALL training data.
        
        Uses median(best_epoch) from CV folds as target stopping point.
        Maintains same LR schedule as CV for consistency.
        
        Args:
            df_train: Full training DataFrame
            
        Returns:
            Trained WSIClassificationModule
        """
        print(f"\n{'='*60}")
        print(f"FINAL MODEL TRAINING (REFIT)")
        print(f"{'='*60}")
        
        set_seed(self.config.seed)

        # Calculate target epoch from CV results
        best_epochs = [
            r["best_epoch"] + 1
            for r in self._fold_results
            if r.get("best_epoch") is not None
        ]

        if not best_epochs:
            raise ValueError("No valid best_epochs found in CV results")

        target_epoch = int(np.ceil(np.median(best_epochs)))

        print(f"CV best epochs: {best_epochs}")
        print(f"Target epoch (median): {target_epoch}")
        print(f"Training samples: {len(df_train)}")

        # Create data module
        dm = FullTrainDataModule(
            combined_data=df_train,
            feature_dir=self.config.feature_dir,
            id_col=self.config.id_col,
            target_col=self.config.target_col,
            batch_size=self.config.batch_size,
            bag_size=self.config.bag_size,
            replacement=self.config.replacement,
            num_workers=self.config.num_workers,
            precision=16,
            current_fold=0,
            return_key=False,
            verbose=False,
        )
        dm.setup("fit")

        # Create model with same max_epochs as CV for consistent LR schedule
        model = self._create_model(
            n_classes=dm.num_classes,
            feature_dim=dm.dim_features,
            class_weights=dm.class_weight_tensor,
            max_epochs=self.config.max_epochs,
        )

        # Train
        trainer = pl.Trainer(
            max_epochs=target_epoch,
            min_epochs=target_epoch,
            precision=self.config.precision,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            gradient_clip_val=self.config.gradient_clip_val,
            callbacks=[],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            deterministic=False,
        )
        trainer.fit(model, datamodule=dm)

        print(f"[Refit] Training complete: {target_epoch} epochs")

        self._log_summary("refit/target_epoch", target_epoch)
        self._log_summary("refit/n_train_samples", len(df_train))

        del dm, trainer
        cleanup_memory()

        return model, target_epoch
    
    # =========================================================================
    # Test Evaluation (Main Method)
    # =========================================================================
    def test_cv(
        self,
        df_test: pd.DataFrame,
        df_train: Optional[pd.DataFrame] = None,
        threshold: float = 0.5,
        compare_refit: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate all K fold models AND refit model on the held-out test set.
        
        This is the main evaluation method that:
        1. Evaluates each fold model individually (per-model metrics)
        2. Computes ensemble predictions (mean_prob strategy)
        3. Optionally trains and evaluates a refit model
        4. Compares all strategies and determines the best one
        5. Logs comprehensive results to wandb
        
        Args:
            df_test: Test set DataFrame (must have id_col and target_col)
            df_train: Training DataFrame for refit (required if compare_refit=True)
            threshold: Decision threshold for binary classification
            compare_refit: Whether to train and evaluate a refit model
            
        Returns:
            Dictionary containing:
            - per_model: Per-fold model results
            - ensemble: Ensemble results (mean_prob)
            - refit: Refit model results (if compare_refit=True)
            - comparison: Strategy comparison summary
            - best_strategy: Name of the best strategy
            - y_true: Ground truth labels
        """
        if not self._fold_results:
            raise RuntimeError("No fold results available. Run run_cv() first.")

        if len(df_test) == 0:
            warnings.warn("Empty test set")
            return self._empty_test_results()

        if compare_refit and df_train is None:
            raise ValueError("df_train is required when compare_refit=True")

        # =====================================================================
        # Setup
        # =====================================================================
        n_classes = self._fold_results[0]["n_classes"]
        is_binary = (n_classes == 2)
        th = threshold if is_binary else None
        
        print(f"\n{'='*70}")
        print(f"TEST SET EVALUATION")
        print(f"{'='*70}")
        print(f"Test samples: {len(df_test)}")
        print(f"Fold models: {len(self._fold_results)}")
        print(f"Binary classification: {is_binary}")
        print(f"Threshold: {threshold:.4f}" if is_binary else "Threshold: N/A (multiclass)")
        print(f"Compare refit: {compare_refit}")

        # Create test data loader
        dm = TestOnlyDataModule(
            combined_data=df_test,
            feature_dir=self.config.feature_dir,
            id_col=self.config.id_col,
            target_col=self.config.target_col,
            batch_size=1,
            num_workers=self.config.num_workers,
            precision=16,
        )
        dm.setup("test")
        test_loader = dm.test_dataloader()

        # =====================================================================
        # 1. Collect predictions from all fold models
        # =====================================================================
        print(f"\n[1/4] Collecting predictions from {len(self._fold_results)} fold models...")
        
        all_logits, all_probs, y_true = self._collect_fold_predictions(test_loader)
        
        n_samples = len(y_true)
        print(f"      Collected: {len(all_probs)} folds × {n_samples} samples × {n_classes} classes")

        # Stack for ensemble
        stacked_logits = np.stack(all_logits, axis=0)  # [K, N, C]
        stacked_probs = np.stack(all_probs, axis=0)    # [K, N, C]

        # =====================================================================
        # 2. Compute per-model metrics
        # =====================================================================
        print(f"\n[2/4] Computing per-model metrics...")
        
        per_model_results = self._evaluate_per_model(y_true, all_probs, th)

        # =====================================================================
        # 3. Compute ensemble metrics
        # =====================================================================
        print(f"\n[3/4] Computing ensemble metrics...")
        
        ensemble_results = self._evaluate_ensemble(
            y_true=y_true,
            stacked_probs=stacked_probs,
            stacked_logits=stacked_logits,
            threshold=th,
        )

        # =====================================================================
        # 4. Train and evaluate refit model (optional)
        # =====================================================================
        refit_results = None
        if compare_refit:
            print(f"\n[4/4] Training and evaluating refit model...")
            
            # Train refit model
            refit_model, refit_epoch = self._train_refit_model(df_train)
            self._refit_model = refit_model
            
            # Evaluate refit model
            refit_results = self._evaluate_refit(
                model=refit_model,
                test_loader=test_loader,
                y_true=y_true,
                threshold=th,
                refit_epoch=refit_epoch,
            )
        else:
            print(f"\n[4/4] Skipping refit model (compare_refit=False)")

        # =====================================================================
        # 5. Compare strategies and determine best
        # =====================================================================
        comparison, best_strategy = self._compare_strategies(
            per_model_results=per_model_results,
            ensemble_results=ensemble_results,
            refit_results=refit_results,
            is_binary=is_binary,
        )

        # =====================================================================
        # 6. Log tables and final summary
        # =====================================================================
        self._log_all_results(
            per_model_results=per_model_results,
            ensemble_results=ensemble_results,
            refit_results=refit_results,
            comparison=comparison,
            best_strategy=best_strategy,
            is_binary=is_binary,
        )

        # Cleanup
        del dm
        cleanup_memory()

        return {
            "per_model": per_model_results,
            "ensemble": ensemble_results,
            "refit": refit_results,
            "comparison": comparison,
            "best_strategy": best_strategy,
            "y_true": y_true,
            "stacked_probs": stacked_probs,
            "stacked_logits": stacked_logits,
        }

    # =========================================================================
    # Helper Methods for test_cv
    # =========================================================================
    def _empty_test_results(self) -> Dict[str, Any]:
        """Return empty results for empty test set."""
        return {
            "per_model": {"metrics": [], "summary": {}},
            "ensemble": {"metrics": {}, "ci": {}, "probs": np.array([])},
            "refit": None,
            "comparison": {},
            "best_strategy": None,
            "y_true": np.array([]),
            "stacked_probs": np.array([]),
            "stacked_logits": np.array([]),
        }

    def _collect_fold_predictions(
        self,
        test_loader: torch.utils.data.DataLoader,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """Collect predictions from all fold models."""
        all_logits = []
        all_probs = []
        y_true = None

        for fold_result in self._fold_results:
            model = self._load_fold_model(fold_result)
            model.to(self.device)
            model.eval()

            logits_i, labels_i = collect_predictions(model, test_loader, self.device)
            probs_i = logits_to_probs(logits_i)

            all_logits.append(logits_i)
            all_probs.append(probs_i)

            if y_true is None:
                y_true = labels_i.numpy()

            del model
            cleanup_memory()

        return all_logits, all_probs, y_true

    def _evaluate_per_model(
        self,
        y_true: np.ndarray,
        all_probs: List[np.ndarray],
        threshold: Optional[float],
    ) -> Dict[str, Any]:
        """Evaluate each fold model individually."""
        metrics_list = []
        
        for fold_result, probs_i in zip(self._fold_results, all_probs):
            metrics_i = compute_metrics(y_true, probs_i, threshold=threshold)
            metrics_i["fold"] = fold_result["fold"]
            metrics_list.append(metrics_i)

        # Aggregate summary statistics
        metric_keys = [k for k in metrics_list[0].keys() if k != "fold"]
        summary = {}
        
        for mk in metric_keys:
            vals = [m[mk] for m in metrics_list if np.isfinite(m.get(mk, np.nan))]
            if vals:
                arr = np.array(vals)
                summary[mk] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }

        # Print summary
        print(f"\n{'─'*50}")
        print(f"PER-MODEL METRICS (mean ± std)")
        print(f"{'─'*50}")
        for mk, stats in summary.items():
            print(f"  {mk:<20}: {stats['mean']:.4f} ± {stats['std']:.4f} [{stats['min']:.4f}, {stats['max']:.4f}]")

        return {"metrics": metrics_list, "summary": summary}

    def _evaluate_ensemble(
        self,
        y_true: np.ndarray,
        stacked_probs: np.ndarray,
        stacked_logits: np.ndarray,
        threshold: Optional[float],
    ) -> Dict[str, Any]:
        """Evaluate ensemble using mean probability aggregation."""
        # Compute ensemble probabilities
        ensemble_probs = stacked_probs.mean(axis=0)  # [N, C]
        
        # Point estimates
        metrics = compute_metrics(y_true, ensemble_probs, threshold=threshold)
        
        # Operating point metrics (binary only)
        op_metrics = {}
        is_binary = (ensemble_probs.shape[1] == 2)
        if is_binary and threshold is not None:
            op_metrics = operating_point_metrics(y_true, ensemble_probs[:, 1], threshold)

        # Bootstrap CI
        ci = bootstrap_ci(
            y_true,
            ensemble_probs,
            n_bootstraps=self.config.n_bootstraps,
            ci=0.95,
            threshold=threshold,
            seed=self.config.seed,
            stratified=True,
        )

        # Print results
        print(f"\n{'─'*50}")
        print(f"ENSEMBLE (mean_prob)")
        print(f"{'─'*50}")
        print("Point estimates:")
        for k, v in metrics.items():
            print(f"  {k:<20}: {v:.4f}")
        
        if op_metrics:
            print("Operating point metrics:")
            for k, v in op_metrics.items():
                print(f"  {k:<20}: {v:.4f}")
        
        print(f"Bootstrap 95% CI ({self.config.n_bootstraps} iterations):")
        for k, v in ci.items():
            print(f"  {k:<20}: {v['mean']:.4f} [{v['lower']:.4f}, {v['upper']:.4f}]")

        return {
            "probs": ensemble_probs,
            "metrics": metrics,
            "op_metrics": op_metrics,
            "ci": ci,
        }

    def _evaluate_refit(
        self,
        model: WSIClassificationModule,
        test_loader: torch.utils.data.DataLoader,
        y_true: np.ndarray,
        threshold: Optional[float],
        refit_epoch: int,
    ) -> Dict[str, Any]:
        """Evaluate the refit model."""
        # Get underlying WSIModel
        wsi_model = model.model
        wsi_model.to(self.device)
        wsi_model.eval()

        # Collect predictions
        logits, _ = collect_predictions(wsi_model, test_loader, self.device)
        probs = logits_to_probs(logits)

        # Point estimates
        metrics = compute_metrics(y_true, probs, threshold=threshold)
        
        # Operating point metrics
        op_metrics = {}
        is_binary = (probs.shape[1] == 2)
        if is_binary and threshold is not None:
            op_metrics = operating_point_metrics(y_true, probs[:, 1], threshold)

        # Bootstrap CI
        ci = bootstrap_ci(
            y_true,
            probs,
            n_bootstraps=self.config.n_bootstraps,
            ci=0.95,
            threshold=threshold,
            seed=self.config.seed,
            stratified=True,
        )

        # Print results
        print(f"\n{'─'*50}")
        print(f"REFIT MODEL (epoch={refit_epoch})")
        print(f"{'─'*50}")
        print("Point estimates:")
        for k, v in metrics.items():
            print(f"  {k:<20}: {v:.4f}")
        
        if op_metrics:
            print("Operating point metrics:")
            for k, v in op_metrics.items():
                print(f"  {k:<20}: {v:.4f}")
        
        print(f"Bootstrap 95% CI ({self.config.n_bootstraps} iterations):")
        for k, v in ci.items():
            print(f"  {k:<20}: {v['mean']:.4f} [{v['lower']:.4f}, {v['upper']:.4f}]")

        cleanup_memory()

        return {
            "probs": probs,
            "logits": logits,
            "metrics": metrics,
            "op_metrics": op_metrics,
            "ci": ci,
            "epoch": refit_epoch,
        }

    def _compare_strategies(
        self,
        per_model_results: Dict[str, Any],
        ensemble_results: Dict[str, Any],
        refit_results: Optional[Dict[str, Any]],
        is_binary: bool,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Compare all strategies and determine the best one.
        
        Returns:
            Tuple of (comparison dict, best strategy name)
        """
        # Define primary metric for comparison
        primary_metric = "roc_auc" if is_binary else "roc_auc_macro"
        secondary_metrics = ["balanced_accuracy", "f1"] if is_binary else ["balanced_accuracy", "f1_macro"]
        
        comparison = {}
        
        # Per-model (mean across folds)
        pm_summary = per_model_results["summary"]
        if primary_metric in pm_summary:
            comparison["per_model_mean"] = {
                "auc": pm_summary[primary_metric]["mean"],
                "auc_std": pm_summary[primary_metric]["std"],
                "description": f"Mean of {len(self._fold_results)} fold models",
            }
            for sm in secondary_metrics:
                if sm in pm_summary:
                    comparison["per_model_mean"][sm] = pm_summary[sm]["mean"]

        # Ensemble
        ens_metrics = ensemble_results["metrics"]
        ens_ci = ensemble_results["ci"]
        if primary_metric in ens_metrics:
            comparison["ensemble"] = {
                "auc": ens_metrics[primary_metric],
                "auc_ci_lower": ens_ci[primary_metric]["lower"] if primary_metric in ens_ci else None,
                "auc_ci_upper": ens_ci[primary_metric]["upper"] if primary_metric in ens_ci else None,
                "description": f"Ensemble of {len(self._fold_results)} models (mean_prob)",
            }
            for sm in secondary_metrics:
                if sm in ens_metrics:
                    comparison["ensemble"][sm] = ens_metrics[sm]

        # Refit
        if refit_results is not None:
            refit_metrics = refit_results["metrics"]
            refit_ci = refit_results["ci"]
            if primary_metric in refit_metrics:
                comparison["refit"] = {
                    "auc": refit_metrics[primary_metric],
                    "auc_ci_lower": refit_ci[primary_metric]["lower"] if primary_metric in refit_ci else None,
                    "auc_ci_upper": refit_ci[primary_metric]["upper"] if primary_metric in refit_ci else None,
                    "description": f"Single model trained for {refit_results['epoch']} epochs on all data",
                }
                for sm in secondary_metrics:
                    if sm in refit_metrics:
                        comparison["refit"][sm] = refit_metrics[sm]

        # Determine best strategy by AUC
        strategies_to_compare = ["ensemble"]
        if refit_results is not None:
            strategies_to_compare.append("refit")
        
        best_strategy = max(
            strategies_to_compare,
            key=lambda s: comparison.get(s, {}).get("auc", 0)
        )
        
        best_auc = comparison[best_strategy]["auc"]

        # Print comparison
        print(f"\n{'='*70}")
        print(f"STRATEGY COMPARISON")
        print(f"{'='*70}")
        print(f"\n{'Strategy':<20} | {'AUC':<22} | {'Bal.Acc':<10} | {'F1':<10}")
        print(f"{'-'*70}")
        
        # Per-model mean
        if "per_model_mean" in comparison:
            pm = comparison["per_model_mean"]
            auc_str = f"{pm['auc']:.4f} ± {pm['auc_std']:.4f}"
            ba_str = f"{pm.get('balanced_accuracy', pm.get('bal_accuracy', 0)):.4f}"
            f1_str = f"{pm.get('f1', pm.get('f1_macro', 0)):.4f}"
            print(f"{'Per-model (mean)':<20} | {auc_str:<22} | {ba_str:<10} | {f1_str:<10}")
        
        # Ensemble
        if "ensemble" in comparison:
            ens = comparison["ensemble"]
            ci_lower = ens.get("auc_ci_lower")
            ci_upper = ens.get("auc_ci_upper")
            if ci_lower is not None and ci_upper is not None:
                auc_str = f"{ens['auc']:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]"
            else:
                auc_str = f"{ens['auc']:.4f}"
            ba_key = "balanced_accuracy" if "balanced_accuracy" in ens else "bal_accuracy"
            f1_key = "f1" if "f1" in ens else "f1_macro"
            ba_str = f"{ens.get(ba_key, 0):.4f}"
            f1_str = f"{ens.get(f1_key, 0):.4f}"
            marker = " ✓" if best_strategy == "ensemble" else ""
            print(f"{'Ensemble':<20} | {auc_str:<22} | {ba_str:<10} | {f1_str:<10}{marker}")
        
        # Refit
        if "refit" in comparison:
            ref = comparison["refit"]
            ci_lower = ref.get("auc_ci_lower")
            ci_upper = ref.get("auc_ci_upper")
            if ci_lower is not None and ci_upper is not None:
                auc_str = f"{ref['auc']:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]"
            else:
                auc_str = f"{ref['auc']:.4f}"
            ba_key = "balanced_accuracy" if "balanced_accuracy" in ref else "bal_accuracy"
            f1_key = "f1" if "f1" in ref else "f1_macro"
            ba_str = f"{ref.get(ba_key, 0):.4f}"
            f1_str = f"{ref.get(f1_key, 0):.4f}"
            marker = " ✓" if best_strategy == "refit" else ""
            print(f"{'Refit':<20} | {auc_str:<22} | {ba_str:<10} | {f1_str:<10}{marker}")

        print(f"\n{'='*70}")
        print(f"BEST STRATEGY: {best_strategy.upper()} (AUC = {best_auc:.4f})")
        print(f"{'='*70}")

        return comparison, best_strategy

    def _log_all_results(
        self,
        per_model_results: Dict[str, Any],
        ensemble_results: Dict[str, Any],
        refit_results: Optional[Dict[str, Any]],
        comparison: Dict[str, Any],
        best_strategy: str,
        is_binary: bool,
    ) -> None:
        """Log all results to wandb."""
        primary_metric = "roc_auc" if is_binary else "roc_auc_macro"
        
        # Per-model metrics
        pm_summary = per_model_results["summary"]
        for mk, stats in pm_summary.items():
            self._log_summary(f"test/per_model/{mk}_mean", stats["mean"])
            self._log_summary(f"test/per_model/{mk}_std", stats["std"])
        
        # Per-model table
        metrics_list = per_model_results["metrics"]
        if metrics_list:
            metric_keys = [k for k in metrics_list[0].keys() if k != "fold"]
            self._log_table(
                "test_per_model_metrics",
                columns=["fold"] + metric_keys,
                data=[[m["fold"]] + [m.get(k, np.nan) for k in metric_keys] for m in metrics_list],
            )

        # Ensemble metrics
        for k, v in ensemble_results["metrics"].items():
            self._log_summary(f"test/ensemble/{k}", v)
        for k, v in ensemble_results.get("op_metrics", {}).items():
            self._log_summary(f"test/ensemble/{k}", v)
        for k, v in ensemble_results["ci"].items():
            self._log_summary(f"test/ensemble/{k}_ci_lower", v["lower"])
            self._log_summary(f"test/ensemble/{k}_ci_upper", v["upper"])

        # Ensemble CI table
        ci = ensemble_results["ci"]
        ci_data = [[k, v["mean"], v["std"], v["lower"], v["upper"]] for k, v in ci.items()]
        self._log_table(
            "test_ensemble_bootstrap_ci",
            columns=["metric", "mean", "std", "ci_lower", "ci_upper"],
            data=ci_data,
        )

        # Refit metrics
        if refit_results is not None:
            for k, v in refit_results["metrics"].items():
                self._log_summary(f"test/refit/{k}", v)
            for k, v in refit_results.get("op_metrics", {}).items():
                self._log_summary(f"test/refit/{k}", v)
            for k, v in refit_results["ci"].items():
                self._log_summary(f"test/refit/{k}_ci_lower", v["lower"])
                self._log_summary(f"test/refit/{k}_ci_upper", v["upper"])
            self._log_summary("test/refit/epoch", refit_results["epoch"])

        # Comparison table
        comparison_data = []
        for strategy, data in comparison.items():
            row = [
                strategy,
                data.get("auc", np.nan),
                data.get("auc_ci_lower", np.nan),
                data.get("auc_ci_upper", np.nan),
                data.get("balanced_accuracy", data.get("bal_accuracy", np.nan)),
                data.get("f1", data.get("f1_macro", np.nan)),
            ]
            comparison_data.append(row)
        
        self._log_table(
            "test_strategy_comparison",
            columns=["strategy", "auc", "auc_ci_lower", "auc_ci_upper", "balanced_accuracy", "f1"],
            data=comparison_data,
        )

        # Best strategy
        self._log_summary("test/best_strategy", best_strategy)
        self._log_summary(f"test/best_{primary_metric}", comparison[best_strategy]["auc"])