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
from modules.cv.callbacks import BestStateTracker, FoldMetricLogger
from utils.cv.eval_utils import (
    logits_to_probs,
    compute_metrics,
    bootstrap_ci,
    tune_threshold_youden,
    operating_point_metrics,
    collect_predictions,
)
from utils.cv.train_utils import set_seed, cleanup_memory, load_data

from data.data_module import WSIMILDataModule
from models.MIL.wsi_model import WSIModel


# =============================================================================
# DataModule Variants
# =============================================================================
class FullTrainDataModule(WSIMILDataModule):
    """All data for training, no validation."""

    def _build_splits(self):
        self.train_ids = self.data_df[self.id_col].astype(str).tolist()
        self.val_ids = []
        self.test_ids = []


class TestOnlyDataModule(WSIMILDataModule):
    """All data for testing only."""

    def _build_splits(self):
        self.train_ids = []
        self.val_ids = []
        self.test_ids = self.data_df[self.id_col].astype(str).tolist()


# =============================================================================
# Ensemble Model Class
# =============================================================================
class EnsembleWSIModel(nn.Module):
    """
    Ensemble of K WSI models for inference.
    
    Supports multiple aggregation strategies:
    - 'mean_prob': Average predicted probabilities (default, recommended)
    - 'mean_logit': Average logits before softmax
    - 'majority_vote': Hard voting on predictions
    
    Attributes:
        models: List of trained WSIModel instances
        n_classes: Number of output classes
        aggregation: Strategy for combining predictions
    """

    def __init__(
        self,
        models: List[WSIModel],
        n_classes: int,
        aggregation: Literal["mean_prob", "mean_logit", "majority_vote"] = "mean_prob",
    ):
        super().__init__()
        
        if not models:
            raise ValueError("At least one model is required for ensemble")
        
        self.models = nn.ModuleList(models)
        self.n_classes = n_classes
        self.aggregation = aggregation
        self.n_models = len(models)
        
        # Set all models to eval mode
        for m in self.models:
            m.eval()

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        return_individual: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input features [1, N, D] (single bag)
            return_individual: If True, also return individual model outputs
            
        Returns:
            logits: Aggregated logits [1, C]
            log_dict: Dictionary with aggregation info and optionally individual outputs
        """
        all_logits = []
        all_probs = []
        
        for model in self.models:
            logits_i, _ = model(x)
            all_logits.append(logits_i)
            all_probs.append(torch.softmax(logits_i, dim=-1))
        
        # Stack: [K, B, C]
        stacked_logits = torch.stack(all_logits, dim=0)
        stacked_probs = torch.stack(all_probs, dim=0)
        
        if self.aggregation == "mean_prob":
            # Average probabilities, then convert back to logits for compatibility
            mean_probs = stacked_probs.mean(dim=0)  # [B, C]
            # Use log for numerical stability (these are "pseudo-logits")
            ensemble_logits = torch.log(mean_probs + 1e-8)
            
        elif self.aggregation == "mean_logit":
            ensemble_logits = stacked_logits.mean(dim=0)  # [B, C]
            
        elif self.aggregation == "majority_vote":
            # Get predictions from each model
            preds = stacked_probs.argmax(dim=-1)  # [K, B]
            # One-hot encode and sum votes
            votes = torch.zeros(preds.shape[1], self.n_classes, device=x.device)
            for k in range(self.n_models):
                votes.scatter_add_(1, preds[k].unsqueeze(-1), torch.ones_like(preds[k].unsqueeze(-1).float()))
            # Convert votes to pseudo-logits
            ensemble_logits = votes  # [B, C]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        log_dict = {
            "aggregation": self.aggregation,
            "n_models": self.n_models,
        }
        
        if return_individual:
            log_dict["individual_logits"] = stacked_logits
            log_dict["individual_probs"] = stacked_probs
        
        return ensemble_logits, log_dict

    @torch.no_grad()
    def forward_with_attention(
        self,
        x: torch.Tensor,
        attention_aggregation: Literal["mean", "max", "all"] = "mean",
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with attention weights from all models.
        
        Args:
            x: Input features [1, N, D]
            attention_aggregation: How to aggregate attention across models
                - 'mean': Average attention weights
                - 'max': Max attention weights
                - 'all': Return all attention weights stacked
                
        Returns:
            logits: Aggregated logits [1, C]
            attention: Aggregated attention weights [1, N] or [K, 1, N] if 'all'
            log_dict: Dictionary with additional info
        """
        all_logits = []
        all_probs = []
        all_attentions = []
        
        for model in self.models:
            logits_i, log_dict_i = model(x, return_raw_attention=True)
            all_logits.append(logits_i)
            all_probs.append(torch.softmax(logits_i, dim=-1))
            
            # Extract attention - handle different model architectures
            attn = log_dict_i.get("attention", log_dict_i.get("raw_attention", None))
            if attn is not None:
                # Ensure shape is [B, N]
                if attn.dim() == 3:
                    attn = attn.squeeze(1)  # [B, 1, N] -> [B, N]
                all_attentions.append(attn)
        
        # Aggregate logits
        stacked_probs = torch.stack(all_probs, dim=0)
        mean_probs = stacked_probs.mean(dim=0)
        ensemble_logits = torch.log(mean_probs + 1e-8)
        
        # Aggregate attention
        if all_attentions:
            stacked_attn = torch.stack(all_attentions, dim=0)  # [K, B, N]
            
            if attention_aggregation == "mean":
                ensemble_attn = stacked_attn.mean(dim=0)  # [B, N]
            elif attention_aggregation == "max":
                ensemble_attn = stacked_attn.max(dim=0).values  # [B, N]
            elif attention_aggregation == "all":
                ensemble_attn = stacked_attn  # [K, B, N]
            else:
                raise ValueError(f"Unknown attention_aggregation: {attention_aggregation}")
        else:
            ensemble_attn = None
            warnings.warn("No attention weights found in model outputs")
        
        log_dict = {
            "aggregation": self.aggregation,
            "attention_aggregation": attention_aggregation,
            "n_models": self.n_models,
        }
        
        return ensemble_logits, ensemble_attn, log_dict

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits, _ = self.forward(x)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        probs = self.predict_proba(x)
        return probs.argmax(dim=-1)


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
        self._n_classes: Optional[int] = None
        self._feature_dim: Optional[int] = None
        
        mode_str = "FINE-TUNE" if config.mode == "ft" else "LINEAR PROBE"
        print(f"\n[KFoldExperimentLIT] Mode: {mode_str}")
        print(f"[KFoldExperimentLIT] LR encoder/head: {config.encoder_lr} / {config.head_lr}")
        print(f"[KFoldExperimentLIT] WD encoder/head: {config.encoder_wd} / {config.head_wd}")
        print(f"[KFoldExperimentLIT] Scheduler: {config.scheduler}")

    # -------------------------------------------------------------------------
    # Logging Utilities
    # -------------------------------------------------------------------------
    def _log_summary(self, key: str, value: Any) -> None:
        if self.wandb_run is not None:
            self.wandb_run.summary[key] = value

    def _log_table(self, name: str, columns: List[str], data: List[List[Any]]) -> None:
        if self.wandb_run is not None and WANDB_AVAILABLE:
            self.wandb_run.log({name: wandb.Table(columns=columns, data=data)})

    def _log_metric(self, key: str, value: Any, step: Optional[int] = None) -> None:
        if self.wandb_run is not None:
            if step is not None:
                self.wandb_run.log({key: value}, step=step)
            else:
                self.wandb_run.log({key: value})

    def _reset_state(self) -> None:
        self._fold_results = []
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
        # monitor = "val/roc_auc" if dm.num_classes <= 2 else "val/roc_auc_macro"
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
        self._log_summary(f"fold_{fold}/n_train", len(dm.train_ids))
        self._log_summary(f"fold_{fold}/n_val", len(dm.val_ids))

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
        
        print(f"\n{'='*60}")
        print(f"CV TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Successfully trained {len(self._fold_results)} folds")
        print(f"Best epochs per fold: {best_epochs}")
        print(f"Median best epoch: {int(np.median(best_epochs)) if best_epochs else 'N/A'}")

        # Log summary
        self._log_summary("cv/n_folds", len(self._fold_results))
        self._log_summary("cv/median_best_epoch", int(np.median(best_epochs)) if best_epochs else None)

        return {
            "fold_results": self._fold_results,
            "n_folds": len(self._fold_results),
            "best_epochs": best_epochs,
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

    def build_ensemble(
        self,
        aggregation: Literal["mean_prob", "mean_logit", "majority_vote"] = "mean_prob",
    ) -> EnsembleWSIModel:
        """
        Build ensemble model from all fold checkpoints.
        
        Args:
            aggregation: Strategy for combining predictions
            
        Returns:
            EnsembleWSIModel containing all K fold models
        """
        if not self._fold_results:
            raise RuntimeError("No fold results available. Run run_cv() first.")

        print(f"\n[Ensemble] Building ensemble with {len(self._fold_results)} models")
        print(f"[Ensemble] Aggregation strategy: {aggregation}")

        models = []
        for fold_result in self._fold_results:
            model = self._load_fold_model(fold_result)
            model.eval()
            models.append(model)

        n_classes = self._fold_results[0]["n_classes"]
        
        ensemble = EnsembleWSIModel(
            models=models,
            n_classes=n_classes,
            aggregation=aggregation,
        )
        
        print(f"[Ensemble] Successfully built ensemble with {ensemble.n_models} models")
        
        return ensemble

    # -------------------------------------------------------------------------
    # Test Evaluation (on held-out test set)
    # -------------------------------------------------------------------------
    def test_cv(
        self,
        df_test: pd.DataFrame,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluate all K fold models on the held-out test set.
        
        This is where we compute and report all metrics:
        1. Per-model metrics (shows variance across folds)
        2. Ensemble metrics (best performance)
        3. Bootstrap confidence intervals
        
        Args:
            df_test: Test set DataFrame
            threshold: Decision threshold for binary classification
            
        Returns:
            Dictionary containing:
            - per_model_metrics: List of metrics for each fold model
            - per_model_summary: Mean±std across fold models
            - ensemble_metrics: Metrics for ensemble prediction
            - ensemble_ci: Bootstrap confidence intervals
            - threshold: Threshold used (may be tuned)
        """
        if not self._fold_results:
            raise RuntimeError("No fold results available. Run run_cv() first.")

        if len(df_test) == 0:
            warnings.warn("Empty test set")
            return {
                "per_model_metrics": [],
                "per_model_summary": {},
                "ensemble_metrics": {},
                "ensemble_ci": {},
                "threshold": threshold,
            }

        print(f"\n{'='*60}")
        print(f"TEST SET EVALUATION")
        print(f"{'='*60}")
        print(f"Test samples: {len(df_test)}")
        print(f"Number of fold models: {len(self._fold_results)}")
        print(f"Initial threshold: {threshold:.4f}")

        # Create test data module
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

        # Collect predictions from each fold model
        all_fold_logits = []
        all_fold_probs = []
        labels = None

        print("\n[Test] Collecting predictions from each fold model...")
        
        for i, fold_result in enumerate(self._fold_results):
            model = self._load_fold_model(fold_result)
            model.to(self.device)
            model.eval()

            logits_i, labels_i = collect_predictions(model, test_loader, self.device)
            probs_i = logits_to_probs(logits_i)

            all_fold_logits.append(logits_i)
            all_fold_probs.append(probs_i)

            if labels is None:
                labels = labels_i
            
            print(f"  Fold {fold_result['fold']}: collected {len(logits_i)} predictions")

            del model
            cleanup_memory()

        y_true = labels.numpy()
        n_classes = all_fold_probs[0].shape[1]
        is_binary = n_classes == 2

        # ---------------------------------------------------------------------
        # Per-model metrics
        # ---------------------------------------------------------------------
        print("\n[Test] Computing per-model metrics...")
        per_model_metrics = []
        
        for i, (fold_result, probs_i) in enumerate(zip(self._fold_results, all_fold_probs)):
            th = threshold if is_binary else None
            metrics_i = compute_metrics(y_true, probs_i, threshold=th)
            metrics_i["fold"] = fold_result["fold"]
            per_model_metrics.append(metrics_i)
            
            # Log to wandb
            for k, v in metrics_i.items():
                if k != "fold":
                    self._log_summary(f"test/fold_{fold_result['fold']}/{k}", v)

        # Aggregate per-model metrics
        metric_keys = [k for k in per_model_metrics[0].keys() if k != "fold"]
        per_model_summary = {}
        
        for mk in metric_keys:
            vals = [m[mk] for m in per_model_metrics if np.isfinite(m.get(mk, np.nan))]
            if vals:
                arr = np.array(vals)
                per_model_summary[mk] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "values": vals,
                }

        # Print per-model summary
        print("\n" + "="*50)
        print("PER-MODEL TEST METRICS (mean ± std)")
        print("="*50)
        for mk, stats in per_model_summary.items():
            print(f"  {mk}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                  f"[{stats['min']:.4f}, {stats['max']:.4f}]")
            self._log_summary(f"test/per_model/{mk}_mean", stats["mean"])
            self._log_summary(f"test/per_model/{mk}_std", stats["std"])

        # ---------------------------------------------------------------------
        # Ensemble metrics
        # ---------------------------------------------------------------------
        print("\n[Test] Computing ensemble metrics...")
        
        # Mean probability ensemble
        stacked_probs = np.stack(all_fold_probs, axis=0)  # [K, N, C]
        ensemble_probs = stacked_probs.mean(axis=0)  # [N, C]
        
        th = threshold if is_binary else None
        ensemble_metrics = compute_metrics(y_true, ensemble_probs, threshold=th)

        # Operating point metrics for binary classification
        op_metrics = {}
        if is_binary:
            op_metrics = operating_point_metrics(y_true, ensemble_probs[:, 1], threshold)

        # Print ensemble metrics
        print("\n" + "="*50)
        print("ENSEMBLE TEST METRICS (point estimates)")
        print("="*50)
        for k, v in ensemble_metrics.items():
            print(f"  {k}: {v:.4f}")
            self._log_summary(f"test/ensemble/{k}", v)
        
        if op_metrics:
            print("\nOperating point metrics:")
            for k, v in op_metrics.items():
                print(f"  {k}: {v:.4f}")
                self._log_summary(f"test/ensemble/{k}", v)

        # ---------------------------------------------------------------------
        # Bootstrap confidence intervals for ensemble
        # ---------------------------------------------------------------------
        print(f"\n[Test] Computing bootstrap CI ({self.config.n_bootstraps} iterations)...")
        
        ensemble_ci = bootstrap_ci(
            y_true,
            ensemble_probs,
            n_bootstraps=self.config.n_bootstraps,
            ci=0.95,
            threshold=th,
            seed=self.config.seed,
            stratified=True,
        )

        # Print bootstrap CI
        print("\n" + "="*50)
        print("ENSEMBLE TEST METRICS (95% Bootstrap CI)")
        print("="*50)
        ci_table_data = []
        for k, v in ensemble_ci.items():
            print(f"  {k}: {v['mean']:.4f} [{v['lower']:.4f}, {v['upper']:.4f}]")
            self._log_summary(f"test/ensemble/{k}_ci_lower", v["lower"])
            self._log_summary(f"test/ensemble/{k}_ci_upper", v["upper"])
            ci_table_data.append([k, v["mean"], v["std"], v["lower"], v["upper"]])

        # Log tables
        self._log_table(
            "test_per_model_metrics",
            columns=["fold"] + metric_keys,
            data=[[m["fold"]] + [m.get(k, np.nan) for k in metric_keys] for m in per_model_metrics],
        )
        
        self._log_table(
            "test_ensemble_bootstrap_ci",
            columns=["metric", "mean", "std", "lower", "upper"],
            data=ci_table_data,
        )

        # Log threshold
        self._log_summary("test/threshold", threshold)

        # Cleanup
        del dm
        cleanup_memory()

        return {
            "per_model_metrics": per_model_metrics,
            "per_model_summary": per_model_summary,
            "ensemble_metrics": ensemble_metrics,
            "ensemble_ci": ensemble_ci,
            "ensemble_probs": ensemble_probs,
            "op_metrics": op_metrics,
            "threshold": threshold,
            "y_true": y_true,
        }

    # -------------------------------------------------------------------------
    # Refit on Full Training Data
    # -------------------------------------------------------------------------
    def train_final(self, df_train: pd.DataFrame) -> WSIClassificationModule:
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

        # Validate target epoch
        if target_epoch > self.config.max_epochs:
            warnings.warn(
                f"Target epoch {target_epoch} > max_epochs {self.config.max_epochs}. "
                f"Using max_epochs."
            )
            target_epoch = self.config.max_epochs

        if target_epoch <= self.config.warmup_epochs:
            warnings.warn(
                f"Target epoch {target_epoch} <= warmup_epochs {self.config.warmup_epochs}. "
                f"Model will stop during warmup phase."
            )

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

        # Custom callback to stop at target epoch
        class StopAtEpoch(Callback):
            def __init__(self, target: int):
                self.target = target

            def on_train_epoch_end(self, trainer, pl_module):
                if trainer.current_epoch + 1 >= self.target:
                    trainer.should_stop = True
                    print(f"\n[StopAtEpoch] Stopping at epoch {trainer.current_epoch + 1}")

        # Train
        trainer = self._create_trainer(
            max_epochs=self.config.max_epochs,
            callbacks=[StopAtEpoch(target_epoch)],
            fold=None,
        )
        trainer.fit(model, datamodule=dm)

        actual_epochs = trainer.current_epoch + 1
        print(f"\n[Final Model] Trained for {actual_epochs} epochs (target: {target_epoch})")
        print(f"[Final Model] Training samples: {len(dm.train_ids)}")

        # Log
        self._log_summary("final/strategy", "refit")
        self._log_summary("final/target_epoch", target_epoch)
        self._log_summary("final/actual_epoch", actual_epochs)
        self._log_summary("final/n_train_samples", len(dm.train_ids))

        del dm, trainer
        cleanup_memory()

        return model

    # -------------------------------------------------------------------------
    # Evaluate Single Model on Test Set
    # -------------------------------------------------------------------------
    def evaluate_model(
        self,
        model: Union[WSIClassificationModule, WSIModel, EnsembleWSIModel],
        df_test: pd.DataFrame,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluate a single model (or ensemble) on test set.
        
        Args:
            model: Model to evaluate
            df_test: Test DataFrame
            threshold: Decision threshold
            
        Returns:
            Dictionary with metrics and CI
        """
        print(f"\n{'='*60}")
        print(f"MODEL EVALUATION")
        print(f"{'='*60}")

        if len(df_test) == 0:
            warnings.warn("Empty test set")
            return {"metrics": {}, "ci": {}}

        # Create test data module
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

        # Get the actual model to use
        if isinstance(model, WSIClassificationModule):
            eval_model = model.model
        else:
            eval_model = model

        eval_model.to(self.device)
        eval_model.eval()

        # Collect predictions
        logits, labels = collect_predictions(eval_model, dm.test_dataloader(), self.device)
        y_true = labels.numpy()
        y_prob = logits_to_probs(logits)

        is_binary = y_prob.shape[1] == 2
        th = threshold if is_binary else None

        # Compute metrics
        metrics = compute_metrics(y_true, y_prob, threshold=th)
        
        # Bootstrap CI
        ci = bootstrap_ci(
            y_true,
            y_prob,
            n_bootstraps=self.config.n_bootstraps,
            ci=0.95,
            threshold=th,
            seed=self.config.seed,
            stratified=True,
        )

        # Operating point metrics
        op_metrics = {}
        if is_binary:
            op_metrics = operating_point_metrics(y_true, y_prob[:, 1], threshold)

        # Print results
        print("\nPoint Estimates:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        print("\n95% Bootstrap CI:")
        for k, v in ci.items():
            print(f"  {k}: {v['mean']:.4f} [{v['lower']:.4f}, {v['upper']:.4f}]")

        del dm
        cleanup_memory()

        return {
            "metrics": metrics,
            "ci": ci,
            "op_metrics": op_metrics,
            "threshold": threshold,
        }

    # -------------------------------------------------------------------------
    # Main Run Function
    # -------------------------------------------------------------------------
    def run(
        self,
        final_strategy: Literal["ensemble", "refit", "best_fold"] = "ensemble",
        save_path: Optional[str] = None,
        ensemble_aggregation: Literal["mean_prob", "mean_logit", "majority_vote"] = "mean_prob",
    ) -> Dict[str, Any]:
        """
        Run the complete K-Fold CV experiment pipeline.
        
        Pipeline:
        1. Train K models via K-Fold CV (validation fold for early stopping only)
        2. Evaluate all K models on held-out test set
        3. Build final model (ensemble, refit, or best_fold)
        4. Report comprehensive metrics with bootstrap CI
        
        Args:
            final_strategy: Strategy for final model
                - 'ensemble': Use ensemble of all K fold models (recommended)
                - 'refit': Retrain on full training data
                - 'best_fold': Use single best fold model
            save_path: Path to save final model weights
            ensemble_aggregation: Aggregation method for ensemble
            
        Returns:
            Dictionary containing all results
        """
        self._reset_state()
        
        print(f"\n{'#'*60}")
        print(f"# K-FOLD CV EXPERIMENT")
        print(f"# Final Strategy: {final_strategy.upper()}")
        print(f"{'#'*60}")

        # Load data
        df = load_data(self.config.csv_path)
        df_train = df[df["k_fold"] >= 0].copy()
        df_test = df[df["k_fold"] == -1].copy()

        print(f"\nData Split:")
        print(f"  Training samples: {len(df_train)}")
        print(f"  Test samples: {len(df_test)}")

        self._log_summary("data/n_train", len(df_train))
        self._log_summary("data/n_test", len(df_test))

        # =====================================================================
        # STEP 1: K-Fold Cross-Validation (Training)
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"STEP 1: K-FOLD CROSS-VALIDATION TRAINING")
        print(f"{'='*60}")
        
        cv_results = self.run_cv()

        # =====================================================================
        # STEP 2: Test Set Evaluation
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"STEP 2: TEST SET EVALUATION")
        print(f"{'='*60}")

        # Initial threshold
        threshold = 0.5
        
        # Evaluate all fold models on test set
        test_results = self.test_cv(df_test, threshold=threshold)

        # # =====================================================================
        # # STEP 3: Build Final Model
        # # =====================================================================
        # print(f"\n{'='*60}")
        # print(f"STEP 3: FINAL MODEL ({final_strategy.upper()})")
        # print(f"{'='*60}")

        # final_model = None
        # final_model_results = None

        # if final_strategy == "ensemble":
        #     # Build ensemble from all fold models
        #     final_model = self.build_ensemble(aggregation=ensemble_aggregation)
        #     self._log_summary("final/strategy", "ensemble")
        #     self._log_summary("final/aggregation", ensemble_aggregation)
            
        #     # Note: Ensemble metrics already computed in test_cv
        #     print("\n[Final Model] Using ensemble of all fold models")
        #     print("[Final Model] Ensemble metrics are reported above")

        # elif final_strategy == "refit":
        #     # Retrain on full data
        #     final_module = self.train_final(df_train)
        #     final_model = final_module.model
            
        #     # Evaluate refit model
        #     print("\n[Final Model] Evaluating refit model on test set...")
        #     final_model_results = self.evaluate_model(final_module, df_test, threshold)
            
        #     # Log refit-specific metrics
        #     for k, v in final_model_results["metrics"].items():
        #         self._log_summary(f"test/refit/{k}", v)

        # elif final_strategy == "best_fold":
        #     # Use best fold model
        #     scores = [
        #         r.get("best_val_score", -np.inf) or -np.inf
        #         for r in self._fold_results
        #     ]
        #     best_idx = int(np.argmax(scores))
        #     best_fold = self._fold_results[best_idx]

        #     print(f"\n[Final Model] Using best fold model: Fold {best_fold['fold']}")
        #     print(f"[Final Model] Best fold validation score: {scores[best_idx]:.4f}")

        #     final_model = self._load_fold_model(best_fold)
            
        #     self._log_summary("final/strategy", "best_fold")
        #     self._log_summary("final/best_fold", best_fold["fold"])

        #     # Best fold metrics are in test_results["per_model_metrics"]
        #     best_fold_metrics = test_results["per_model_metrics"][best_idx]
        #     print(f"\n[Final Model] Best fold test metrics:")
        #     for k, v in best_fold_metrics.items():
        #         if k != "fold":
        #             print(f"  {k}: {v:.4f}")

        # else:
        #     raise ValueError(f"Unknown final_strategy: {final_strategy}")

        # # =====================================================================
        # # STEP 4: Save Model
        # # =====================================================================
        # if save_path and final_model is not None:
        #     print(f"\n{'='*60}")
        #     print(f"STEP 4: SAVING MODEL")
        #     print(f"{'='*60}")

        #     save_dir = os.path.dirname(save_path)
        #     if save_dir:
        #         os.makedirs(save_dir, exist_ok=True)

        #     torch.save(final_model.state_dict(), save_path)
        #     print(f"Model saved to: {save_path}")

        #     # Log artifact to wandb
        #     if self.wandb_run is not None and WANDB_AVAILABLE:
        #         artifact = wandb.Artifact(
        #             name=f"model-{self.wandb_run.id}",
        #             type="model",
        #             description=f"Final model (strategy={final_strategy})",
        #         )
        #         artifact.add_file(save_path)
        #         self.wandb_run.log_artifact(artifact)

        # =====================================================================
        # Final Summary
        # =====================================================================
        print(f"\n{'#'*60}")
        print(f"# EXPERIMENT COMPLETE")
        print(f"{'#'*60}")
        print(f"\nFinal Strategy: {final_strategy}")
        print(f"Threshold: {threshold:.4f}")
        print(f"\nKey Results (Ensemble):")
        for k in ["roc_auc", "accuracy", "f1"]:
            if k in test_results["ensemble_ci"]:
                ci = test_results["ensemble_ci"][k]
                print(f"  {k}: {ci['mean']:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]")

        return {
            "cv_results": cv_results,
            "test_results": test_results,
            "threshold": threshold,
            # "final_model": final_model,
            # "final_strategy": final_strategy,
            # "final_model_results": final_model_results,
        }