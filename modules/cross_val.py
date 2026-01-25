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
from modules.cv.kfolds import KFoldExperimentLIT
# from modules.cv.oof_exp import OOFExperimentLIT 
# from modules.cv.mccv_exp import MCCVExperimentLIT

from data.data_module import WSIMILDataModule
from models.MIL.wsi_model import WSIModel
from modules.cv.utils.train_utils import set_seed, cleanup_memory, load_data

# =============================================================================
# Main Experiment Class
# =============================================================================
class RunCV:
    """
    Cross-Validation Experiment
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
        self.cv_type = self.config.cv_type.lower()
        
        if self.cv_type == "k-fold":
            self.cv_experiment = KFoldExperimentLIT(
                config=config,
                device=self.device,
                wandb_run=wandb_run,
            )
        # elif self.cv_type == "oof":
        #     self.cv_experiment = OOFExperimentLIT(
        #         config=config,
        #         device=self.device,
        #         wandb_run=wandb_run,
        #     )
        # elif self.cv_type == "mccv":
        #     self.cv_experiment = MCCVExperimentLIT(
        #         config=config,
        #         device=self.device,
        #         wandb_run=wandb_run,
        #     )
        else:
            raise ValueError(f"Unknown cv_type: {self.cv_type} (use k-fold|oof|mccv)")
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
        print(f"# CV EXPERIMENT START")
        print(f"{'#'*60}")
        print(f"\nCV Type: {self.cv_type.upper()}")
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
        
        cv_results = self.cv_experiment.run_cv()

        # =====================================================================
        # STEP 2: Test Set Evaluation
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"STEP 2: TEST SET EVALUATION")
        print(f"{'='*60}")

        # Initial threshold
        threshold = 0.5
        
        # Evaluate all fold models on test set
        test_results = self.cv_experiment.test_cv(df_test = df_test,
                                                  df_train = df_train,
                                                  threshold=threshold,
                                                  compare_refit=True)
        
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

        return {
            "cv_results": cv_results,
            "test_results": test_results,
            # "final_model": final_model,
            # "final_strategy": final_strategy,
            # "final_model_results": final_model_results,
        }