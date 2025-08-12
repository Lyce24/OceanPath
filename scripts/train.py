from __future__ import annotations

import numpy as np
from typing import Dict, List
from contextlib import nullcontext

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, log_loss, roc_auc_score
)
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, StratifiedGroupKFold
from torch.amp import autocast
from torch.amp import GradScaler
import copy
import random
from typing import Dict, List, Tuple, Callable, Optional
from torch.nn.utils import clip_grad_norm_
# from timm.scheduler import create_scheduler_v2
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
from collections import defaultdict
import math
import torch.nn.functional as F
import json

# -----------------------------
# Repro
# -----------------------------
SEED = 42
def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _to_device(d) -> torch.device:
    return d if isinstance(d, torch.device) else torch.device(d)

# -----------------------------
# Dataset (returns with or without key)
# -----------------------------
class WSIBagDataset(Dataset):
    """
    features_dict: {id: Tensor [N,C] (float16)}
    labels_dict:   {id: torch.long scalar OR int}
    """
    def __init__(
        self,
        ids: List[str],
        features_dict: Dict[str, torch.Tensor],
        labels_dict: Dict[str, torch.Tensor | int],
        bag_size: Optional[int] = None,
        replacement: bool = True,
        return_key: bool = False,
        cpu_cast_float32: bool = False,
        preloaded: bool = True,
        feature_dir: Optional[str] = None,
        precision: int = 16,
        mmap_mode: str = "c",
    ):
        self.ids = list(ids)
        self.X = features_dict
        self.y = labels_dict
        self.bag_size = bag_size
        self.replacement = replacement
        self.return_key = return_key
        self.cpu_cast_float32 = cpu_cast_float32
        self.preloaded = preloaded
        self.precision = int(precision)
        if self.precision not in (16, 32):
            raise ValueError("precision must be 16 or 32")
        self._mode = "preloaded" if preloaded else "mmap"
        self._feature_dim = None
        
        if not preloaded:
            if feature_dir is None:
                raise ValueError("preloaded=False requires feature_dir (dir with features.npy & metadata.json).")
            meta_p = os.path.join(feature_dir, f"combined_mmap_{precision}", "metadata.json")
            feat_p = os.path.join(feature_dir, f"combined_mmap_{precision}", "features.npy")
            if not os.path.exists(meta_p):
                raise FileNotFoundError(f"metadata.json not found: {meta_p}")
            if not os.path.exists(feat_p):
                raise FileNotFoundError(f"features.npy not found: {feat_p}")
            with open(meta_p, "r") as f:
                meta = json.load(f)
            if "feature_dim" not in meta:
                raise KeyError("metadata.json missing 'feature_dim'.")
            self.feature_dim = int(meta["feature_dim"])

            total_patches = int(meta.get("total_patches", 0))
            if total_patches <= 0:
                raise ValueError("metadata.json must include a positive 'total_patches' in lazy mode.")

            self._np_dtype = {16: np.float16, 32: np.float32}.get(int(precision))
            if self._np_dtype is None:
                raise ValueError("precision must be 16 or 32.")
            self._feat_path = feat_p
            self._shape0 = total_patches
            self._mmap_mode = mmap_mode
            # open memmap (will be re-opened in workers)
            self._features_mmap = np.memmap(
                self._feat_path, dtype=self._np_dtype, mode=self._mmap_mode,
                shape=(self._shape0, self.feature_dim)
            )            
            
    def __len__(self) -> int:
        return len(self.ids)

    # ---------- multiprocessing: re-open memmap in workers ----------
    def __getstate__(self):
        st = self.__dict__.copy()
        st["_features_mmap"] = None  # don't pickle live handle
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        if not self.preloaded and self._features_mmap is None:
            self._features_mmap = np.memmap(
                self._feat_path, dtype=self._np_dtype, mode=self._mmap_mode,
                shape=(self._shape0, self.feature_dim)
            )

    def _sample_instances(self, bag: torch.Tensor) -> torch.Tensor:
        N = bag.size(0)
        if self.bag_size is None or N == self.bag_size:
            return bag
        if N > self.bag_size:
            idx = torch.randperm(N)[: self.bag_size]
            return bag[idx]
        if not self.replacement:
            return bag
        extra = torch.randint(0, N, (self.bag_size - N,))
        idx = torch.cat([torch.arange(N), extra], dim=0)
        return bag[idx]

    def __getitem__(self, i: int):
        key = self.ids[i]
        if self.preloaded:
            bag = self.X[key]
        else:
            s, e = self.X[key]
            bag = self._features_mmap[s:e]
            
        if isinstance(bag, np.ndarray):
            bag = torch.from_numpy(bag)
        bag = bag.to(dtype=torch.float16, copy=False)  # features are preloaded fp16

        bag = self._sample_instances(bag)

        # Cast to float32 if staying on CPU (safer; many CPU ops don't support fp16)
        if self.cpu_cast_float32:
            bag = bag.float()

        lab = self.y[key]
        if torch.is_tensor(lab):
            lab = int(lab.item())
        else:
            lab = int(lab)

        if self.return_key:
            return bag, lab, key
        return bag, lab

# -----------------------------
# Utilities
# -----------------------------
@torch.no_grad()
def _predict_logits_with_keys(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    model.eval()
    outs, ys, keys = [], [], []
    ctx = autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda"))
    with ctx:
        for bag, lab, key in loader:
            bag = bag.to(device, non_blocking=True)  # keep dtype as-is (fp16 on cuda)
            out = model(bag)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)  # [B] -> [B,1] for binary
            outs.append(logits.detach().cpu())
            ys.append(torch.as_tensor(lab))
            if isinstance(key, list):
                keys.extend(key)
            else:
                keys.append(key)
    return torch.cat(outs, 0), torch.cat(ys, 0), keys

def make_stratified_group_splitter(y, groups=None, n_splits=5, seed=42):
    if groups is None:
        print("No groups provided, using StratifiedKFold.")
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    groups = np.asarray(groups)
    # If every group is unique, grouping has no effect → use StratifiedKFold
    if np.unique(groups).size == groups.size:
        print("All groups are unique, using StratifiedKFold.")
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

# =========================
# Metric aggregation
# =========================
def aggregate_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculates classification metrics for binary and multi-class tasks.

    - Works even if y_true labels aren't 0..C-1 (relabels internally).
    - Handles binary (C=2) and multi-class (C>2).
    - Precision/Recall/F1 are 'positive class' for binary, macro for multi-class.
    - Safely computes log_loss and ROC-AUC; returns NaN when not computable.
    - Accepts y_prob as probability matrix (N, C). Will renormalize rows if slightly off.

    Args:
        y_true: Array of true labels, shape (N,).
        y_prob: Array of predicted probabilities, shape (N, C).

    Returns:
        dict of metrics.
    """
    y_true = np.asarray(y_true).ravel()
    if y_true.size == 0 or y_prob.size == 0:
        return {}

    if y_prob.ndim != 2:
        raise ValueError(f"y_prob must be 2D (N,C); got shape {y_prob.shape}")

    N, C = y_prob.shape
    if N != y_true.shape[0]:
        raise ValueError(f"Mismatched N: len(y_true)={y_true.shape[0]} vs y_prob.shape[0]={N}")

    # Relabel y_true to {0,1,...,K-1} to avoid issues with metrics expecting encoded labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_true)
    K = len(le.classes_)

    # If model outputs more classes than appear in y_true, we still keep all columns.
    # For metrics that require labels, we’ll pass labels=range(C) when supported.

    # Numerical safety for log_loss and argmax: clip + renormalize per row
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    y_prob_row_sums = y_prob_clipped.sum(axis=1, keepdims=True)
    # avoid division by zero in pathological inputs
    y_prob_safe = y_prob_clipped / np.where(y_prob_row_sums > 0, y_prob_row_sums, 1.0)

    # Predictions
    y_pred = np.argmax(y_prob_safe, axis=1)

    # Averaging rule
    avg_method = 'binary' if C == 2 else 'macro'

    metrics = {
        "accuracy": float(accuracy_score(y_enc, y_pred)),
        "precision": float(precision_score(y_enc, y_pred, average=avg_method, zero_division=0)),
        "recall": float(recall_score(y_enc, y_pred, average=avg_method, zero_division=0)),
        "f1": float(f1_score(y_enc, y_pred, average=avg_method, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_enc, y_pred)),
    }

    # Log loss (supports missing classes in y_true via labels=range(C))
    try:
        metrics["log_loss"] = float(log_loss(y_enc, y_prob_safe, labels=np.arange(C)))
    except Exception:
        metrics["log_loss"] = float('nan')

    # ROC-AUC
    try:
        if K <= 1:
            metrics["roc_auc"] = float('nan')
        elif C == 2:
            # Use probability of the positive class; map positive to column 1
            # If model produced only one column or columns inverted, we still assume col 1 is "positive".
            if C < 2:
                metrics["roc_auc"] = float('nan')
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_enc, y_prob_safe[:, 1]))
        else:
            # Multi-class AUC (macro OVR). Fails if a class missing in y_true; catch and set NaN.
            metrics["roc_auc"] = float(
                roc_auc_score(y_enc, y_prob_safe, multi_class="ovr", average="macro", labels=np.arange(C))
            )
    except Exception:
        metrics["roc_auc"] = float('nan')

    return metrics

def _oof_metrics_from_logits(y_true: np.ndarray, logits: torch.Tensor) -> Dict[str, float]:
    logits = logits.clone()
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)
    N, C = logits.shape
    # Convert to probabilities
    if C == 1:
        p = torch.sigmoid(logits).squeeze(1).numpy()
        y_prob = np.vstack([1 - p, p]).T
    else:
        y_prob = torch.softmax(logits, dim=1).numpy()

    return aggregate_metrics(y_true, y_prob)

def evaluate_classifier(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor = None,
    use_amp: bool = True,
) -> tuple[dict, float]:
    """Evaluates a binary or multi-class classifier.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        loader (DataLoader): The DataLoader for the evaluation dataset.
        device (str): The device to run evaluation on ('cpu' or 'cuda').
        class_weights (torch.Tensor, optional): Weights for each class for the loss function.
        use_amp (bool): Whether to use automatic mixed precision.

    Returns:
        A tuple containing:
        - metrics (dict): A dictionary of performance metrics, including:
            - accuracy, precision, recall, f1, balanced_accuracy, log_loss, roc_auc.
            - Precision, recall, and F1 are macro-averaged for multi-class.
        - avg_loss (float): The mean cross-entropy loss over the entire dataset.
    """
    # --- Setup ---
    model = model.to(device)
    model.eval()

    # Use reduction='sum' to accurately average loss across all samples
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="sum")

    total_loss = 0.0
    all_probs = []
    all_labels = []

    # --- Evaluation Loop ---
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            # Automatic mixed precision
            with autocast(device_type=device.type, enabled=use_amp):
                if model.encoder_type == "CLAM":
                    outputs, log_dict = model(inputs, return_raw_attention=True, labels=labels)
                    loss = loss_fn(outputs, labels)

                    if model.encoder_type == "CLAM" and 'instance_loss' in log_dict and log_dict['instance_loss'] != -1:
                        loss = 0.7 * loss + 0.3 * log_dict['instance_loss']

                else:
                    outputs, _ = model(inputs)
                    loss = loss_fn(outputs, labels)
                probs = torch.softmax(outputs, dim=1)

            # Note: loss is already summed over the batch due to reduction='sum'
            total_loss += loss.item()
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Handle empty loader case
    if not all_labels:
        return {}, 0.0

    # --- Post-processing and Metric Calculation ---
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    num_samples = len(y_true)

    # Calculate final average loss
    avg_loss = total_loss / max(1, num_samples)
    
    # --- Build Metrics Dictionary ---
    metrics = aggregate_metrics(y_true, y_prob)

    return metrics, avg_loss

def train_one_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
    class_weights: torch.Tensor = None,
    use_amp: bool = False,
    scaler: GradScaler = None,
    max_grad_norm: float = None,
    scheduler: Optional[Callable] = None,
    accum_steps: int = 1,
    epoch: int = 0,
    updates_per_epoch: int = 0
) -> float:
    """
    Gradient accumulation: performs one optimizer step every `accum_steps` batches.
    Set accum_steps = effective_batch_size / (physical batch size).
    """
    assert accum_steps >= 1
    model.train()
    total_loss = 0.0
    scaler = scaler or GradScaler(enabled=use_amp)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer.zero_grad(set_to_none=True)
    amp_enabled = bool(use_amp and device.type == "cuda")
    update_idx = 0  # counts optimizer updates within this epoch
    
    for batch_idx, (feats, labs) in enumerate(train_loader):
        feats = feats.to(device, non_blocking=True)
        labs  = labs.to(device, non_blocking=True).long()

        with autocast(device_type=device.type, enabled=amp_enabled):
            if getattr(model, "encoder_type", None) == "CLAM":
                logits, log_dict = model(feats, return_raw_attention=True, labels=labs)
                instance_loss = log_dict.get("instance_loss", -1) if isinstance(log_dict, dict) else -1
            else:
                logits, _ = model(feats)
                instance_loss = -1

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"Invalid logits at batch {batch_idx}. Sanitizing.")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

            if task != "survival":
                loss = loss_fn(logits, labs)
                if instance_loss is not None and instance_loss != -1:
                    loss = 0.7 * loss + 0.3 * instance_loss
            else:
                raise NotImplementedError("Survival task not implemented in this function.")

            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"Invalid loss ({loss.item()}) at batch {batch_idx}")

        # For reporting: accumulate the real (unscaled) loss value
        total_loss += float(torch.clamp(loss, min=0.0).detach().cpu())

        # Scale loss for accumulation
        loss_for_backward = torch.clamp(loss, min=0.0) / accum_steps
        scaler.scale(loss_for_backward).backward()

        # Step when we've accumulated enough micro-batches, or at the very end
        do_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(train_loader))
        if do_step:
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            if scheduler is not None:
                scheduler.step()
                update_idx += 1

    avg_loss = total_loss / max(1, len(train_loader))
    return avg_loss

def refit_to_all(
    model_builder: Callable[[], torch.nn.Module],
    features,
    labels,
    cv_results,
    task,
    device,
    epochs,
    lr,
    l2_reg,
    scheduler,
    class_weights,
    batch_size,
    bag_size,
    replacement,
    use_amp,
    pretrained_state=None,
    pretrained_model_path=None,
    warmup_epochs=0,
    step_on_epochs=False,
    accum_steps=1,
    preloaded=True,
    feature_dir=None  # New: directory for preloaded features
):
    """
    Refits the model on the entire training and validation dataset.

    This function trains a new model from scratch on all available data (excluding the test set)
    for a number of epochs determined by the cross-validation results.
    """
    print("\n=== Refitting model on the full training + validation dataset ===")

    # Determine the number of epochs: use the ceiling of the mean epochs from CV
    # This is a common heuristic to avoid overfitting while using all data.
    if cv_results["fold_epochs"]:
        # num_epochs = int(np.ceil(np.median(cv_results["fold_epochs"])))
        # print(f"Training for {num_epochs} epochs (based on average from CV).")
        p75 = np.percentile(cv_results["fold_epochs"], 75)
        num_epochs = int(np.ceil(p75))
        print(f"Training for {num_epochs} epochs (based on 75th percentile from CV).")
    else:
        # Fallback to the max epochs if early stopping didn't trigger in any fold
        num_epochs = epochs
        print(f"CV did not trigger early stopping. Training for the full {num_epochs} epochs.")

    train_keys = list(labels.keys())

    # Create the full dataset and loader
    full_train_ds = WSIBagDataset(train_keys, features, labels, bag_size=bag_size, replacement=replacement, preloaded=preloaded,
                            feature_dir=feature_dir)
    full_train_loader = DataLoader(full_train_ds, 
                                   batch_size=batch_size if (bag_size is not None and replacement) else 1,
                                   shuffle=True, num_workers=4, pin_memory=True)
    
    # Initialize a new model, optimizer, and scheduler
    refit_model = model_builder().to(device)

    # load or init weights
    if pretrained_state is not None:
        refit_model.load_state_dict(pretrained_state)
        print(f"Loaded pretrained state_dict.")
    elif pretrained_model_path and os.path.exists(pretrained_model_path):
        state = torch.load(pretrained_model_path, map_location=device)
        refit_model.load_state_dict(state)
        print(f"Loaded pretrained model from '{pretrained_model_path}'.")
    else:
        refit_model.initialize_weights()
        print(f"Initialized weights from scratch.")

    optimizer = torch.optim.AdamW(refit_model.parameters(), 
                                    lr=lr, 
                                    betas=(0.9, 0.999),
                                    eps=1e-8,
                                    weight_decay=l2_reg)

    sched = None
    updates_per_epoch = math.ceil(len(full_train_loader) / accum_steps)

    if scheduler == 'cosine':
        if step_on_epochs:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.000005)
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates_per_epoch * epochs, eta_min=0.000005)

    scaler = GradScaler(enabled=use_amp)

    # Main training loop
    for ep in range(1, num_epochs + 1):
        # Assuming train_one_epoch is defined elsewhere
        train_loss = train_one_epoch(
            model=refit_model, train_loader=full_train_loader,
            optimizer=optimizer,
            device=device, task=task, class_weights=class_weights,
            use_amp=use_amp, scaler=scaler, max_grad_norm=5.0,
            scheduler=None if step_on_epochs else sched,
            accum_steps=accum_steps, epoch=ep - 1,  # zero-indexed for logging
            updates_per_epoch=updates_per_epoch
        )
        if sched and step_on_epochs:
            sched.step()
        
        if ep % 5 == 0 or ep == num_epochs:
            print(f"[Refit | Epoch {ep}/{num_epochs}] TrainLoss={train_loss:.4f}")

    print("=== Refitting complete ===")
    return refit_model

# --- Helper Class for Model Ensembling ---
class ModelEnsemble(nn.Module):
    """
    Ensemble by combining member model outputs and returning logits for downstream use.

    Parameters
    ----------
    models : list[nn.Module]
        List of models that, when called, return either:
          - logits (Tensor), or
          - (logits, aux) tuple (the first item is logits)
    combine : {'logits', 'probs'}, default='logits'
        How to combine member outputs:
          - 'logits': arithmetic mean of logits (cheap; common in practice).
          - 'probs' : average probabilities, then convert back to logits.
                      For multi-class: returns log-probs (valid "logits" for softmax/argmax).
                      For binary (single-logit): returns a single logit log(p/(1-p)).
    eps : float
        Numerical stability for log/softmax/sigmoid conversions.
    """

    def __init__(self, models, combine: str = "logits", eps: float = 1e-8):
        super().__init__()
        if not models:
            raise ValueError("ModelEnsemble requires at least one model.")
        self.models = nn.ModuleList(models)
        self.combine = combine
        self.eps = eps
    
    # torch no grad decorator to avoid tracking gradients
    @torch.no_grad()
    def forward(self, x, return_raw_attention: bool = False) -> torch.Tensor:
        # Get predictions from each model in the ensemble
        logits_list = []
        attn_list = []

        # evaluate each submodel
        for m in self.models:
            m.eval()
            logits, ld = m(x, return_raw_attention=return_raw_attention)

            if not torch.is_floating_point(logits):
                logits = logits.float()

            # Standardize shape: allow [B] -> [B,1]
            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)

            logits_list.append(logits)

            if return_raw_attention and isinstance(ld, dict):
                # try common keys; take the first that exists
                att = None
                if "attention" in ld and ld["attention"] is not None:
                    att = ld["attention"]

                if att is not None:
                    attn_list.append(att)
        
        # stack across models -> weighted mean
        # logits_list[i]: [B, C]
        stacked = torch.stack(logits_list, dim=0)
        if self.combine == "logits":
            # Simple arithmetic mean in logit space
            combined_logits = stacked.mean(dim=0)  # [B, C]
        elif self.combine == "probs":
            # Average probabilities then return logits that reproduce those probs under softmax/sigmoid
            B, C = stacked.shape[-2], stacked.shape[-1]
            if C == 1:
                # Binary, single-logit case: use sigmoid
                probs = torch.sigmoid(stacked)              # [M, B, 1]
                p = probs.mean(dim=0).clamp(self.eps, 1 - self.eps)  # [B, 1]
                combined_logits = torch.log(p) - torch.log(1 - p)    # logit(p)
            else:
                # Multi-class: softmax then average
                probs = F.softmax(stacked, dim=-1)          # [M, B, C]
                p = probs.mean(dim=0).clamp_min(self.eps)   # [B, C], sums to 1
                # Return log-probs; downstream softmax(log p) = p, argmax identical
                combined_logits = torch.log(p)              # [B, C]
        else:
            raise ValueError("combine must be one of {'logits', 'probs'}")

        out_logs = {}
        if return_raw_attention and len(attn_list) > 0:
            # assume same shape across models for the same input
            A = torch.stack(attn_list, dim=0)  # [M, B, N...] or [M, N...] etc.
            out_logs["attention"] = A.mean(dim=0)  # [B, N...] or [N...]

        return combined_logits, out_logs

def ensemble_models(base_model, fold_states, device, topk=None, combine="logits", fold_scores=None):
    """
    Creates an ensemble from the best models of the cross-validation folds.

    Args:
        base_model: The base model architecture (e.g., an un-trained instance).
        fold_states (list): A list of state_dicts from each fold's best epoch.
        device: The device to load the models onto ('cpu' or 'cuda').
        topk (int, optional): If specified, ensemble only the top-k models based on fold_scores.
        fold_scores (list, optional): Validation scores for each fold. Required if topk is used.

    Returns:
        A ModelEnsemble instance containing the trained models.
    """
    models = []
    states_to_ensemble = fold_states

    if topk is not None:
        if fold_scores is None:
            raise ValueError("`fold_scores` must be provided to select top-k models.")
        # Get indices of the top-k scores in descending order
        sorted_indices = np.argsort(fold_scores)[::-1]
        topk_indices = sorted_indices[:topk]
        states_to_ensemble = [fold_states[i] for i in topk_indices]
        print(f"Ensembling top {len(states_to_ensemble)} models based on validation scores.")
    else:
        print(f"Ensembling all {len(states_to_ensemble)} models from CV.")
        
    for state_dict in states_to_ensemble:
        model = copy.deepcopy(base_model).to(device)
        model.load_state_dict(state_dict)
        model.eval()  # Set model to evaluation mode
        models.append(model)
        
    return ModelEnsemble(models = models, combine=combine, eps=1e-8).to(device)

# -----------------------------
# Main: k-fold OOF CV
# -----------------------------
def run_oof_cv_kfold(
    model_builder: Callable[[], torch.nn.Module],
    preloaded_features: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor | int],
    patient_id_mapping: Dict[str, str],
    task = "classification",
    key_metric = None,
    n_splits=5,
    device='cpu',
    epochs=60,
    lr=3e-4,
    l2_reg=1e-3,
    bag_size=None,
    replacement=True,
    scheduler="cosine",
    early_stopping=10,
    class_weights=None,
    pretrained_state=None,      # New: load pretrained weights if provided
    pretrained_model_path=None,  # New: path to pretrained model
    batch_size=1,
    warmup_epochs=0,
    use_amp=True,
    accum_steps=1,
    step_on_epochs=True,
    preloaded=True,  # New: whether to use preloaded features
    feature_dir=None  # New: directory for preloaded features
) -> Dict:
    """
    Returns:
      {
        'oof': {'logits': Tensor[N,C], 'y': np.ndarray[N], 'fold': np.ndarray[N], 'metrics': dict},
        'fold_best_states': List[dict],
        'fold_scores': List[float],
        'fold_best_metrics': List[dict],
    }
    """
    device = _to_device(device)

    print(f"\n=== Running Experiment with {n_splits}-Fold Cross-Validation ===")

    # --- ordered ids / labels / groups ---
    ids = list(labels.keys())
    y_vec = np.array([int(labels[k].item() if torch.is_tensor(labels[k]) else labels[k]) for k in ids], dtype=int)
    groups = np.array([patient_id_mapping[k] for k in ids])
    print(f"Task: {task}, Sample Size: {len(ids)} (train+val)")
    
    # --- sample to infer C ---
    C = model_builder().n_classes

    # --- OOF containers ---
    N = len(ids)
    oof_logits = torch.empty(N, C, dtype=torch.float32)
    oof_y      = y_vec.copy()
    oof_fold   = np.empty(N, dtype=int)
    key2row    = {k: i for i, k in enumerate(ids)}

    # --- CV splitter ---
    skf = make_stratified_group_splitter(y_vec, groups=groups, n_splits=5, seed=SEED)

    fold_best_states: List[dict] = []
    fold_best_metrics: List[Dict[str, float]] = []
    fold_key_scores: List[float] = []
    fold_epochs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(ids, y_vec, groups), start=1):
        tr_ids = [ids[i] for i in tr_idx]
        va_ids = [ids[i] for i in va_idx]

        # Train dataset: allow instance sampling; Validation: use full bags (deterministic)
        train_ds = WSIBagDataset(tr_ids, preloaded_features, labels,
                                 bag_size=bag_size, replacement=replacement, preloaded=preloaded,
                                 feature_dir=feature_dir)
        val_ds_eval = WSIBagDataset(va_ids, preloaded_features, labels,
                                    bag_size=None, replacement=False, preloaded=preloaded,
                                    feature_dir=feature_dir)
        val_ds_oof  = WSIBagDataset(va_ids, preloaded_features, labels,
                                    bag_size=None, replacement=False, preloaded=preloaded,
                                    feature_dir=feature_dir,
                                    return_key=True)

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
        val_loader_eval = DataLoader(val_ds_eval, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        val_loader_oof  = DataLoader(val_ds_oof,  batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        # Fresh model + optimizer
        model = model_builder().to(device)
        
        # load or init weights
        if pretrained_state is not None:
            model.load_state_dict(pretrained_state)
            print(f"Fold {fold}: Loaded pretrained state_dict.")
        elif pretrained_model_path and os.path.exists(pretrained_model_path):
            state = torch.load(pretrained_model_path, map_location=device)
            model.load_state_dict(state)
            print(f"Fold {fold}: Loaded pretrained model from '{pretrained_model_path}'.")
        else:
            if hasattr(model, "initialize_weights"):
                model.initialize_weights()
        
        # param groups
        decay, no_decay = [], []
        for n,p in model.named_parameters():
            (no_decay if p.ndim==1 or n.endswith('.bias') else decay).append(p)
        
        
        scaler = GradScaler(enabled=(use_amp and device.type == "cuda"))

        optimizer = torch.optim.AdamW(model.parameters(), 
                                        lr=lr, 
                                        betas=(0.9, 0.999),
                                        eps=1e-8,
                                        weight_decay=l2_reg)


        sched = None
        updates_per_epoch = math.ceil(len(train_loader) / accum_steps)

        if scheduler == 'cosine':
            if step_on_epochs:
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.000005)
            else:
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates_per_epoch * epochs, eta_min=0.000005)

        # Early stopping on validation metric
        best_state, best_metric_val = None, -float("inf")
        best_metrics_dict = {}
        no_imp = 0
        best_epochs = 0
        for ep in range(1, epochs + 1):            
            train_loss = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                task=task,
                class_weights=class_weights,
                use_amp=(use_amp and device.type == "cuda"),
                scaler=scaler,
                max_grad_norm=5.0,
                scheduler=None if step_on_epochs else sched,
                accum_steps=accum_steps,
                epoch=ep - 1,  # zero-indexed for logging
                updates_per_epoch=updates_per_epoch
            )

            val_metrics, val_loss = evaluate_classifier(
                model=model,
                loader=val_loader_eval,
                device=device,
                class_weights=class_weights,
                use_amp=(use_amp and device.type == "cuda"),
            )

            # Step scheduler per epoch if not stepped per iteration
            if sched and step_on_epochs:
                sched.step()

            # choose key metric
            if key_metric and key_metric in val_metrics:
                score = val_metrics[key_metric]
            else:
                # sane defaults
                score = val_metrics["roc_auc"] if C == 1 or C == 2 else val_metrics["balanced_accuracy"]

            if score > best_metric_val:
                best_metric_val = score
                best_state = copy.deepcopy(model.state_dict())
                best_metrics_dict = dict(val_metrics)
                no_imp = 0
                best_epochs = ep
            else:
                no_imp += 1

            if (ep % 5 == 0) or (ep == epochs):
                val_mets_str = "; ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                print(f"[Fold {fold} | Epoch {ep}/{epochs}] TrainLoss={train_loss:.4f} | ValScore={score:.4f} | ValLoss={val_loss:.4f} | {val_mets_str}")

            if no_imp >= early_stopping:
                print(f"[Fold {fold}] Early stopping at epoch {ep}; Saving best model at epoch {best_epochs}\n")
                break

        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())  # fallback

        # Reload best and write OOF logits for this fold
        model.load_state_dict(best_state, strict=True)
        logits_va, ys_va, keys_va = _predict_logits_with_keys(model, val_loader_oof, device, use_amp=(use_amp and device.type == "cuda"))
        for logit_row, y_label, key in zip(logits_va, ys_va, keys_va):
            ridx = key2row[key]
            # store as float32 logits
            oof_logits[ridx] = logit_row.to(dtype=torch.float32)
            oof_fold[ridx]   = fold - 1

        fold_best_states.append(best_state)
        fold_best_metrics.append(best_metrics_dict)
        fold_key_scores.append(best_metric_val)
        fold_epochs.append(best_epochs)

        print(f"[Fold {fold}] Best Val Score: {best_metric_val:.4f}")
        for k, v in best_metrics_dict.items():
            print(f"[Fold {fold}] [{task}] {k}: {v:.4f}")
        print()

    # --- OOF metrics over all train+val samples ---
    oof_mets = _oof_metrics_from_logits(oof_y, oof_logits)

    print(f"\n=== OOF Metrics ({n_splits}-fold, stratified by label, grouped by patient) ===")
    for k, v in oof_mets.items():
        print(f"[{task}] {k}: {v:.4f}")
        
    # --- Summarize based on validation performance ---
    print("\n=== CV Summary Based on Validation Scores ===")
    metric_names = fold_best_metrics[0].keys()
    all_scores = {metric: [] for metric in metric_names}
    for fold_dict in fold_best_metrics:
        for metric, value in fold_dict.items():
            all_scores[metric].append(value)
            
    final_results = {}
    for metric, values in all_scores.items():
        mean_val = np.mean(values)
        std_val  = np.std(values, ddof=1)  # sample std (ddof=1) or ddof=0 for population
        final_results[metric] = (mean_val, std_val)

    for metric, (mean, std) in final_results.items():
        print(f"[{task}] {metric}: {mean:.4f} ± {std:.4f}")

    return {
        "oof": {
            "logits": oof_logits,        # torch.FloatTensor [N,C]
            "y": oof_y,                  # np.ndarray [N]
            "fold": oof_fold,            # np.ndarray [N]
            "metrics": oof_mets,         # dict
            "ids": ids,                  # list[str]
        },
        "fold_best_states": fold_best_states,      # list of state_dict
        "fold_best_metrics": fold_best_metrics,    # list of metric dicts per fold
        "fold_key_scores": fold_key_scores,        # list[float] key metric per fold
        "fold_epochs": fold_epochs,                  # List of epochs at which early stopping occurred for each fold
        "final_results": final_results                # dict of (mean, std) per metric across folds
    }

# =========================
# Bootstrap CI computation
# =========================
def _stratified_bootstrap_indices(
    y_true: np.ndarray,
    n: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Returns indices sampled with replacement, stratified by class, totaling exactly n.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    N = len(y_true)
    if N == 0:
        return np.array([], dtype=int)

    # Map original indices to each class
    class_indices = [np.where(y_true == c)[0] for c in np.unique(y_true)]
    classes, counts = np.unique(y_true, return_counts=True)

    # If only one class, just sample n indices uniformly with replacement
    if len(classes) <= 1:
        return rng.choice(N, size=n, replace=True)

    # Compute target per-class sizes that sum exactly to n
    props = counts / N
    target_sizes = np.floor(props * n).astype(int)
    shortfall = n - target_sizes.sum()

    # Distribute the remainder based on largest fractional parts
    if shortfall > 0:
        fracs = (props * n) - target_sizes
        distribute_indices = np.argsort(fracs)[-shortfall:]
        target_sizes[distribute_indices] += 1
    
    # Sample with replacement from each class's index pool
    bootstrap_indices = []
    for i, size in enumerate(target_sizes):
        if size > 0:
            indices_for_class = class_indices[i]
            # Ensure there are indices to sample from
            if len(indices_for_class) > 0:
                sampled = rng.choice(indices_for_class, size=size, replace=True)
                bootstrap_indices.append(sampled)
    
    if not bootstrap_indices:
        return np.array([], dtype=int)
        
    return np.concatenate(bootstrap_indices)

@torch.no_grad()
def get_bootstrap_ci(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Computes bootstrap confidence intervals for classification metrics.

    - Runs the model once to collect probs/labels, then bootstraps indices.
    - Stratified bootstrap keeps class balance stable in each resample.
    - Handles degenerate folds (single-class resamples) gracefully by NaN-ing AUC.

    Args:
        model: Trained model to evaluate. Must return (logits, log_dict?) or logits.
        loader: DataLoader yielding (inputs, labels).
        device: torch.device('cpu') or torch.device('cuda').
        use_amp: Enable automatic mixed precision on CUDA.
        n_bootstraps: Number of bootstrap samples.
        ci: Confidence level, e.g., 0.95.
        seed: RNG seed for reproducibility.

    Returns:
        dict: metric -> {mean, std, lower, upper}.
    """
    model.to(device)
    model.eval()

    # 1) Collect predictions once
    all_probs, all_labels = [], []
    amp_ctx = nullcontext()
    if use_amp and device.type == 'cuda':
        # safe AMP only on CUDA
        amp_ctx = autocast(device_type=device.type, enabled=use_amp)

    for batch in loader:
        # support (inputs, labels) or dict-like
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, labels = batch
        elif isinstance(batch, dict):
            inputs, labels = batch["inputs"], batch["labels"]
        else:
            raise ValueError("DataLoader must yield (inputs, labels) or {'inputs': ..., 'labels': ...}.")

        inputs = inputs.to(device, non_blocking=True)
        with amp_ctx:
            out = model(inputs)
            # Support (logits, log_dict) or logits
            if isinstance(out, (list, tuple)) and len(out) >= 1:
                logits = out[0]
            else:
                logits = out
            probs = torch.softmax(logits, dim=-1)

        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    if not all_labels:
        print("Warning: DataLoader is empty. Cannot compute CIs.")
        return {}

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0).ravel()

    # If model emitted a single probability vector (N,), convert to (N,2)
    if y_prob.ndim == 1:
        y_prob = np.vstack([1 - y_prob, y_prob]).T

    if y_true.size == 0:
        print("Warning: No labels found. Cannot compute CIs.")
        return {}

    rng = np.random.default_rng(seed)
    N = y_true.shape[0]
    metric_buckets = defaultdict(list)

    # 2) Bootstrapping
    for _ in range(n_bootstraps):
        idx = _stratified_bootstrap_indices(y_true, N, rng)
        # In rare cases (pathological tiny N), idx length might drift; guard anyway
        if idx.size == 0:
            continue
        boot_metrics = aggregate_metrics(y_true[idx], y_prob[idx])
        for name, value in boot_metrics.items():
            if np.isfinite(value):
                metric_buckets[name].append(float(value))

    # 3) Summarize
    alpha = (1.0 - ci) / 2.0
    ci_results: Dict[str, Dict[str, float]] = {}
    for name, values in metric_buckets.items():
        if not values:
            ci_results[name] = {'mean': np.nan, 'std': np.nan, 'lower': np.nan, 'upper': np.nan}
        else:
            arr = np.array(values, dtype=float)
            ci_results[name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                'lower': float(np.percentile(arr, 100 * alpha)),
                'upper': float(np.percentile(arr, 100 * (1 - alpha))),
            }

    return ci_results

def run_experiment(
    model_builder: Callable[[], torch.nn.Module],
    preloaded_features: Dict[str, torch.Tensor],
    train_labels: Dict[str, torch.Tensor | int],
    test_labels: Dict[str, torch.Tensor | int] | None,
    patient_id_mapping: Dict[str, str], # always for train_labels
    preloaded = True,
    feature_dir=None,  # New: directory for preloaded features
    task = "classification",  # "classification" | "survival"
    key_metric = None,
    n_splits=5,
    device='cpu',
    epochs=60,
    lr=3e-4,
    l2_reg=1e-3,
    bag_size=None,
    replacement=True,
    scheduler="cosine",
    early_stopping=10,
    class_weights=None,
    pretrained_state=None,      # New: load pretrained weights if provided
    pretrained_model_path=None,  # New: path to pretrained model
    model_saving_path=None,
    precision=16,
    batch_size=1,
    combine = "logits",  # "logits" | "probs"
    final_strategy="refit_full",      # "ensemble" | "refit_full" | "best_fold"
    topk_for_ensemble=None,         # None => use all folds; else use top-k by val score
    warmup_epochs=0,
    step_on_epochs=False,
    accum_steps=1,
):
    device = _to_device(device)
    print(f"\n=== Running Experiment on Device: {device} ===")
    
    if len(train_labels) != len(patient_id_mapping):
        print("Warning: train_labels size does not match patient_id_mapping size. Check data consistency.")
    
    if test_labels is not None and len(test_labels) > 0:
        pass
    else:
        ids = list(train_labels.keys())
        y_vec = np.array([int(train_labels[k].item() if torch.is_tensor(train_labels[k]) else train_labels[k]) for k in ids], dtype=int)
        groups = np.array([patient_id_mapping[k] for k in ids])
        
        print("No test set provided. Splitting train set into train and hold-out test set.")
        skf = make_stratified_group_splitter(y_vec, groups=groups, n_splits=5, seed=SEED)
        tr_idx, ho_idx = next(skf.split(ids, y_vec, groups=groups))
        
        test_labels = {k: train_labels[k] for k in np.array(ids)[ho_idx].tolist()}
        train_labels = {k: train_labels[k] for k in np.array(ids)[tr_idx].tolist()}
        
    patient_id_mapping_train = {k: patient_id_mapping[k] for k in list(train_labels.keys())}

    print(f"Train set size: {len(train_labels)}")
    print(f"Test set size: {len(test_labels)}")
    use_amp = (precision == 16 and device.type == 'cuda')

    # Run OOF CV
    cv_results = run_oof_cv_kfold(
        model_builder=model_builder,
        preloaded_features=preloaded_features,
        labels=train_labels,
        patient_id_mapping=patient_id_mapping_train,
        task=task,
        key_metric=key_metric,
        n_splits=n_splits,
        device=device,
        epochs=epochs,
        lr=lr,
        l2_reg=l2_reg,
        bag_size=bag_size,
        replacement=replacement,
        scheduler=scheduler,
        early_stopping=early_stopping,
        class_weights=class_weights,
        pretrained_state=pretrained_state,
        pretrained_model_path=pretrained_model_path,
        batch_size=batch_size,
        use_amp=use_amp,
        warmup_epochs=warmup_epochs,
        accum_steps=accum_steps,
        step_on_epochs=step_on_epochs,
        preloaded=preloaded,
        feature_dir=feature_dir,  # New: pass feature directory if preloaded
    )
    
    # --- Final Model Evaluation ---
    print("\n=== Final Model Evaluation ===")
    print(f"Final Strategy: {final_strategy}")
    final_model = None

    # Step 1: Get the final model based on the chosen strategy
    if final_strategy == "ensemble":
        final_model = ensemble_models(
            base_model=model_builder(),
            fold_states=cv_results["fold_best_states"],
            device=device,
            topk=topk_for_ensemble,
            fold_scores=cv_results["fold_key_metrics"],
            combine=combine
        )
    elif final_strategy == "refit_full":
        final_model = refit_to_all(
            model_builder=model_builder, features=preloaded_features, labels=train_labels,
            cv_results=cv_results, task=task, device=device, epochs=epochs, lr=lr, l2_reg=l2_reg,
            scheduler=scheduler, class_weights=class_weights, batch_size=batch_size,
            bag_size=bag_size, replacement=replacement, use_amp=use_amp, pretrained_state=pretrained_state,
            pretrained_model_path=pretrained_model_path, warmup_epochs=warmup_epochs,
            step_on_epochs=step_on_epochs, accum_steps=accum_steps, preloaded=preloaded,
            feature_dir=feature_dir
        )
    else:  # Default to using the single best model from CV
        best_fold_idx = np.argmax(cv_results["fold_key_metrics"])
        best_state = cv_results["fold_best_states"][best_fold_idx]
        print(f"\n--- Using best model from Fold {best_fold_idx + 1} for final evaluation ---")
        final_model = model_builder().to(device)
        final_model.load_state_dict(best_state)
        
    
    # Step 2: Evaluate the final model on the test set
    test_keys = list(test_labels.keys())
    print("\n=== Evaluating on Hold-Out Test Set ===")
    test_ds = WSIBagDataset(test_keys, 
                            preloaded_features, 
                            test_labels, 
                            bag_size=None, 
                            replacement=False,
                            preloaded=preloaded,
                            feature_dir=feature_dir)
    test_loader = DataLoader(test_ds,
                             batch_size=1, 
                             shuffle=False, 
                             num_workers=4, 
                             pin_memory=True)
    
    # point estimate and confidence intervals
    test_metrics, _ = evaluate_classifier(
        final_model, test_loader, device, class_weights, use_amp=use_amp
    )
    
    ci_dict = get_bootstrap_ci(
        final_model, test_loader, device, use_amp=use_amp, n_bootstraps=5000, ci=0.95, seed=SEED
    )
    
    # ensure keys are matched
    test_keys = list(test_metrics.keys())
    ci_keys = list(ci_dict.keys())
    if set(test_keys) != set(ci_keys):
        print(f"Warning: Mismatched keys between test metrics and CI: {set(test_keys) ^ set(ci_keys)}")
    
    # Print final test metrics
    print("\n=== Point Estimate Metrics ===")
    for k, v in test_metrics.items():
        print(f"[Point Estimate] {k}: {v:.4f}")

    # Print confidence intervals
    print("\n=== Confidence Intervals ===")
    for k, v in ci_dict.items():
        print(f"[Confidence Interval] {k}: {v['mean']:.4f} ± {v['std']:.4f} [{v['lower']:.4f}, {v['upper']:.4f}]")

    # Save the final model if a path is provided
    if model_saving_path:
        os.makedirs(os.path.dirname(model_saving_path), exist_ok=True)
        torch.save(final_model.state_dict(), model_saving_path)
        print(f"Final model saved to {model_saving_path}")

    return test_metrics, ci_dict, final_model, cv_results