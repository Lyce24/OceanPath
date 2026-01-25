from __future__ import annotations

import numpy as np
from typing import Dict, List
from contextlib import nullcontext

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix
)
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import torch.nn.functional as F
import warnings

# =============================================================================
# Evaluation Functions
# =============================================================================
def logits_to_probs(logits: torch.Tensor) -> np.ndarray:
    """
    Convert logits to probability matrix.
    
    Args:
        logits: Shape [N], [N, 1], or [N, C]
        
    Returns:
        Probabilities [N, C] where C >= 2
    """
    logits = logits.detach().cpu().float()
    
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)
    
    if logits.size(-1) == 1:
        p = torch.sigmoid(logits.squeeze(-1))
        probs = torch.stack([1 - p, p], dim=-1)
    else:
        probs = torch.softmax(logits, dim=-1)
    
    return probs.numpy()


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Compute classification metrics with proper error handling."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob)
    
    if y_true.size == 0 or y_prob.size == 0:
        return {}
    
    if y_prob.ndim != 2:
        raise ValueError(f"y_prob must be 2D, got shape {y_prob.shape}")
    
    N, C = y_prob.shape
    if N != len(y_true):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_prob={N}")
    
    # Numerical stability
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
    row_sums = y_prob.sum(axis=1, keepdims=True)
    y_prob = y_prob / np.where(row_sums > 0, row_sums, 1.0)
    
    # Predictions
    if threshold is not None and C == 2:
        y_pred = (y_prob[:, 1] >= threshold).astype(int)
    else:
        y_pred = np.argmax(y_prob, axis=1)
    
    avg = "binary" if C == 2 else "macro"
    
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }
    
    # ROC-AUC with proper exception handling
    try:
        if len(np.unique(y_true)) < 2:
            metrics["roc_auc"] = float("nan")
        elif C == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["roc_auc"] = float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            ))
    except ValueError as e:
        warnings.warn(f"ROC-AUC computation failed: {e}")
        metrics["roc_auc"] = float("nan")
    
    return metrics


def tune_threshold_youden(y_true: np.ndarray, p_pos: np.ndarray) -> Tuple[float, float]:
    """Find optimal threshold using Youden's J statistic."""
    y_true = np.asarray(y_true).ravel()
    p_pos = np.asarray(p_pos).ravel()
    
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    
    fpr, tpr, thresholds = roc_curve(y_true, p_pos)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    
    # Handle edge cases where threshold is 0 or > 1
    threshold = float(np.clip(thresholds[best_idx], 0.01, 0.99))
    return threshold, float(j_scores[best_idx])


def operating_point_metrics(
    y_true: np.ndarray,
    p_pos: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute binary operating point metrics."""
    y_true = np.asarray(y_true).ravel()
    p_pos = np.asarray(p_pos).ravel()
    y_pred = (p_pos >= threshold).astype(int)
    
    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    if len(classes) > 2:
        warnings.warn("operating_point_metrics is for binary classification only")
        return {"threshold": float(threshold)}
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return {
            "threshold": float(threshold),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "ppv": float("nan"),
            "npv": float("nan"),
        }
    
    tn, fp, fn, tp = cm.ravel()
    eps = 1e-10
    
    return {
        "threshold": float(threshold),
        "sensitivity": float(tp / (tp + fn + eps)),
        "specificity": float(tn / (tn + fp + eps)),
        "ppv": float(tp / (tp + fp + eps)),
        "npv": float(tn / (tn + fn + eps)),
    }


def _stratified_bootstrap_indices(
    y_true: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate stratified bootstrap indices to maintain class balance."""
    classes, counts = np.unique(y_true, return_counts=True)
    
    if len(classes) <= 1:
        return rng.choice(len(y_true), size=n_samples, replace=True)
    
    # Target samples per class
    props = counts / len(y_true)
    target_sizes = np.round(props * n_samples).astype(int)
    
    # Adjust to ensure exact n_samples
    diff = n_samples - target_sizes.sum()
    if diff != 0:
        idx = rng.choice(len(classes))
        target_sizes[idx] += diff
    
    # Sample from each class
    indices = []
    for cls, size in zip(classes, target_sizes):
        cls_indices = np.where(y_true == cls)[0]
        if size > 0 and len(cls_indices) > 0:
            sampled = rng.choice(cls_indices, size=size, replace=True)
            indices.append(sampled)
    
    return np.concatenate(indices) if indices else np.array([], dtype=int)


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    threshold: Optional[float] = None,
    seed: int = 42,
    stratified: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap confidence intervals with stratified sampling."""
    y_true = np.asarray(y_true).ravel()
    rng = np.random.default_rng(seed)
    N = len(y_true)
    
    metric_samples: Dict[str, List[float]] = defaultdict(list)
    
    for _ in range(n_bootstraps):
        if stratified:
            idx = _stratified_bootstrap_indices(y_true, N, rng)
        else:
            idx = rng.choice(N, size=N, replace=True)
        
        if len(idx) == 0:
            continue
            
        m = compute_metrics(y_true[idx], y_prob[idx], threshold)
        for k, v in m.items():
            if np.isfinite(v):
                metric_samples[k].append(v)
    
    alpha = (1 - ci) / 2
    results: Dict[str, Dict[str, float]] = {}
    
    for name, vals in metric_samples.items():
        if vals:
            arr = np.array(vals)
            results[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "lower": float(np.percentile(arr, 100 * alpha)),
                "upper": float(np.percentile(arr, 100 * (1 - alpha))),
            }
        else:
            results[name] = {"mean": np.nan, "std": np.nan, "lower": np.nan, "upper": np.nan}
    
    return results

def _normalize_y(y: torch.Tensor) -> torch.Tensor:
    """Normalize labels to 1D long tensor."""
    y = torch.as_tensor(y)
    if y.ndim == 2:
        if y.size(-1) > 1:      # one-hot / probs
            y = y.argmax(dim=-1)
        else:                   # [B,1]
            y = y.squeeze(-1)
    return y.long().view(-1)


@torch.inference_mode()
def collect_predictions(model, loader, device, use_amp=True):
    """Collect predictions from a model on a dataloader."""
    model.eval()
    all_logits, all_labels = [], []
    amp_enabled = use_amp and device.type == "cuda"

    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)

        with autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out

        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)

        all_logits.append(logits.detach().float().cpu())
        all_labels.append(_normalize_y(y).cpu())

    return torch.cat(all_logits, 0), torch.cat(all_labels, 0)