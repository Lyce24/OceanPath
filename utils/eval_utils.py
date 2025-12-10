from __future__ import annotations

import numpy as np
from typing import Dict, List
from contextlib import nullcontext

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, log_loss, roc_auc_score, roc_curve,
    confusion_matrix
)
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
from typing import Dict, List, Tuple
from collections import defaultdict
import torch.nn.functional as F

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

def aggregate_metrics(y_true: np.ndarray, y_prob: np.ndarray, optimal_threshold: float | None = None) -> Dict[str, float]:
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
    if optimal_threshold is not None:
        y_pred = (y_prob_safe[:, 1] >= optimal_threshold).astype(int)
    else:
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
    optimal_threshold = None
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
                if model.encoder_type == "CLAM" or model.encoder_type == "DSMIL":
                    outputs, log_dict = model(inputs, return_raw_attention=True, labels=labels)
                    loss = loss_fn(outputs, labels)

                    if model.encoder_type == "CLAM" and 'instance_loss' in log_dict and log_dict['instance_loss'] != -1:
                        loss = 0.7 * loss + 0.3 * log_dict['instance_loss']
                        
                    elif model.encoder_type == "DSMIL" and 'instance_loss' in log_dict and log_dict['instance_loss'] != -1:
                        loss = 0.5 * loss + 0.5 * log_dict['instance_loss']

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
    metrics = aggregate_metrics(y_true, y_prob, optimal_threshold=optimal_threshold)

    return metrics, avg_loss

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
    optimal_threshold: float | None = None
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
        boot_metrics = aggregate_metrics(y_true[idx], y_prob[idx], optimal_threshold=optimal_threshold)
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

def oof_positive_probs(oof_logits): # torch.Tensor [N,1] or [N,2]
    if oof_logits.ndim == 1 or oof_logits.size(1) == 1:
        return torch.sigmoid(oof_logits.squeeze(-1)).cpu().numpy()
    else:
        return F.softmax(oof_logits, dim=1)[:, 1].cpu().numpy()

def tune_threshold_youden(y_true, p1):
    fpr, tpr, thr = roc_curve(y_true, p1)
    j = tpr - fpr
    i = int(np.argmax(j))  # roc_curve returns len(thr)=len(tpr)=len(fpr); thr are decision thresholds on p1
    return float(thr[i]), float(j[i])

def op_metrics(y_true, p1, t=0.5):
    y_pred = (p1 >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn + 1e-12)   # recall
    spec = tn / (tn + fp + 1e-12)
    ppv  = precision_score(y_true, y_pred, zero_division=0)
    npv  = tn / (tn + fn + 1e-12)
    return dict(threshold=t, sensitivity=sens, specificity=spec, ppv=ppv, npv=npv)

# --- helpers ---
def logits_to_p1(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns positive-class probability from logits.
    Supports shapes [B], [B,1], or [B,2].
    """
    if logits.ndim == 1 or logits.size(-1) == 1:
        return torch.sigmoid(logits.squeeze(-1))
    else:
        return torch.softmax(logits, dim=-1)[:, 1]

def probs_vector_from_p1(p1: float) -> np.ndarray:
    return np.array([1.0 - p1, p1], dtype=np.float32)
