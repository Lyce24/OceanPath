from __future__ import annotations

from typing import Dict, List, Callable

import torch
import os
import numpy as np

from utils.train_utils import (
    _to_device, _worker_init_fn,
    WSIBagDataset, make_stratified_group_splitter, SEED
)
from utils.eval_utils import (
     _predict_logits_with_keys, _oof_metrics_from_logits
)
from scripts.train import train

from torch.utils.data import DataLoader

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
    skf = make_stratified_group_splitter(y_vec, groups=groups, n_splits=n_splits, seed=SEED)

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

        train_loader = DataLoader(train_ds, 
                                  batch_size=batch_size if (bag_size is not None and replacement == True) else 1,
                                  shuffle=True, num_workers=4, pin_memory=True,
                                   worker_init_fn=_worker_init_fn, generator=torch.Generator().manual_seed(SEED))
        val_loader_eval = DataLoader(val_ds_eval, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
                                      worker_init_fn=_worker_init_fn, generator=torch.Generator().manual_seed(SEED))
        val_loader_oof  = DataLoader(val_ds_oof,  batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
                                      worker_init_fn=_worker_init_fn, generator=torch.Generator().manual_seed(SEED))

        print(f"\n=== Fold {fold}/{n_splits} ===")
        train_results = train(
            model_builder=model_builder,
            train_loader=train_loader,
            val_loader=val_loader_eval,
            task=task,
            key_metric=key_metric,
            device=device,
            epochs=epochs,
            lr=lr,
            l2_reg=l2_reg,
            scheduler=scheduler,
            early_stopping=early_stopping,
            class_weights=class_weights,
            pretrained_state=pretrained_state,      # New: load pretrained weights if provided
            pretrained_model_path=pretrained_model_path,  # New: path to pretrained model
            warmup_epochs=warmup_epochs,
            use_amp=use_amp,
            accum_steps=accum_steps,
            step_on_epochs=step_on_epochs,
            mode="normal", # normal | refit
            cv_results=None
        )
        
        model = model_builder().to(device)
        best_state = train_results['best_state']
        best_metrics_dict = train_results['best_metrics']
        best_metric_val = train_results['best_metric_val']
        best_epochs = train_results['best_epoch']

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