from __future__ import annotations

from utils.train_utils import (
    _to_device, adamw_param_groups, train_one_epoch
)

from utils.eval_utils import (
    evaluate_classifier
)

import math
from torch.amp import GradScaler
from timm.scheduler import create_scheduler_v2
import copy
from typing import Dict, Callable
import torch
import os
from torch.utils.data import DataLoader
import numpy as np

# -----------------------------
# Main: k-fold OOF CV
# -----------------------------
def train(
    model_builder: Callable[[], torch.nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    task = "classification",
    key_metric = None,
    device='cpu',
    epochs=60,
    lr=3e-4,
    l2_reg=1e-3,
    scheduler="cosine",
    early_stopping=10,
    class_weights=None,
    pretrained_state=None,      # New: load pretrained weights if provided
    pretrained_model_path=None,  # New: path to pretrained model
    warmup_epochs=0,
    use_amp=True,
    accum_steps=1,
    step_on_epochs=True,
    mode="normal", # normal | refit
    cv_results=None
) -> Dict:
    
    device = _to_device(device)

    # Fresh model + optimizer
    model = model_builder().to(device)
    C = model.n_classes
    
    if mode == "refit":
        assert cv_results is not None and "fold_epochs" in cv_results, "cv_results with fold_epochs must be provided for refit mode."
        assert val_loader is None, "Validation loader must be None in refit mode."
        if cv_results["fold_epochs"]:
                # num_epochs = int(np.ceil(np.median(cv_results["fold_epochs"])))
                # print(f"Training for {num_epochs} epochs (based on average from CV).")
                p75 = np.percentile(cv_results["fold_epochs"], 75)
                epochs = int(np.ceil(p75))
                print(f"Training for {epochs} epochs (based on 75th percentile from CV).")
        else:
            # Fallback to the max epochs if early stopping didn't trigger in any fold
            epochs = epochs
            print(f"CV did not trigger early stopping. Training for the full {epochs} epochs.")
    else:
        assert val_loader is not None, "Validation loader must be provided for normal training mode."
        assert epochs > 0, "Number of epochs must be greater than 0 in normal training mode."
        assert cv_results is None, "cv_results should be None in normal training mode."
        
    # load or init weights
    if pretrained_state is not None:
        model.load_state_dict(pretrained_state)
        print(f"Loaded pretrained state_dict.")
    elif pretrained_model_path and os.path.exists(pretrained_model_path):
        state = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded pretrained model from '{pretrained_model_path}'.")
    else:
        if hasattr(model, "initialize_weights"):
            model.initialize_weights()
        
    scaler = GradScaler(enabled=(use_amp and device.type == "cuda"))

    optimizer = torch.optim.AdamW(adamw_param_groups(model, l2_reg),
                                    lr=lr, 
                                    betas=(0.9, 0.999),
                                    eps=1e-8)

    sched = None
    updates_per_epoch = math.ceil(len(train_loader) / accum_steps)

    if scheduler == 'cosine':
        sched,_ = create_scheduler_v2(
            optimizer=optimizer,
            sched="cosine",
            num_epochs=epochs,
            min_lr=1e-6,
            warmup_epochs=warmup_epochs,
            step_on_epochs=step_on_epochs,
            updates_per_epoch=updates_per_epoch
        )

    # Early stopping on validation metric
    best_state, best_metric_val = None, -float("inf")
    no_imp = 0
    best_epochs = 0
    best_metrics_dict = {}
    
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
        if val_loader is not None:
            val_metrics, val_loss = evaluate_classifier(
                model=model,
                loader=val_loader,
                device=device,
                class_weights=class_weights,
                use_amp=(use_amp and device.type == "cuda"),
            )
            
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
                print(f"[Epoch {ep}/{epochs}] TrainLoss={train_loss:.4f} | ValScore={score:.4f} | ValLoss={val_loss:.4f} | {val_mets_str}")

            if no_imp >= early_stopping:
                print(f"[Early stopping at epoch {ep}; Saving best model at epoch {best_epochs}\n")
                break

        # Step scheduler per epoch if not stepped per iteration
        if sched and step_on_epochs:
            sched.step(epoch=ep)

    if val_loader is not None: # normal mode
        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())  # fallback
        
        return {
            "best_state": best_state,
            "best_metrics": best_metrics_dict,
            "best_epoch": best_epochs,
            "best_metric_val": best_metric_val,
        }
    else: # refit mode => only best state needed
        best_state = copy.deepcopy(model.state_dict())
        return {
            "best_state": best_state,
        }