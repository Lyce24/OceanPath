from utils.train_utils import (
    _to_device, _worker_init_fn,
    WSIBagDataset, make_stratified_group_splitter, SEED, set_global_seed
)

from utils.eval_utils import (
     oof_positive_probs, tune_threshold_youden, op_metrics,
     evaluate_classifier, get_bootstrap_ci
)
from scripts.train import train
from scripts.oof import run_oof_cv_kfold

import numpy as np
from typing import Dict
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Callable

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
    final_strategy="refit_full",      # | "refit_full" | "best_fold"
    warmup_epochs=0,
    step_on_epochs=False,
    accum_steps=1,
    seed = SEED,
    optimal_threshold=True
):
    # Set global seed for reproducibility
    set_global_seed(seed)
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

    # --- Post-Processing ---
    oof = cv_results["oof"]
    C = oof["logits"].shape[1]
    p1 = oof_positive_probs(oof["logits"])
    best_thr, best_j = tune_threshold_youden(oof["y"], p1)
    print(f"\n=== Optimal Threshold (Youden's J): {best_thr} ===")
    
    if not optimal_threshold:
        best_thr = 0.5  # Default threshold if not tuning
    
    op_dict = op_metrics(oof["y"], p1, t=best_thr)
    for k, v in op_dict.items():
        print(f"[Optimal Threshold] {k}: {v:.4f}")

    # --- Final Model Evaluation ---
    print("\n=== Final Model Evaluation ===")
    print(f"Final Strategy: {final_strategy}")
    
    final_model = model_builder().to(device)
    
    # Step 1: Get the final model based on the chosen strategy
    if final_strategy == "refit_full":
        full_training_dataset = WSIBagDataset(
            list(train_labels.keys()),
            preloaded_features,
            train_labels,
            bag_size=bag_size,
            replacement=replacement,
            preloaded=preloaded,
            feature_dir=feature_dir
        )
        full_training_dataloader = DataLoader(
            full_training_dataset,
            batch_size=batch_size if (bag_size is not None and replacement == True) else 1,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=_worker_init_fn,
            generator=torch.Generator().manual_seed(SEED)
        )
        refit_results = train(
            model_builder=model_builder, train_loader=full_training_dataloader,
            val_loader=None, cv_results=cv_results, task=task, key_metric=key_metric,
            device=device, epochs=epochs, lr=lr, l2_reg=l2_reg,
            scheduler=scheduler, class_weights=class_weights, use_amp=use_amp, pretrained_state=pretrained_state,
            pretrained_model_path=pretrained_model_path, warmup_epochs=warmup_epochs,
            step_on_epochs=step_on_epochs, accum_steps=accum_steps, mode="refit"
        )
        final_model.load_state_dict(refit_results["best_state"])
    else:  # Default to using the single best model from CV
        best_fold_idx = np.argmax(cv_results["fold_key_scores"])
        best_state = cv_results["fold_best_states"][best_fold_idx]
        print(f"\n--- Using best model from Fold {best_fold_idx + 1} for final evaluation ---")
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
                             pin_memory=True,
                             worker_init_fn=_worker_init_fn, 
                             generator=torch.Generator().manual_seed(SEED))
    
    # point estimate and confidence intervals
    test_metrics, _ = evaluate_classifier(
        final_model, test_loader, device, class_weights, use_amp=use_amp, 
        optimal_threshold=best_thr
    )

    ci_dict = get_bootstrap_ci(
        final_model, test_loader, device, use_amp=use_amp, n_bootstraps=5000, ci=0.95, seed=SEED, 
        optimal_threshold=best_thr
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