import pandas as pd
import torch
import os

data_combined = pd.read_excel('./data/combined_labels_with_patient_id.xlsx')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("Number of samples:", len(data_combined))

from utils.train_utils import load_features_mmap, prepare_labels
from models.wsi_model import WSIModel
from scripts.experiment import run_experiment

major_dir = "../combined_features/wsi_processed_no_penmarks"

def run_one_experiment(task, 
                        mil,
                        encoder,
                        D,
                        preloaded_features,
                        training_cohorts, 
                        data_combined, 
                        lr_range,
                        wd_range,
                        device, 
                        precision,
                        save_dir=None,
                        verbose= True,
                        preloaded=True,
                        feature_dir=None):
    verbose = verbose
    precision = precision

    _, train_labels, train_class_weights, n_classes, patient_id_mapping = prepare_labels(data_combined, "De-ID", task, verbose, cohorts=training_cohorts, patient_id_col="Patient ID")
    _, test_labels, _, _, _ = prepare_labels(data_combined, "De-ID", task, verbose, cohorts=["RIH"], patient_id_col=None)
    train_class_weights = train_class_weights.to(device) if train_class_weights is not None else None

    mil_type = mil
    if mil_type == "ABMIL":
        mil_attrs = {
            "attn_dim": 384,
            "gate": True
        }
    elif mil_type == "TransMIL":
        mil_attrs = {
            "num_attention_layers": 2,
            "num_heads": 4
        }
    elif mil_type == "WIKGMIL":
        mil_attrs = {
            "agg_type": "bi-interaction",
            "pool": "attn",
            "topk": 4
        }
    elif mil_type == "DSMIL":
        mil_attrs = {
            "attn_dim": 384,
            "dropout_v": 0.0
        }
    elif mil_type == "CLAM":
        mil_attrs = {
            "attention_dim": 384,
            "gate": True,
            "k_sample": 8,
            "subtyping": False,
            "instance_loss_fn": "svm",
            "bag_weight": 0.7
        }
    else:
        mil_attrs = {}

    for lr in lr_range:
        for wd in wd_range:
            print(f"Running experiment for task: {task} with MIL: {mil_type}, lr: {lr}, wd: {wd}")
            n_splits: int = 5
            epochs: int = 50
            early_stopping: int = 10
            step_on_epochs  = True
            accum_steps     = 1
            
            test_metrics, ci_dict, _, cv_results = run_experiment(
                    model_builder=lambda: WSIModel(
                        input_feature_dim=D,  # Example input feature dimension
                        n_classes=n_classes,
                        encoder_type=mil_type,
                        head_dropout=0.25,
                        head_dim=512,
                        num_fc_layers=1,
                        hidden_dim=128,
                        ds_dropout=0.3,
                        simple_mlp=False,
                        freeze_encoder=False,
                        encoder_attrs=mil_attrs
                    ),
                    preloaded_features=preloaded_features,
                    train_labels=train_labels,
                    test_labels=test_labels,
                    patient_id_mapping=patient_id_mapping,
                    device=device,
                    epochs=epochs,
                    task=task,
                    lr=lr,
                    l2_reg=wd,
                    early_stopping=early_stopping,
                    bag_size=None,
                    replacement=False,
                    class_weights=train_class_weights,
                    key_metric="roc_auc" if n_classes == 2 else "balanced_accuracy",
                    precision=precision,
                    warmup_epochs=0,
                    accum_steps=accum_steps,
                    step_on_epochs=step_on_epochs,
                    preloaded=preloaded,
                    feature_dir=feature_dir
                )

            save_path = f"{save_dir}/{task}/{encoder}/mil{mil_type}_lr{lr}_wd{wd}_cohort{'-'.join(training_cohorts)}_v2.txt"
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            with open(save_path, "w") as f:
                f.write(f"Task: {task}\n")
                f.write(f"Encoder: {encoder}\n")
                f.write(f"MIL Type: {mil_type}\n")
                f.write(f"Training Cohorts: {', '.join(training_cohorts)}\n")
                f.write(f"Learning Rate: {lr}\n")
                f.write(f"Weight Decay: {wd}\n")
                f.write(f"Number of Samples: {len(train_labels)}\n")
                f.write(f"Number of Classes: {n_classes}\n")

                f.write(f"\n=== OOF Metrics ({n_splits}-fold, stratified by label, grouped by patient) ===\n")
                for k, v in cv_results["oof"]["metrics"].items():
                    f.write(f"[OOF] {k}: {v:.4f}\n")

                f.write("\n=== CV Summary Based on Validation Scores ===\n")
                for metric, (mean, std) in cv_results["final_results"].items():
                    f.write(f"[CV] {metric}: {mean:.4f} ± {std:.4f}\n")

                f.write("\n=== Point Estimate Metrics ===\n")
                for k, v in test_metrics.items():
                    f.write(f"[Point Estimate] {k}: {v:.4f}\n")

                f.write("\n=== Confidence Intervals ===\n")
                for k, v in ci_dict.items():
                    f.write(f"[Confidence Interval] {k}: {v['mean']:.4f} ± {v['std']:.4f} [{v['lower']:.4f}, {v['upper']:.4f}]\n")

            print(f"Results saved to {save_path}")

if __name__ == "__main__":
    tasks = ["Binary Stage N"] # tasks = ["Binary Stage N", "Binary Stage T", "Binary TNM Stage", "died_within_5_years", "MSI", "BRAF"]

    training_cohorts = [["SR"]]
    lr_range = [2e-04] # [1e-04, 2e-04]
    wd_range = [5e-05, 5e-04, 5e-03, 5e-02] # [5e-05, 1e-04, 5e-04, 1e-03, 5e-03, 1e-02]
    mils = ["ABMIL"]
    encoders = ["uni_v1"]

    for encoder in encoders:
        feature_dir = f"{major_dir}/features_{encoder}"
        preloaded_features, D, preloaded = load_features_mmap(data_combined, feature_dir=feature_dir, id_col="De-ID", verbose=True, precision=16)
        for training_cohort in training_cohorts:
            for task in tasks:
                for mil in mils:
                    print(f"Running experiment for task: {task} with training cohorts: {training_cohort} (MIL: {mil}; Encoder: {encoder})")
                    run_one_experiment(task = task,
                                        mil=mil,
                                        encoder=encoder,
                                        D = D,
                                        preloaded_features=preloaded_features,
                                        training_cohorts=training_cohort,
                                        data_combined=data_combined,
                                        lr_range=lr_range,
                                        wd_range=wd_range,
                                        device=device,
                                        precision=16,
                                        save_dir="./results/test",
                                        verbose=True,
                                        preloaded=preloaded,
                                        feature_dir=feature_dir)