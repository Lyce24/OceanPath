import pandas as pd
import torch
import os
from utils.train_utils import load_features_mmap, prepare_labels, logits_to_p1, probs_vector_from_p1, oof_positive_probs, tune_threshold_youden, op_metrics
from models.wsi_model import WSIModel
from scripts.experiment import run_experiment
from torch.amp import autocast

df = pd.read_excel('./data/AI Bladder Cases_V25.xlsx', sheet_name='Test Samples')
available_wsi = []

for file in os.listdir('./bladder_data'):
    if file.endswith('.tiff'):
        available_wsi.append(file.split('.')[0])

# only keep the WSI IDs that are available in the directory
df['De ID'] = df['De ID'].apply(lambda x: x.split('.')[0])
df = df[df['De ID'].isin(available_wsi)]

df['Binary WHO 2022'] = df['WHO 2022'].apply(lambda x: 0 if x == 'low' else 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("Number of samples:", len(df))

major_dir = "./bladder_combined_features/"

def run_one_experiment(task, 
                        mil,
                        encoder,
                        D,
                        preloaded_features,
                        train_df,
                        test_df,
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

    if train_df is not None and test_df is not None:
        _, train_labels, train_class_weights, n_classes, patient_id_mapping = prepare_labels(train_df, "De ID", task, verbose, cohorts=None, patient_id_col="Case")
        _, test_labels, _, _, _ = prepare_labels(test_df, "De ID", task, verbose, cohorts=None, patient_id_col="Case")
    elif train_df is not None:
        _, train_labels = prepare_labels(train_df, "De ID", task, verbose, cohorts=None, patient_id_col="Case")
        test_labels = None

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
            "dropout_v": 0.0,
            "layernorm": True
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
            epochs: int = 40
            early_stopping: int = 8
            step_on_epochs  = True
            accum_steps     = 1
            
            test_metrics, ci_dict, final_model, cv_results = run_experiment(
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
                    feature_dir=feature_dir,
                    optimal_threshold=True
                )

            oof = cv_results["oof"]
            y = oof["y"]
            p1 = oof_positive_probs(oof["logits"])

            t_j,   _ = tune_threshold_youden(y, p1)

            op_dict = op_metrics(y, p1, t=t_j)

            G2_df = test_df[test_df["WHO 1973"] == 2].copy()
            G2_features = {k: v for k, v in preloaded_features.items() if k in G2_df["De ID"].values}
            G2_labels = {k: test_labels[k] for k in G2_df["De ID"].values if k in test_labels}

            print(f"Number of G2 samples: {len(G2_features)}")

            return_dict = {}
            wrong_predictions = []

            with torch.no_grad():
                final_model.to(device).eval()

                for key in list(G2_features.keys()):
                    feats = G2_features[key].to(device, non_blocking=True).unsqueeze(0)  # [1, M, D]
                    labs  = G2_labels[key].to(device, non_blocking=True).unsqueeze(0)    # [1]

                    with autocast(device_type=device.type, enabled=True):
                        logits, _ = final_model(feats)  # logits shape: [1, 1] or [1, 2]

                    # --- USE t_j HERE ---
                    p1 = logits_to_p1(logits).item()              # scalar positive prob
                    pred = int(p1 >= t_j)                         # thresholded decision
                    prob_vec = probs_vector_from_p1(p1)           # [2] for metrics

                    true_label = int(labs.item())

                    if pred != true_label:
                        wrong_predictions.append(key)
                        print(f"Slide ID: {key} | True Label: {true_label} | Predicted: {pred} | "
                              f"Probabilities: {prob_vec}")

                    return_dict[key] = {
                        "true_label": true_label,
                        "predicted": pred,
                        "probabilities": prob_vec
                    }
                    
            print(f"Number of wrong predictions: {len(wrong_predictions)}")

            if save_dir:
                save_path = f"{save_dir}/{task}/{encoder}/mil{mil_type}_lr{lr}_wd{wd}_blca.txt"
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))

                with open(save_path, "w") as f:
                    f.write(f"Task: {task}\n")
                    f.write(f"Encoder: {encoder}\n")
                    f.write(f"MIL Type: {mil_type}\n")
                    f.write(f"Learning Rate: {lr}\n")
                    f.write(f"Weight Decay: {wd}\n")
                    f.write(f"Number of Samples: {len(train_labels)}\n")
                    f.write(f"Number of Classes: {n_classes}\n")

                    f.write(f"\n=== OOF Metrics ({n_splits}-fold, stratified by label, grouped by patient) ===\n")
                    for k, v in cv_results["oof"]["metrics"].items():
                        f.write(f"[OOF] {k}: {v:.4f}\n")

                    f.write(f"\n=== Optimal Threshold Metrics ===\n")
                    for k, v in op_dict.items():
                        f.write(f"[Optimal Threshold] {k}: {v:.4f}\n")

                    f.write("\n=== CV Summary Based on Validation Scores ===\n")
                    for metric, (mean, std) in cv_results["final_results"].items():
                        f.write(f"[CV] {metric}: {mean:.4f} ± {std:.4f}\n")

                    f.write("\n=== Point Estimate Metrics ===\n")
                    for k, v in test_metrics.items():
                        f.write(f"[Point Estimate] {k}: {v:.4f}\n")

                    f.write("\n=== Confidence Intervals ===\n")
                    for k, v in ci_dict.items():
                        f.write(f"[Confidence Interval] {k}: {v['mean']:.4f} ± {v['std']:.4f} [{v['lower']:.4f}, {v['upper']:.4f}]\n")

                    f.write("\n=== G2 Predictions ===\n")
                    for k, v in return_dict.items():
                        f.write(f"[G2] Slide ID: {k} | True Label: {v['true_label']} | Predicted: {v['predicted']} | "
                                f"Probabilities: {v['probabilities']}\n")

                    f.write(f"\n=== Wrong Predictions ({len(wrong_predictions)}) ===\n")
                    wrong_pred_str = ", ".join(wrong_predictions)
                    f.write(f"[Wrong Predictions] {wrong_pred_str}\n")

                print(f"Results saved to {save_path}")

if __name__ == "__main__":
    tasks = ["Binary WHO 2022"] # tasks = ["Binary Stage N", "Binary Stage T", "Binary TNM Stage", "died_within_5_years", "MSI", "BRAF"]

    # training_cohorts = [["SR"]]
    lr_range = [5e-5] # 5e-5, 1e-04, 2e-04, 3e-4, 5e-4, 7e-4, 1e-3
    wd_range = [5e-05, 5e-03, 1e-02] # [5e-05, 1e-04, 5e-04, 1e-03, 5e-03, 1e-02]
    mils = ["ABMIL"] # ["ABMIL", "CLAM", "DSMIL", "TransMIL", "WIKGMIL", "Mean", "Max"]
    encoders = ["uni_v1"]
    
    train_df = pd.read_csv('./data/bladder_train_G1_G3.csv')
    test_df = pd.read_csv('./data/bladder_test_G2.csv')

    for encoder in encoders:
        feature_dir = f"{major_dir}/features_{encoder}"
        preloaded_features, D, preloaded = load_features_mmap(df, feature_dir=feature_dir, id_col="De ID", verbose=True, precision=16)
        for task in tasks:
            for mil in mils:
                print(f"Running experiment for task: {task} with MIL: {mil} and Encoder: {encoder})")
                run_one_experiment(task = task,
                                    mil=mil,
                                    encoder=encoder,
                                    D=D,
                                    preloaded_features=preloaded_features,
                                    train_df=train_df,
                                    test_df=test_df,
                                    lr_range=lr_range,
                                    wd_range=wd_range,
                                    device=device,
                                    precision=16,
                                    save_dir="./results/test_gs",
                                    verbose=True,
                                    preloaded=preloaded,
                                    feature_dir=feature_dir)