# OceanPath

OceanPath is a modernized MIL pathology pipeline built from `OceanPath_v1` concepts.

## What was migrated from OceanPath_v1

- WSI model definition (`ABMIL`, `TransMIL`, static pooling MIL)
- Lightning WSI training module (fine-tune + linear probe)
- k-fold cross-validation training/evaluation pipeline

## New workflow features

- Visualization during evaluation (ROC/PR + probability histograms)
- Model export (TorchScript + ONNX)
- Statistical analysis (paired permutation test for AUC)

## Expected training CSV format

Use a CSV containing at least:

- `feature_path`: path to `.pt` or `.npy` bag feature file
- `label`: integer class label
- `k_fold`: fold id (`0..K-1` for train/val folds, `-1` for held-out test)

## Run k-fold training

```bash
python scripts/train.py \
	--csv /path/to/folds.csv \
	--output-dir outputs/cv_abmil \
	--mil ABMIL \
	--feature-dim 1024 \
	--n-classes 2
```

This writes:

- per-fold checkpoints in `outputs/cv_abmil/fold_*/`
- `cv_results.json`
- `ensemble_test_predictions.csv` (if test rows with `k_fold=-1` exist)

## Evaluate + visualize

```bash
python scripts/evaluate.py \
	--pred-csv outputs/cv_abmil/ensemble_test_predictions.csv \
	--output-dir outputs/cv_abmil/eval
```

Outputs:

- `metrics.json`
- `roc_pr_curves.png`
- `probability_histogram.png`

## Export model

```bash
python scripts/export_model.py \
	--checkpoint outputs/cv_abmil/fold_0/best.ckpt \
	--output-dir outputs/cv_abmil/export \
	--feature-dim 1024
```

Outputs:

- `wsi_model.ts`
- `wsi_model.onnx`

## Compare two models (analysis)

```bash
python scripts/analyze.py \
	--pred-a outputs/model_a_preds.csv \
	--pred-b outputs/model_b_preds.csv \
	--output outputs/significance.json
```
