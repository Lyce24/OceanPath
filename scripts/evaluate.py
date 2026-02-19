from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from oceanpath.eval.metrics import bootstrap_ci, classification_metrics


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate WSI predictions and generate plots")
	parser.add_argument("--pred-csv", required=True, help="CSV with columns y_true,y_prob")
	parser.add_argument("--output-dir", required=True)
	parser.add_argument("--threshold", type=float, default=0.5)
	parser.add_argument("--bootstrap", type=int, default=1000)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	df = pd.read_csv(args.pred_csv)
	if not {"y_true", "y_prob"}.issubset(df.columns):
		raise ValueError("Prediction CSV must contain columns y_true and y_prob")

	y_true = df["y_true"].to_numpy().astype(int)
	y_prob = df["y_prob"].to_numpy().astype(float)

	metrics = classification_metrics(y_true, y_prob, threshold=args.threshold)
	if "roc_auc" in metrics:
		point, low, high = bootstrap_ci(y_true, y_prob, metric_name="roc_auc", n_bootstrap=args.bootstrap)
		metrics["roc_auc_ci"] = {"point": point, "low": low, "high": high}

	metrics_path = output_dir / "metrics.json"
	with open(metrics_path, "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)

	try:
		import matplotlib.pyplot as plt

		fig, axes = plt.subplots(1, 2, figsize=(12, 5))
		RocCurveDisplay.from_predictions(y_true=y_true, y_pred=y_prob, ax=axes[0])
		axes[0].set_title("ROC Curve")
		PrecisionRecallDisplay.from_predictions(y_true=y_true, y_pred=y_prob, ax=axes[1])
		axes[1].set_title("Precision-Recall Curve")
		fig.tight_layout()
		fig.savefig(output_dir / "roc_pr_curves.png", dpi=200)
		plt.close(fig)

		fig2, ax2 = plt.subplots(figsize=(7, 5))
		bins = np.linspace(0.0, 1.0, 21)
		ax2.hist(y_prob[y_true == 0], bins=bins, alpha=0.6, label="Negative")
		ax2.hist(y_prob[y_true == 1], bins=bins, alpha=0.6, label="Positive")
		ax2.set_title("Prediction Distribution")
		ax2.set_xlabel("Predicted probability")
		ax2.legend()
		fig2.tight_layout()
		fig2.savefig(output_dir / "probability_histogram.png", dpi=200)
		plt.close(fig2)
	except Exception:
		pass

	print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
	main()

