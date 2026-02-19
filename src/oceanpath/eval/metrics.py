from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	balanced_accuracy_score,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
	y_true = np.asarray(y_true).astype(int)
	y_prob = np.asarray(y_prob).astype(float)
	y_pred = (y_prob >= threshold).astype(int)

	output: Dict[str, float] = {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
		"precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"f1": float(f1_score(y_true, y_pred, zero_division=0)),
	}

	unique = np.unique(y_true)
	if len(unique) == 2:
		output["roc_auc"] = float(roc_auc_score(y_true, y_prob))
		output["average_precision"] = float(average_precision_score(y_true, y_prob))

	return output


def bootstrap_ci(
	y_true: np.ndarray,
	y_prob: np.ndarray,
	metric_name: str = "roc_auc",
	n_bootstrap: int = 1000,
	alpha: float = 0.95,
	threshold: float = 0.5,
	seed: int = 42,
) -> Tuple[float, float, float]:
	rng = np.random.default_rng(seed)
	y_true = np.asarray(y_true).astype(int)
	y_prob = np.asarray(y_prob).astype(float)
	n = len(y_true)

	if metric_name == "roc_auc" and len(np.unique(y_true)) < 2:
		return float("nan"), float("nan"), float("nan")

	def _metric(yt: np.ndarray, yp: np.ndarray) -> float:
		if metric_name == "roc_auc":
			return float(roc_auc_score(yt, yp))
		if metric_name == "average_precision":
			return float(average_precision_score(yt, yp))
		if metric_name == "accuracy":
			return float(accuracy_score(yt, (yp >= threshold).astype(int)))
		raise ValueError(f"Unsupported metric for bootstrap_ci: {metric_name}")

	point = _metric(y_true, y_prob)

	samples = []
	for _ in range(n_bootstrap):
		indices = rng.integers(0, n, size=n)
		yt = y_true[indices]
		yp = y_prob[indices]
		if metric_name == "roc_auc" and len(np.unique(yt)) < 2:
			continue
		samples.append(_metric(yt, yp))

	if not samples:
		return point, float("nan"), float("nan")

	lower_q = (1.0 - alpha) / 2.0
	upper_q = 1.0 - lower_q
	lower = float(np.quantile(samples, lower_q))
	upper = float(np.quantile(samples, upper_q))
	return point, lower, upper

