from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score


def paired_permutation_auc_test(
	y_true: np.ndarray,
	y_prob_a: np.ndarray,
	y_prob_b: np.ndarray,
	n_permutations: int = 5000,
	seed: int = 42,
) -> Dict[str, float]:
	y_true = np.asarray(y_true).astype(int)
	y_prob_a = np.asarray(y_prob_a).astype(float)
	y_prob_b = np.asarray(y_prob_b).astype(float)

	auc_a = float(roc_auc_score(y_true, y_prob_a))
	auc_b = float(roc_auc_score(y_true, y_prob_b))
	observed_diff = auc_a - auc_b

	rng = np.random.default_rng(seed)
	greater_or_equal = 0

	for _ in range(n_permutations):
		swap_mask = rng.random(len(y_true)) < 0.5
		pa = y_prob_a.copy()
		pb = y_prob_b.copy()
		pa[swap_mask], pb[swap_mask] = pb[swap_mask], pa[swap_mask]

		diff = float(roc_auc_score(y_true, pa) - roc_auc_score(y_true, pb))
		if abs(diff) >= abs(observed_diff):
			greater_or_equal += 1

	p_value = (greater_or_equal + 1) / (n_permutations + 1)
	return {
		"auc_a": auc_a,
		"auc_b": auc_b,
		"auc_diff": observed_diff,
		"p_value": float(p_value),
	}

