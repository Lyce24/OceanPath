"""Reusable supervised-training mechanics.

This package owns the training-time building blocks that operate on typed
inputs: fold planning (:mod:`oceanpath.training.folds`), the Lightning module
(:mod:`oceanpath.training.lightning`), and the training callbacks
(:mod:`oceanpath.training.callbacks`).

Only the lightweight fold planner is re-exported here. The Lightning module and
callbacks are imported directly from their submodules by callers, so that
importing :mod:`oceanpath.training` for fold planning never eagerly pulls in
``torch``/``lightning``. Experiment orchestration (running folds, finalizing
models) lives in :mod:`oceanpath.workflows`, not here.
"""

from oceanpath.training.folds import FoldPlanConfig, FoldSpec, plan_folds

__all__ = ["FoldPlanConfig", "FoldSpec", "plan_folds"]
