"""Canonical fold planning for supervised experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FoldPlanConfig:
    """Typed values needed to turn a split scheme into runnable folds."""

    scheme: str
    n_folds: int = 0
    n_repeats: int = 0


@dataclass(frozen=True)
class FoldSpec:
    """One independently trainable fold or repeated holdout."""

    index: int
    scheme: str


def plan_folds(config: FoldPlanConfig) -> tuple[FoldSpec, ...]:
    """Return the folds to run; reject unsupported split schemes loudly.

    Nested CV needs an explicit inner-loop selection contract and is therefore
    intentionally not part of the foundation trainer. Project branches may add
    a dedicated planner without changing the supported foundation schemes.
    """
    scheme = config.scheme
    if scheme in {"holdout", "custom_holdout"}:
        count = 1
    elif scheme in {"kfold", "custom_kfold"}:
        count = config.n_folds
    elif scheme == "monte_carlo":
        count = config.n_repeats
    else:
        raise ValueError(f"Unknown split scheme: {scheme!r}")

    if count < 1:
        raise ValueError(f"Split scheme {scheme!r} requires at least one fold/repeat")
    return tuple(FoldSpec(index=index, scheme=scheme) for index in range(count))
