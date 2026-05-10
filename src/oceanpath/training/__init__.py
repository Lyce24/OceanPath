"""
Shared training-orchestrator primitives.

This package collects platform / DDP / hyperparameter resolvers and other
utilities used by `scripts/train_supervised.py` (and reusable by any future
training entry-point that needs the same Hydra-config -> Trainer plumbing).

`scripts/train.py` accesses `cfg.platform.X` directly and does not currently
use these helpers; consolidating it here is left for a future refactor.
"""

from oceanpath.training._helpers import (
    barrier,
    ddp_active,
    global_rank,
    is_rank_zero,
    json_default,
    resolve_accelerator,
    resolve_devices,
    resolve_force_float32,
    resolve_num_nodes,
    resolve_num_workers,
    resolve_persistent_workers,
    resolve_pin_memory,
    resolve_precision,
    resolve_prefetch_factor,
    resolve_strategy,
)

__all__ = [
    "barrier",
    "ddp_active",
    "global_rank",
    "is_rank_zero",
    "json_default",
    "resolve_accelerator",
    "resolve_devices",
    "resolve_force_float32",
    "resolve_num_nodes",
    "resolve_num_workers",
    "resolve_persistent_workers",
    "resolve_pin_memory",
    "resolve_precision",
    "resolve_prefetch_factor",
    "resolve_strategy",
]
