"""
Platform / DDP / hyperparameter resolvers shared across training scripts.

Single source of truth for `cfg.training.* | cfg.platform.*` lookups so
multiple training entry-points (train_supervised.py, future scripts) don't
silently disagree on defaults. Read-only against the Hydra config; never
mutates state.

DDP helpers are environment-aware: they query `RANK` / `torch.distributed`
without depending on Lightning being initialized, so they can be called
both before and after `Trainer.fit`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from omegaconf import DictConfig

# ─────────────────────────────────────────────────────────────────────────────
# Local re-import to avoid pulling oceanpath.data into this leaf module's
# import graph at module-load time. The function is small and stable.
# ─────────────────────────────────────────────────────────────────────────────


def _cuda_pin_memory_available() -> bool:
    return bool(torch.cuda.is_available())


# ── Hyperparameter resolvers ────────────────────────────────────────────────


def resolve_precision(cfg: DictConfig) -> str:
    """Prefer task-specific precision, fall back to the platform default."""
    from omegaconf import OmegaConf

    return OmegaConf.select(
        cfg,
        "training.precision",
        default=OmegaConf.select(cfg, "platform.precision", default="bf16-mixed"),
    )


def resolve_num_workers(cfg: DictConfig) -> int:
    """Prefer task-specific workers, fall back to the platform default."""
    from omegaconf import OmegaConf

    return int(
        OmegaConf.select(
            cfg,
            "training.num_workers",
            default=OmegaConf.select(cfg, "platform.num_workers", default=0),
        )
    )


def resolve_force_float32(cfg: DictConfig) -> bool:
    """Allow tasks to preserve the mmap feature dtype."""
    from omegaconf import OmegaConf

    return bool(OmegaConf.select(cfg, "training.force_float32", default=False))


def resolve_prefetch_factor(cfg: DictConfig) -> int:
    from omegaconf import OmegaConf

    return int(OmegaConf.select(cfg, "training.prefetch_factor", default=2))


def resolve_pin_memory(cfg: DictConfig) -> bool:
    from omegaconf import OmegaConf

    pin_memory = OmegaConf.select(cfg, "training.pin_memory", default=None)
    return _cuda_pin_memory_available() if pin_memory is None else bool(pin_memory)


def resolve_persistent_workers(cfg: DictConfig, num_workers: int) -> bool:
    from omegaconf import OmegaConf

    persistent_workers = OmegaConf.select(cfg, "training.persistent_workers", default=None)
    if persistent_workers is None:
        return num_workers > 0
    return bool(persistent_workers) and num_workers > 0


def resolve_accelerator(cfg: DictConfig) -> str:
    from omegaconf import OmegaConf

    return OmegaConf.select(
        cfg,
        "platform.accelerator",
        default="gpu" if torch.cuda.is_available() else "cpu",
    )


def resolve_devices(cfg: DictConfig):
    from omegaconf import OmegaConf

    return OmegaConf.select(cfg, "platform.devices", default=1)


def resolve_strategy(cfg: DictConfig) -> str:
    from omegaconf import OmegaConf

    return OmegaConf.select(cfg, "platform.strategy", default="auto")


def resolve_num_nodes(cfg: DictConfig) -> int:
    from omegaconf import OmegaConf

    return int(OmegaConf.select(cfg, "platform.num_nodes", default=1))


# ── DDP helpers ─────────────────────────────────────────────────────────────


def global_rank() -> int:
    """Rank of the current process. Defaults to 0 when DDP is not active.

    Reads ``RANK`` from the environment so it works both before Trainer
    construction (Lightning sets it before importing user code under
    ``ddp_spawn`` / ``torchrun``) and after.
    """
    return int(os.environ.get("RANK", "0"))


def is_rank_zero() -> bool:
    return global_rank() == 0


def ddp_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    """Synchronize all DDP ranks. No-op when DDP is not active."""
    if ddp_active():
        dist.barrier()


# ── JSON serialization ──────────────────────────────────────────────────────


def json_default(obj: Any):
    """JSON serializer for numpy / torch scalars and arrays.

    Mirrors the behaviour previously inlined in train_supervised.py so output
    formatting is byte-identical: scalar tensors -> .item(), n-D tensors and
    arrays -> .tolist(), unrecognized types fall back to str(obj) rather than
    raising (we want best-effort JSON dumps from heterogeneous summary dicts).
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    return str(obj)
