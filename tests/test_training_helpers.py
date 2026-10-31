"""
Tests for the config resolvers and rank helpers that scripts/train_supervised.py
imports from ``oceanpath.training``.

Covers:
  * precision and DataLoader workers: training.* over platform.* over the
    built-in default; the defaults and pass-through of the other resolvers
  * pin_memory, persistent_workers and accelerator auto-detection
  * RANK-based rank helpers; barrier only synchronises under DDP
  * json_default for numpy / torch scalars and arrays (summary JSON dumps)
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from oceanpath import training


def cfg(training_cfg=None, platform_cfg=None):
    return OmegaConf.create({"training": training_cfg or {}, "platform": platform_cfg or {}})


# ═════════════════════════════════════════════════════════════════════════════
# Resolvers
# ═════════════════════════════════════════════════════════════════════════════


class TestResolvers:
    @pytest.mark.parametrize(
        "training_cfg, platform_cfg, expected",
        [
            ({"precision": "32-true"}, {"precision": "16-mixed"}, "32-true"),
            ({}, {"precision": "16-mixed"}, "16-mixed"),
            ({}, {}, "bf16-mixed"),
        ],
    )
    def test_precision_prefers_training_then_platform(self, training_cfg, platform_cfg, expected):
        assert training.resolve_precision(cfg(training_cfg, platform_cfg)) == expected

    @pytest.mark.parametrize(
        "training_cfg, platform_cfg, expected",
        [({"num_workers": 2}, {"num_workers": 8}, 2), ({}, {"num_workers": 8}, 8), ({}, {}, 0)],
    )
    def test_num_workers_prefers_training_then_platform(self, training_cfg, platform_cfg, expected):
        workers = training.resolve_num_workers(cfg(training_cfg, platform_cfg))
        assert workers == expected
        assert isinstance(workers, int)

    def test_defaults_for_an_empty_config(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        empty = cfg()
        assert training.resolve_force_float32(empty) is False
        assert training.resolve_prefetch_factor(empty) == 2
        assert training.resolve_devices(empty) == 1
        assert training.resolve_strategy(empty) == "auto"
        assert training.resolve_num_nodes(empty) == 1
        assert training.resolve_accelerator(empty) == "cpu"

    def test_configured_values_pass_through(self):
        c = cfg(
            {"force_float32": True, "prefetch_factor": 1},
            {"devices": 2, "strategy": "ddp", "num_nodes": 4, "accelerator": "cpu"},
        )
        assert training.resolve_force_float32(c) is True
        assert training.resolve_prefetch_factor(c) == 1
        assert training.resolve_devices(c) == 2
        assert training.resolve_strategy(c) == "ddp"
        assert training.resolve_num_nodes(c) == 4
        assert training.resolve_accelerator(c) == "cpu"

    @pytest.mark.parametrize("cuda", [True, False])
    def test_pin_memory_follows_cuda_unless_set(self, monkeypatch, cuda):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        assert training.resolve_pin_memory(cfg({"pin_memory": None})) is cuda
        assert training.resolve_pin_memory(cfg()) is cuda
        assert training.resolve_pin_memory(cfg({"pin_memory": False})) is False
        assert training.resolve_pin_memory(cfg({"pin_memory": True})) is True

    @pytest.mark.parametrize(
        "setting, num_workers, expected",
        [(None, 0, False), (None, 4, True), (True, 0, False), (True, 4, True), (False, 4, False)],
    )
    def test_persistent_workers_need_workers(self, setting, num_workers, expected):
        c = cfg({"persistent_workers": setting})
        assert training.resolve_persistent_workers(c, num_workers) is expected

    def test_accelerator_defaults_to_gpu_only_with_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert training.resolve_accelerator(cfg()) == "gpu"
        assert training.resolve_accelerator(cfg(platform_cfg={"accelerator": "cpu"})) == "cpu"


# ═════════════════════════════════════════════════════════════════════════════
# Rank helpers
# ═════════════════════════════════════════════════════════════════════════════


class TestRankHelpers:
    def test_rank_comes_from_the_rank_environment_variable(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        assert (training.global_rank(), training.is_rank_zero()) == (0, True)
        monkeypatch.setenv("RANK", "3")
        assert (training.global_rank(), training.is_rank_zero()) == (3, False)

    def test_barrier_only_synchronises_under_ddp(self, monkeypatch):
        calls = []
        monkeypatch.setattr(dist, "barrier", lambda *a, **k: calls.append("barrier"))
        assert training.ddp_active() is False
        training.barrier()
        assert calls == []
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        assert training.ddp_active() is True
        training.barrier()
        assert calls == ["barrier"]


# ═════════════════════════════════════════════════════════════════════════════
# JSON serialisation
# ═════════════════════════════════════════════════════════════════════════════


class TestJsonDefault:
    def test_numpy_and_torch_values_serialise(self):
        payload = {
            "n": np.int64(3),
            "x": np.float32(0.5),
            "arr": np.arange(3),
            "scalar": torch.tensor(2.5),
            "matrix": torch.tensor([[1, 2], [3, 4]]),
            "grad": torch.ones(2, requires_grad=True),
            "path": Path("runs/fold_0"),
        }
        decoded = json.loads(json.dumps(payload, default=training.json_default))
        assert decoded == {
            "n": 3,
            "x": 0.5,
            "arr": [0, 1, 2],
            "scalar": 2.5,
            "matrix": [[1, 2], [3, 4]],
            "grad": [1.0, 1.0],
            "path": "runs/fold_0",
        }
