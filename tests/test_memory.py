"""
Tests for oceanpath.utils.memory — GPU memory budget estimation.

Covers:
  - Parameter count estimation matches known architectures
  - Memory components scale correctly with bag size / feature dim
  - TransMIL attention is quadratic in N
  - AMP halves activation memory
  - Gradient checkpointing reduces backward buffer
  - fits_gpu flag and warnings
"""

import pytest


def _make_config(**overrides):
    """Create a plain-dict config that mimics Hydra DictConfig."""
    cfg = {
        "model": {
            "arch": "abmil",
            "embed_dim": 512,
            "num_fc_layers": 1,
            "num_attention_layers": 2,
            "num_heads": 8,
            "gradient_checkpointing": False,
        },
        "encoder": {"feature_dim": 1024, "name": "uni_v1"},
        "training": {"max_instances": 4000, "batch_size": 1},
        "platform": {"precision": "32"},
    }

    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value

    return cfg


class TestEstimateMemory:
    def test_returns_memory_budget(self):
        from oceanpath.utils.memory import estimate_memory

        cfg = _make_config()
        budget = estimate_memory(cfg, gpu_memory_mb=24000)
        assert budget.total_mb > 0
        assert budget.model_params_mb > 0

    def test_total_is_sum_of_components(self):
        from oceanpath.utils.memory import estimate_memory

        cfg = _make_config()
        budget = estimate_memory(cfg, gpu_memory_mb=24000)

        expected = (
            budget.model_params_mb
            + budget.optimizer_state_mb
            + budget.gradient_mb
            + budget.forward_activation_mb
            + budget.attention_mb
            + budget.backward_buffer_mb
            + budget.cuda_overhead_mb
        )
        assert budget.total_mb == pytest.approx(expected, abs=0.01)

    def test_larger_bag_more_memory(self):
        """Doubling max_instances should significantly increase memory."""
        from oceanpath.utils.memory import estimate_memory

        cfg_small = _make_config(**{"training.max_instances": 2000})
        cfg_large = _make_config(**{"training.max_instances": 8000})

        small = estimate_memory(cfg_small, gpu_memory_mb=24000)
        large = estimate_memory(cfg_large, gpu_memory_mb=24000)

        assert large.forward_activation_mb > small.forward_activation_mb * 3

    def test_larger_feature_dim_more_memory(self):
        """UNI v2 (1536-dim) uses more memory than UNI v1 (1024-dim)."""
        from oceanpath.utils.memory import estimate_memory

        cfg_1024 = _make_config(**{"encoder.feature_dim": 1024})
        cfg_1536 = _make_config(**{"encoder.feature_dim": 1536})

        m1024 = estimate_memory(cfg_1024, gpu_memory_mb=24000)
        m1536 = estimate_memory(cfg_1536, gpu_memory_mb=24000)

        assert m1536.total_mb > m1024.total_mb

    def test_transmil_quadratic_attention(self):
        """TransMIL attention should be much larger than ABMIL."""
        from oceanpath.utils.memory import estimate_memory

        cfg_abmil = _make_config(**{"model.arch": "abmil", "training.max_instances": 4000})
        cfg_transmil = _make_config(**{"model.arch": "transmil", "training.max_instances": 4000})

        abmil = estimate_memory(cfg_abmil, gpu_memory_mb=24000)
        transmil = estimate_memory(cfg_transmil, gpu_memory_mb=24000)

        # TransMIL N² attention >> ABMIL Nx1 attention
        assert transmil.attention_mb > abmil.attention_mb * 100

    def test_amp_reduces_activation_memory(self):
        """16-mixed precision should halve activation memory."""
        from oceanpath.utils.memory import estimate_memory

        cfg_fp32 = _make_config(**{"platform.precision": "32"})
        cfg_fp16 = _make_config(**{"platform.precision": "16-mixed"})

        fp32 = estimate_memory(cfg_fp32, gpu_memory_mb=24000)
        fp16 = estimate_memory(cfg_fp16, gpu_memory_mb=24000)

        assert fp16.forward_activation_mb == pytest.approx(
            fp32.forward_activation_mb / 2,
            rel=0.01,
        )

    def test_gradient_checkpointing_reduces_backward(self):
        from oceanpath.utils.memory import estimate_memory

        cfg_off = _make_config(**{"model.gradient_checkpointing": False})
        cfg_on = _make_config(**{"model.gradient_checkpointing": True})

        off = estimate_memory(cfg_off, gpu_memory_mb=24000)
        on = estimate_memory(cfg_on, gpu_memory_mb=24000)

        assert on.backward_buffer_mb < off.backward_buffer_mb

    def test_fits_gpu_true_when_within_budget(self):
        from oceanpath.utils.memory import estimate_memory

        cfg = _make_config(**{"training.max_instances": 100})
        budget = estimate_memory(cfg, gpu_memory_mb=24000)
        assert budget.fits_gpu is True
        assert budget.headroom_pct > 0

    def test_fits_gpu_false_when_over_budget(self):
        from oceanpath.utils.memory import estimate_memory

        # Extreme config: TransMIL with huge bags on tiny GPU
        cfg = _make_config(**{"model.arch": "transmil", "training.max_instances": 50000})
        budget = estimate_memory(cfg, gpu_memory_mb=8000)
        assert budget.fits_gpu is False
        assert any("Estimated" in w for w in budget.warnings)

    def test_transmil_large_bag_warning(self):
        """TransMIL with N>4000 should generate a warning."""
        from oceanpath.utils.memory import estimate_memory

        cfg = _make_config(**{"model.arch": "transmil", "training.max_instances": 8000})
        budget = estimate_memory(cfg, gpu_memory_mb=24000)
        assert any("attention matrix" in w.lower() for w in budget.warnings)

    def test_static_no_attention(self):
        from oceanpath.utils.memory import estimate_memory

        cfg = _make_config(**{"model.arch": "static"})
        budget = estimate_memory(cfg, gpu_memory_mb=24000)
        assert budget.attention_mb == 0.0

    def test_optimizer_state_is_2x_params(self):
        """AdamW stores momentum + variance = 2x parameter memory."""
        from oceanpath.utils.memory import estimate_memory

        cfg = _make_config()
        budget = estimate_memory(cfg, gpu_memory_mb=24000)
        assert budget.optimizer_state_mb == pytest.approx(
            budget.model_params_mb * 2,
            rel=0.01,
        )

    def test_to_dict(self):
        from oceanpath.utils.memory import estimate_memory

        cfg = _make_config()
        budget = estimate_memory(cfg, gpu_memory_mb=24000)
        d = budget.to_dict()
        assert isinstance(d, dict)
        assert "total_mb" in d
        assert "arch" in d


class TestParamCountEstimation:
    def test_abmil_nonzero(self):
        from oceanpath.utils.memory import _estimate_param_count

        count = _estimate_param_count("abmil", 1024, 512, 1)
        assert count > 500_000  # ABMIL with D=1024, E=512 ≈ 920K

    def test_transmil_larger_than_abmil(self):
        from oceanpath.utils.memory import _estimate_param_count

        abmil = _estimate_param_count("abmil", 1024, 512, 1)
        transmil = _estimate_param_count("transmil", 1024, 512, 1, 2, 8)
        assert transmil > abmil

    def test_static_smallest(self):
        from oceanpath.utils.memory import _estimate_param_count

        static = _estimate_param_count("static", 1024, 512, 1)
        abmil = _estimate_param_count("abmil", 1024, 512, 1)
        assert static < abmil

    def test_more_layers_more_params(self):
        from oceanpath.utils.memory import _estimate_param_count

        one = _estimate_param_count("abmil", 1024, 512, 1)
        three = _estimate_param_count("abmil", 1024, 512, 3)
        assert three > one


class TestPrintBudget:
    """Smoke test: print_memory_budget doesn't crash."""

    def test_print(self, capsys):
        from oceanpath.utils.memory import estimate_memory, print_memory_budget

        cfg = _make_config()
        budget = estimate_memory(cfg, gpu_memory_mb=24000)
        print_memory_budget(budget)

        captured = capsys.readouterr()
        assert "GPU Memory Budget" in captured.out
        assert "TOTAL" in captured.out
