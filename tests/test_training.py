"""CPU contract tests for the foundation Lightning training module."""

import torch

from oceanpath.training.lightning import MILTrainModule


def _tiny_module() -> MILTrainModule:
    return MILTrainModule(
        arch="abmil",
        in_dim=8,
        num_classes=2,
        model_cfg={"embed_dim": 16, "attn_dim": 8, "dropout": 0.0},
        lr=1.0e-3,
        max_epochs=2,
        canary_interval=0,
        collect_embeddings=False,
    )


def test_training_step_backpropagates_through_classifier(monkeypatch):
    module = _tiny_module()
    monkeypatch.setattr(module, "log", lambda *_args, **_kwargs: None)
    batch = {
        "features": torch.randn(2, 5, 8),
        "mask": torch.ones(2, 5, dtype=torch.bool),
        "labels": torch.tensor([0, 1]),
        "slide_ids": ["a", "b"],
    }

    loss = module.training_step(batch, batch_idx=0)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert module.model.head.weight.grad is not None
    assert torch.isfinite(module.model.head.weight.grad).all()


def test_training_module_builds_optimizer_and_scheduler_contract():
    configured = _tiny_module().configure_optimizers()

    assert configured["optimizer"].param_groups
    assert hasattr(configured["lr_scheduler"], "step")
