"""
Random-weight serving artifacts, for tests, the serving image's CI smoke test and
latency benchmarks (scripts/make_demo_artifact.py, scripts/benchmark_serving.py).

The weights are random, so the predictions mean nothing; the artifact exercises the
export -> serve path end to end (ONNX + TorchScript export with numerical
validation, model card, FastAPI server) without data or a trained checkpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path

import lightning as L
import torch
from omegaconf import OmegaConf

from oceanpath.modules.train_module import MILTrainModule
from oceanpath.serving.exporter import Exporter
from oceanpath.serving.model_card import ModelCard

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

logger = logging.getLogger(__name__)


def export_random_model(
    out_dir: Path,
    arch: str = "abmil",
    in_dim: int = 1536,
    num_classes: int = 2,
    seed: int = 0,
    validation_n_patches: list[int] | None = None,
) -> dict:
    """Build, checkpoint and export one random model; return the export report."""
    torch.manual_seed(seed)
    model_cfg = OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / "model" / f"{arch}.yaml"))
    if not isinstance(model_cfg, dict):
        raise TypeError(f"configs/model/{arch}.yaml must be a mapping")
    module = MILTrainModule(
        arch=model_cfg["arch"],
        in_dim=in_dim,
        num_classes=num_classes,
        model_cfg=model_cfg,
        collect_embeddings=False,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir.parent / f".{out_dir.name}.random-init.ckpt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": L.__version__,
            "epoch": 0,
            "global_step": 0,
        },
        ckpt_path,
    )

    report = Exporter(
        checkpoint_path=str(ckpt_path),
        output_dir=str(out_dir),
        in_dim=in_dim,
        num_classes=num_classes,
        validation_n_patches=validation_n_patches,
    ).run()
    ckpt_path.unlink()

    cfg = {
        "exp_name": f"demo_random_{arch}",
        "num_classes": num_classes,
        "model": model_cfg,
        "encoder": {"name": "synthetic", "feature_dim": in_dim},
        "data": {"name": "synthetic"},
    }
    card = ModelCard(cfg, report)
    card.add_custom("note", "Random weights: for serving tests and latency benchmarks only.")
    card.write(str(out_dir / "model_card.json"))
    return report
