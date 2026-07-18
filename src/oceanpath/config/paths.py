"""Canonical filesystem layout for the foundation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oceanpath.config.access import cfg_select
from oceanpath.utils.repro import config_fingerprint

_ARTIFACT_IDENTITY_GROUPS = (
    "data",
    "encoder",
    "extraction",
    "splits",
    "model",
    "training",
    "eval",
)

_TRAINING_IDENTITY_GROUPS = (
    "data",
    "encoder",
    "extraction",
    "splits",
    "model",
)


def _to_plain_config(value: Any) -> Any:
    """Resolve an OmegaConf node without imposing Hydra on callers."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except (ImportError, ValueError):
        pass
    return value


def artifact_config_fingerprint(cfg: Any) -> str:
    """Hash only settings that can change an exported model artifact.

    Runtime controls such as dry-run, force, logging, tracking, DAG rendering,
    and destination paths must never create a new artifact identity.
    """
    payload = {
        group: _to_plain_config(cfg_select(cfg, group, {})) for group in _ARTIFACT_IDENTITY_GROUPS
    }
    training = payload.get("training")
    if isinstance(training, dict):
        training = dict(training)
        training.pop("force_rerun", None)
        payload["training"] = training
    payload["experiment_name"] = str(cfg_select(cfg, "exp_name", "experiment"))
    payload["export"] = {
        key: _to_plain_config(cfg_select(cfg, f"export.{key}", None))
        for key in (
            "model_strategy",
            "formats",
            "opset_version",
            "skip_validation",
            "atol",
            "rtol",
            "validation_n_patches",
        )
    }
    return config_fingerprint(payload)


def training_config_fingerprint(cfg: Any) -> str:
    """Hash config-only settings that can change trained weights.

    This deliberately excludes input file contents so path resolution stays
    stable before and after upstream stages run. Training completion metadata
    separately fingerprints the manifest, splits, mmap, and preload weights.
    """
    payload = {
        group: _to_plain_config(cfg_select(cfg, group, {})) for group in _TRAINING_IDENTITY_GROUPS
    }
    training = _to_plain_config(cfg_select(cfg, "training", {}))
    if isinstance(training, dict):
        training = dict(training)
        training.pop("force_rerun", None)
    payload["training"] = training
    payload["platform_compute"] = {
        key: _to_plain_config(cfg_select(cfg, f"platform.{key}", None))
        for key in ("accelerator", "devices", "strategy", "precision", "num_nodes")
    }
    return config_fingerprint(payload)


@dataclass(frozen=True)
class FoundationPaths:
    """Resolved paths shared by extraction, training, evaluation, and export."""

    dataset_name: str
    experiment_name: str
    slide_dir: Path
    manifest_path: Path
    extraction_dir: Path
    feature_h5_dir: Path
    mmap_dir: Path
    splits_dir: Path
    train_dir: Path
    eval_dir: Path
    artifact_dir: Path

    @classmethod
    def from_config(cls, cfg: Any) -> FoundationPaths:
        """Build the path contract once from a composed workflow config."""
        dataset_name = str(cfg_select(cfg, "data.name", "dataset"))
        encoder_name = str(cfg_select(cfg, "encoder.name", "encoder"))
        split_name = str(cfg_select(cfg, "splits.name", "splits"))
        experiment_name = str(cfg_select(cfg, "exp_name", dataset_name))

        project_root = Path(str(cfg_select(cfg, "platform.project_root", ".")))
        output_root = Path(str(cfg_select(cfg, "platform.output_root", project_root / "outputs")))

        slide_dir = Path(
            str(cfg_select(cfg, "data.slide_dir", project_root / "slides" / dataset_name))
        )
        manifest_path = Path(
            str(
                cfg_select(
                    cfg,
                    "data.csv_path",
                    project_root / "manifests" / f"{dataset_name}.csv",
                )
            )
        )
        extraction_dir = Path(
            str(
                cfg_select(
                    cfg,
                    "data.feature_job_dir",
                    project_root / "features" / dataset_name,
                )
            )
        )
        feature_h5_dir = Path(
            str(
                cfg_select(
                    cfg,
                    "data.feature_h5_dir",
                    extraction_dir / f"features_{encoder_name}",
                )
            )
        )
        mmap_dir = Path(
            str(
                cfg_select(
                    cfg,
                    "data.mmap_dir",
                    project_root / "mmap" / dataset_name / encoder_name,
                )
            )
        )
        splits_dir = Path(
            str(
                cfg_select(
                    cfg,
                    "splits.output_dir",
                    Path(str(cfg_select(cfg, "platform.splits_root", project_root / "splits")))
                    / dataset_name
                    / split_name,
                )
            )
        )
        explicit_train_dir = cfg_select(cfg, "train_dir", None)
        if explicit_train_dir not in (None, "null"):
            train_dir = Path(str(explicit_train_dir))
        else:
            train_root = Path(str(cfg_select(cfg, "experiment.train_root", output_root / "train")))
            train_dir = train_root / (f"{experiment_name}_{training_config_fingerprint(cfg)}")

        explicit_eval_dir = cfg_select(cfg, "eval_dir", None)
        if explicit_eval_dir not in (None, "null"):
            eval_dir = Path(str(explicit_eval_dir))
        else:
            eval_subdir = str(cfg_select(cfg, "experiment.eval_subdir", "eval"))
            eval_dir = train_dir / eval_subdir

        explicit_artifact_dir = cfg_select(cfg, "export.artifact_dir", None)
        if explicit_artifact_dir not in (None, "null"):
            artifact_dir = Path(str(explicit_artifact_dir))
        else:
            artifact_root = Path(
                str(cfg_select(cfg, "export.artifact_root", output_root / "artifacts"))
            )
            artifact_dir = artifact_root / (f"{experiment_name}_{artifact_config_fingerprint(cfg)}")

        return cls(
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            slide_dir=slide_dir,
            manifest_path=manifest_path,
            extraction_dir=extraction_dir,
            feature_h5_dir=feature_h5_dir,
            mmap_dir=mmap_dir,
            splits_dir=splits_dir,
            train_dir=train_dir,
            eval_dir=eval_dir,
            artifact_dir=artifact_dir,
        )
