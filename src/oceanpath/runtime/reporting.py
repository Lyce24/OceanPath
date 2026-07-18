"""Experiment-tracking seam.

Centralizes construction of the experiment tracker (Weights & Biases today) so
that no workflow imports ``wandb`` directly. A project that wants a different
tracker — MLflow, TensorBoard, or none — replaces the factories here and leaves
every workflow untouched.

There are two entry points because training and post-hoc reporting use trackers
differently:

- :func:`build_lightning_logger` returns a Lightning ``Logger`` that streams
  per-step/epoch metrics during ``trainer.fit`` (used by the training workflow).
- :func:`build_reporter` returns a :class:`Reporter` for post-hoc summary/plot
  logging after a job completes (used by the evaluation workflow).

Workflows own *what* to report; this module owns *how* it reaches the tracker.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from oceanpath.config import cfg_select

logger = logging.getLogger(__name__)


# ── Tracker configuration helpers ───────────────────────────────────────────────


def _tracking_enabled(cfg: Any) -> bool:
    """Return True when a live tracker run should be created for this config."""
    if cfg_select(cfg, "wandb", default=None) is None:
        return False
    return bool(cfg_select(cfg, "wandb.enabled", default=False))


def _resolved_config(cfg: Any) -> Any:
    """Convert a config object to a plain container for the tracker's ``config``."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except Exception:  # noqa: BLE001 - config logging must not crash a job
        pass
    return cfg


def _standard_tags(cfg: Any) -> list[str]:
    """The dataset/encoder/model/split tags shared by every tracked run."""
    return [
        cfg_select(cfg, "data.name", default=""),
        cfg_select(cfg, "encoder.name", default=""),
        cfg_select(cfg, "model.arch", default=""),
        cfg_select(cfg, "splits.scheme", default=""),
    ]


def _wandb_finish() -> None:
    try:
        import wandb

        wandb.finish()
    except Exception:  # noqa: BLE001 - finishing a run must never crash a job
        pass


def finish_run() -> None:
    """Finish the active tracker run. Safe no-op if there is none."""
    _wandb_finish()


# ── Post-hoc reporter ───────────────────────────────────────────────────────────


@runtime_checkable
class Reporter(Protocol):
    """Post-hoc experiment reporter used by non-training workflows.

    Implementations must tolerate being called when tracking is disabled; the
    default :class:`NullReporter` makes every method a no-op.
    """

    @property
    def active(self) -> bool:
        """Whether a live run is backing this reporter."""
        ...

    def log_summary(self, values: dict[str, Any]) -> None:
        """Record run-level summary scalars (final values, not time series)."""
        ...

    def log_metrics(self, values: dict[str, Any]) -> None:
        """Log a step of metrics."""
        ...

    def log_image(self, key: str, path: str | Path) -> None:
        """Upload an image artifact under ``key``."""
        ...

    def finish(self) -> None:
        """Close the run."""
        ...


class NullReporter:
    """No-op reporter used when experiment tracking is disabled/unavailable."""

    active = False

    def log_summary(self, values: dict[str, Any]) -> None:
        pass

    def log_metrics(self, values: dict[str, Any]) -> None:
        pass

    def log_image(self, key: str, path: str | Path) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbReporter:
    """:class:`Reporter` backed by a live Weights & Biases run."""

    def __init__(self, run: Any) -> None:
        self._run = run

    @property
    def active(self) -> bool:
        return self._run is not None

    def log_summary(self, values: dict[str, Any]) -> None:
        if self._run is None:
            return
        for key, value in values.items():
            self._run.summary[key] = value

    def log_metrics(self, values: dict[str, Any]) -> None:
        if self._run is None:
            return
        self._run.log(values)

    def log_image(self, key: str, path: str | Path) -> None:
        if self._run is None:
            return
        try:
            import wandb

            self._run.log({key: wandb.Image(str(path))})
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to log image %s: %s", path, exc)

    def finish(self) -> None:
        if self._run is not None:
            _wandb_finish()


def build_reporter(
    cfg: Any,
    *,
    output_dir: str | Path,
    name: str,
    job_type: str,
    extra_tags: tuple[str, ...] = (),
) -> Reporter:
    """Build a post-hoc reporter from config, or a :class:`NullReporter`.

    Returns a :class:`NullReporter` when tracking is disabled or the tracker
    fails to initialize, so callers never branch on ``None``.
    """
    if not _tracking_enabled(cfg):
        return NullReporter()
    try:
        import wandb

        run = wandb.init(
            project=cfg_select(cfg, "wandb.project", default="oceanpath_v2"),
            entity=cfg_select(cfg, "wandb.entity", default=None),
            group=cfg_select(cfg, "wandb.group", default=cfg_select(cfg, "exp_name", default=None)),
            name=name,
            job_type=job_type,
            tags=[*extra_tags, *_standard_tags(cfg)],
            dir=str(output_dir),
            config=_resolved_config(cfg),
            reinit=True,
            mode="offline" if bool(cfg_select(cfg, "wandb.offline", default=False)) else "online",
        )
        logger.info("W&B %s run initialized: %s", job_type, run.url)
        return WandbReporter(run)
    except Exception as exc:  # noqa: BLE001
        logger.warning("W&B init failed: %s. Proceeding without tracking.", exc)
        return NullReporter()


# ── Training-time Lightning logger ──────────────────────────────────────────────


def build_lightning_logger(
    cfg: Any,
    *,
    save_dir: str | Path,
    name: str,
    extra_tags: tuple[str, ...] = (),
):
    """Build the Lightning logger for training-time metric streaming.

    Returns ``None`` when tracking is disabled or the logger fails to
    initialize, so ``Trainer(logger=...)`` falls back to Lightning's default.
    """
    if not _tracking_enabled(cfg):
        return None
    try:
        from lightning.pytorch.loggers import WandbLogger

        return WandbLogger(
            project=cfg_select(cfg, "wandb.project", default="oceanpath_v2"),
            entity=cfg_select(cfg, "wandb.entity", default=None),
            group=cfg_select(cfg, "wandb.group", default=None),
            name=name,
            tags=[*list(cfg_select(cfg, "wandb.tags", default=[]) or []), *extra_tags],
            save_dir=str(save_dir),
            config=_resolved_config(cfg),
            offline=bool(cfg_select(cfg, "wandb.offline", default=False)),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("W&B logger init failed: %s. Training without tracking.", exc)
        return None
