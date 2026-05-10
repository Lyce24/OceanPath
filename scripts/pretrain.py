"""
SSL pretraining orchestrator.

Pretrains a MIL aggregator on unlabeled slide features using one of four
slide-level SSL objectives:

    - VICReg      (var/inv/cov, no EMA, no predictor)
    - JEPA        (symmetric predictive MSE/Huber to EMA target)
    - LeJEPA-2C   (predictive-to-centroid + SIGReg, dual view)
    - LeJEPA-MC   (multi-crop generalization of LeJEPA, V_g + V_l views)

Method-specific configs live in `configs/pretrain_training/{vicreg,jepa,
lejepa_2c,lejepa_mc}.yaml` and override the shared defaults in `base.yaml`.

Recommended first run:
    python scripts/pretrain.py pretrain_training=vicreg

After pretraining, the aggregator checkpoint can be used downstream via:
    training.aggregator_weights_path=outputs/pretrain/<exp_name>/aggregator_weights.pt
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, ListConfig, OmegaConf

from oceanpath.ssl.data import PretrainDataModule
from oceanpath.ssl.modules import (
    LinearProbeEvalCallback,
    LinearProbeTask,
    SSLPretrainModule,
    SSLQualityCallback,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _setup_logging(cfg: DictConfig) -> None:
    level = logging.DEBUG if cfg.get("verbose", False) else logging.INFO
    exp_name = OmegaConf.select(cfg, "exp_name", default="pretrain")
    fmt = f"%(asctime)s | %(levelname)-7s | exp={exp_name} | %(message)s"
    logging.basicConfig(level=level, format=fmt, force=True)


def _get_output_dir(cfg: DictConfig) -> Path:
    output_dir = OmegaConf.select(cfg, "output_dir", default=None)
    if output_dir:
        return Path(str(output_dir))

    output_root = OmegaConf.select(cfg, "platform.output_root", default="outputs")
    exp_name = OmegaConf.select(cfg, "exp_name", default="pretrain")
    return Path(str(output_root)) / "pretrain" / str(exp_name)


def _get_mmap_dir(cfg: DictConfig) -> str:
    """Resolve mmap directory from config with robust fallbacks."""
    mmap_dir = OmegaConf.select(cfg, "mmap_dir", default=None)
    if mmap_dir:
        return str(mmap_dir)

    mmap_dir = OmegaConf.select(cfg, "data.mmap_dir", default=None)
    if mmap_dir:
        return str(mmap_dir)

    mmap_root = OmegaConf.select(cfg, "platform.mmap_root", default=None)
    if mmap_root:
        return str(Path(str(mmap_root)) / str(cfg.data.name) / str(cfg.encoder.name))

    features_root = OmegaConf.select(
        cfg,
        "platform.features_root",
        default=OmegaConf.select(cfg, "platform.feature_root", default=None),
    )
    if features_root:
        return str(Path(str(features_root)) / str(cfg.data.name) / str(cfg.encoder.name) / "mmap")

    raise ValueError(
        "Could not resolve mmap directory. Set one of: "
        "cfg.mmap_dir, cfg.data.mmap_dir, cfg.platform.mmap_root, "
        "or cfg.platform.features_root."
    )


def _to_plain_dict(x: Any) -> dict:
    if x is None:
        return {}
    if isinstance(x, DictConfig):
        return OmegaConf.to_container(x, resolve=True)
    return dict(x)


def _maybe_get(cfg: DictConfig, key: str, default: Any = None) -> Any:
    return OmegaConf.select(cfg, key, default=default)


def _trainer_setting(cfg: DictConfig, key: str, default: Any = None) -> Any:
    """Resolve trainer setting with pretrain overrides before platform defaults."""
    value = OmegaConf.select(cfg, f"pretrain_training.{key}", default=None)
    if value is not None:
        return value
    value = OmegaConf.select(cfg, f"platform.{key}", default=None)
    if value is not None:
        return value
    return default


def _auto_device_count(accelerator: Any) -> int:
    accelerator_s = str(accelerator).lower()
    if accelerator_s in {"gpu", "cuda"} or (accelerator_s == "auto" and torch.cuda.is_available()):
        return max(1, torch.cuda.device_count())
    return 1


def _infer_num_devices(devices: Any, accelerator: Any) -> int:
    """Infer Lightning's configured device count before the Trainer launches DDP."""
    if isinstance(devices, (list, tuple, ListConfig)):
        return max(1, len(devices))

    if isinstance(devices, int):
        return _auto_device_count(accelerator) if devices == -1 else max(1, devices)

    devices_s = str(devices).strip().lower()
    if devices_s in {"", "none", "auto"}:
        return _auto_device_count(accelerator)
    if devices_s == "-1":
        return _auto_device_count(accelerator)
    if "," in devices_s:
        return max(1, len([d for d in devices_s.split(",") if d.strip()]))

    try:
        parsed = int(devices_s)
    except ValueError:
        return 1
    return _auto_device_count(accelerator) if parsed == -1 else max(1, parsed)


def _resolve_batch_sizes(
    cfg: DictConfig,
    t_cfg: DictConfig,
) -> tuple[int, int, int, int]:
    """Return (global_loss_batch, per_rank_batch, world_size, optimizer_batch).

    ``pretrain_training.batch_size`` is intentionally interpreted as the
    global/effective VICReg loss batch across DDP ranks. The DataLoader still
    needs a per-rank batch size, so we derive it here from the configured
    Trainer device count.
    """
    global_loss_batch = int(t_cfg.batch_size)
    accumulate = int(_maybe_get(t_cfg, "accumulate_grad_batches", 1))

    accelerator = _trainer_setting(cfg, "accelerator", "auto")
    devices = _trainer_setting(cfg, "devices", "auto")
    num_nodes = int(_trainer_setting(cfg, "num_nodes", 1))
    world_size = _infer_num_devices(devices, accelerator) * max(1, num_nodes)

    if global_loss_batch < world_size:
        raise ValueError(
            "pretrain_training.batch_size is the global/effective loss batch, "
            f"but got batch_size={global_loss_batch} < configured world_size={world_size}. "
            "Increase batch_size or reduce devices/num_nodes."
        )
    if global_loss_batch % world_size != 0:
        raise ValueError(
            "pretrain_training.batch_size must divide evenly by the configured "
            f"DDP world size. Got batch_size={global_loss_batch}, world_size={world_size}. "
            "Choose a divisible effective loss batch so every rank gets the same "
            "per-rank micro-batch."
        )

    per_rank_batch = global_loss_batch // world_size
    optimizer_effective_batch = global_loss_batch * max(1, accumulate)
    return global_loss_batch, per_rank_batch, world_size, optimizer_effective_batch


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
@hydra.main(config_path="../configs", config_name="pretrain", version_base="1.3")
def main(cfg: DictConfig) -> None:
    _setup_logging(cfg)

    t_cfg = cfg.pretrain_training
    output_dir = _get_output_dir(cfg)
    # Append a timestamp so every run gets its own subdirectory —
    # prevents checkpoint/config collisions across restarts.
    from datetime import datetime

    output_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config.
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, str(config_path))
    logger.info("Config saved to %s", config_path)

    logger.info(
        "SSL pretraining: method=%s, arch=%s, data=%s, encoder=%s",
        t_cfg.ssl_method,
        cfg.model.arch,
        cfg.data.name,
        cfg.encoder.name,
    )

    global_loss_batch, per_rank_batch, configured_world_size, optimizer_batch = (
        _resolve_batch_sizes(cfg, t_cfg)
    )
    logger.info(
        "Batch size: global_loss_batch=%d, per_rank_loader_batch=%d, "
        "configured_world_size=%d, accumulate_grad_batches=%d, "
        "optimizer_effective_batch=%d",
        global_loss_batch,
        per_rank_batch,
        configured_world_size,
        int(_maybe_get(t_cfg, "accumulate_grad_batches", 1)),
        optimizer_batch,
    )

    # Dry run.
    if cfg.get("dry_run", False):
        logger.info("[DRY RUN] Config validated. Exiting.")
        print(OmegaConf.to_yaml(cfg))
        return

    # Seed.
    L.seed_everything(int(t_cfg.seed), workers=True)
    torch.set_float32_matmul_precision("high")

    # ── Data ──────────────────────────────────────────────────────────────
    mmap_dir = _get_mmap_dir(cfg)
    logger.info("Mmap directory: %s", mmap_dir)

    pretrain_csv = _maybe_get(cfg, "pretrain_csv", None) or _maybe_get(cfg, "data.csv_path", None)
    filename_column = _maybe_get(cfg, "pretrain_filename_column", None) or _maybe_get(
        cfg, "data.filename_column", "slide_id"
    )

    augmentation_cfg = _to_plain_dict(_maybe_get(t_cfg, "augmentation", default={}))

    dm = PretrainDataModule(
        mmap_dir=mmap_dir,
        fixed_n=int(t_cfg.fixed_n),
        batch_size=per_rank_batch,
        csv_path=pretrain_csv,
        filename_column=str(filename_column),
        split_manifest_path=_maybe_get(cfg, "pretrain_split_manifest", None),
        ssl_method=str(t_cfg.ssl_method),
        augmentation_cfg=augmentation_cfg,
        dataset_max_instances=_maybe_get(t_cfg, "dataset_max_instances", 12000),
        dataset_pre_cap_mode=_maybe_get(t_cfg, "dataset_pre_cap_mode", "spatial_stratified"),
        num_workers=int(t_cfg.num_workers),
        prefetch_factor=int(_maybe_get(t_cfg, "prefetch_factor", 4)),
        pin_memory=_maybe_get(t_cfg, "pin_memory", None),
        persistent_workers=_maybe_get(t_cfg, "persistent_workers", None),
        val_frac=float(t_cfg.val_frac),
        seed=int(t_cfg.seed),
        force_float32=bool(_maybe_get(t_cfg, "force_float32", False)),
        coords_aware=_maybe_get(t_cfg, "coords_aware", True),
        manifest_min_patches=_maybe_get(
            t_cfg,
            "manifest_min_patches",
            int(t_cfg.fixed_n),  # safe default: matches fixed_n
        ),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    in_dim = int(cfg.encoder.feature_dim)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    ssl_cfg = OmegaConf.to_container(t_cfg.ssl, resolve=True)

    module = SSLPretrainModule(
        ssl_method=str(t_cfg.ssl_method),
        arch=str(cfg.model.arch),
        in_dim=in_dim,
        model_cfg=model_cfg,
        ssl_cfg=ssl_cfg,
        lr=float(t_cfg.lr),
        weight_decay=float(t_cfg.weight_decay),
        adam_betas=tuple(_maybe_get(t_cfg, "adam_betas", (0.9, 0.95))),
        warmup_epochs=int(t_cfg.warmup_epochs),
        max_epochs=int(t_cfg.max_epochs),
        lr_scheduler=str(t_cfg.lr_scheduler),
    )

    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    logger.info(
        "Model params: %s total, %s trainable", f"{total_params:,}", f"{trainable_params:,}"
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    monitor_metric = str(t_cfg.monitor_metric)
    monitor_mode = str(t_cfg.monitor_mode)

    # Detect monitoring of an LP composite by comparing against the
    # configured composite_metric.name (more robust than hardcoded
    # prefix checks). The composite is only logged on LP-eval epochs,
    # so ModelCheckpoint must skip silently on intermediate val ends.
    lp_eval_cfg = OmegaConf.select(t_cfg, "linear_probe_eval", default=None)
    composite_name_cfg = (
        OmegaConf.select(lp_eval_cfg, "composite_metric.name", default=None)
        if lp_eval_cfg is not None
        else None
    )
    monitors_lp_composite = composite_name_cfg is not None and monitor_metric == str(
        composite_name_cfg
    )

    checkpoint_filename = "best-{epoch:03d}-{val/loss:.4f}"
    if monitor_metric == "ssl/rankme_ratio":
        checkpoint_filename = "best-{epoch:03d}-{val/loss:.4f}-{ssl/rankme_ratio:.3f}"
    elif monitors_lp_composite:
        checkpoint_filename = "best-{epoch:03d}-{val/loss:.4f}-{" + monitor_metric + ":.4f}"

    checkpoint_kwargs: dict[str, Any] = {
        "dirpath": str(output_dir / "checkpoints"),
        "filename": checkpoint_filename,
        "monitor": monitor_metric,
        "mode": monitor_mode,
        "save_top_k": int(t_cfg.save_top_k),
        "save_last": bool(t_cfg.save_last),
        "auto_insert_metric_name": False,
    }
    # Always trigger at validation end (after LinearProbeEvalCallback has
    # logged), never on train_epoch_end where the LP composite isn't yet
    # in callback_metrics.
    checkpoint_kwargs["save_on_train_epoch_end"] = False
    # Intentionally do NOT pass `every_n_epochs` to ModelCheckpoint when
    # monitoring an LP composite. Lightning's `every_n_epochs=N` saves
    # at the end of epoch N-1 (because it uses (current_epoch+1) % N),
    # but LinearProbeEvalCallback runs at epoch N (uses current_epoch % N).
    # Setting `every_n_epochs=5` would therefore demand the monitor one
    # epoch *before* LP eval has produced it. Instead let ModelCheckpoint
    # check at every val end and silently skip when the monitor is absent
    # (handled by the subclass below).

    if monitors_lp_composite:
        # The LP composite only lands in callback_metrics on epochs where
        # LinearProbeEvalCallback ran. On every other val_end hook the
        # key is missing and stock ModelCheckpoint raises. We override
        # the public `on_validation_end` hook (more stable across
        # Lightning versions than `_save_topk_checkpoint`) so the top-k
        # save is skipped when the monitor key isn't present, while the
        # save_last path still runs every epoch.
        class _LPMonitorCheckpoint(ModelCheckpoint):
            def on_validation_end(self, trainer, pl_module):
                # Borrow the parent's preconditions (sanity-check skips,
                # train-epoch-end gating, etc.) by replicating just the
                # decision logic instead of calling super().
                if self._should_skip_saving_checkpoint(trainer):
                    return
                if self._should_save_on_train_epoch_end(trainer):
                    return
                monitor_candidates = self._monitor_candidates(trainer)

                # Top-k: only when the monitor is available on EVERY rank.
                # If ranks disagree here, one rank can enter Lightning's
                # top-k checkpoint collectives while another skips straight to
                # save_last / the next train loop, producing an NCCL collective
                # mismatch. Synchronize the boolean before deciding.
                monitor_available = self.monitor is None or self.monitor in monitor_candidates
                if (
                    getattr(trainer, "world_size", 1) > 1
                    and torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    flag = torch.tensor(
                        int(monitor_available),
                        device=pl_module.device,
                        dtype=torch.int32,
                    )
                    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
                    monitor_available = bool(flag.item())

                if monitor_available:
                    self._save_topk_checkpoint(trainer, monitor_candidates)
                # save_last: always (independent of monitor).
                self._save_last_checkpoint(trainer, monitor_candidates)

        checkpoint_cb = _LPMonitorCheckpoint(**checkpoint_kwargs)
    else:
        checkpoint_cb = ModelCheckpoint(**checkpoint_kwargs)

    callbacks = [
        checkpoint_cb,
        LearningRateMonitor(logging_interval="epoch"),
        SSLQualityCallback(
            compute_every_n_epochs=int(t_cfg.rankme_every_n_epochs),
            max_samples=int(t_cfg.rankme_max_samples),
        ),
        RichProgressBar(),
    ]

    # Defensive sanity check: monitoring an LP composite without LP eval
    # enabled means the metric will never be logged → save_top_k will
    # never fire and only `last.ckpt` would be written. Fail loud.
    if monitors_lp_composite and (
        lp_eval_cfg is None or not bool(lp_eval_cfg.get("enabled", False))
    ):
        raise ValueError(
            f"monitor_metric='{monitor_metric}' requires "
            f"linear_probe_eval.enabled=true (the composite metric is only "
            f"computed when LP eval runs)."
        )

    if lp_eval_cfg is not None and bool(lp_eval_cfg.get("enabled", False)):
        lp_tasks: list[LinearProbeTask] = []

        for task_cfg in lp_eval_cfg.tasks:
            lp_tasks.append(
                LinearProbeTask(
                    name=str(task_cfg.name),
                    protocol=str(task_cfg.protocol),
                    mmap_dir=str(task_cfg.mmap_dir),
                    manifest_csv=str(task_cfg.manifest_csv),
                    label_column=str(task_cfg.label_column),
                    patient_column=str(task_cfg.patient_column),
                    filename_column=str(task_cfg.get("filename_column", "filename")),
                    slide_id_column=task_cfg.get("slide_id_column", None),
                    split_column=task_cfg.get("split_column", None),
                    test_mmap_dir=task_cfg.get("test_mmap_dir", None),
                    test_manifest_csv=task_cfg.get("test_manifest_csv", None),
                    test_label_column=task_cfg.get("test_label_column", None),
                    test_patient_column=task_cfg.get("test_patient_column", None),
                    test_filename_column=task_cfg.get("test_filename_column", None),
                    test_slide_id_column=task_cfg.get("test_slide_id_column", None),
                    n_splits=int(task_cfg.get("n_splits", 5)),
                    inner_splits=int(task_cfg.get("inner_splits", 3)),
                    primary_metric=task_cfg.get("primary_metric", None),
                    c_grid=tuple(
                        task_cfg.get(
                            "c_grid",
                            [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
                        )
                    ),
                    class_weight=str(task_cfg.get("class_weight", "balanced")),
                    max_iter=int(task_cfg.get("max_iter", 5000)),
                    multi_class_mode=str(task_cfg.get("multi_class_mode", "auto")),
                    train_split_names=tuple(task_cfg.get("train_split_names", ["train", "tr"])),
                    eval_split_names=tuple(task_cfg.get("eval_split_names", ["test", "te"])),
                    max_instances=task_cfg.get("max_instances", None),
                    batch_size=int(task_cfg.get("batch_size", 1)),
                    num_workers=int(task_cfg.get("num_workers", 4)),
                    allow_missing_mmap=bool(task_cfg.get("allow_missing_mmap", False)),
                    allow_missing_manifest=bool(task_cfg.get("allow_missing_manifest", False)),
                    aggregate_to_patient=bool(task_cfg.get("aggregate_to_patient", True)),
                    sampling_mode=str(task_cfg.get("sampling_mode", "contiguous")),
                    sampling_seed=int(task_cfg.get("sampling_seed", 42)),
                    seed=int(t_cfg.seed),
                    wandb_log_metrics=(
                        tuple(task_cfg.wandb_log_metrics)
                        if task_cfg.get("wandb_log_metrics", None) is not None
                        else None
                    ),
                )
            )

        composite_cfg = OmegaConf.select(lp_eval_cfg, "composite_metric", default=None)
        composite_metric_name: str | None = None
        composite_metric_sources: list[str] = []
        composite_metric_weights: list[float] | None = None
        if composite_cfg is not None:
            composite_metric_name = str(composite_cfg.get("name", "lp/composite_score"))

            # weights accepts two equivalent forms in YAML:
            #   - dict {metric_key: weight, ...}      ← preferred
            #   - list of weights, paired with `sources: [...]`
            # Internally we always convert to (sources_list, weights_list).
            raw_weights = composite_cfg.get("weights", None)
            raw_sources = composite_cfg.get("sources", None)

            if raw_weights is not None and isinstance(raw_weights, (dict, DictConfig)):
                weights_dict = (
                    OmegaConf.to_container(raw_weights, resolve=True)
                    if isinstance(raw_weights, DictConfig)
                    else dict(raw_weights)
                )
                if not weights_dict:
                    raise ValueError("linear_probe_eval.composite_metric.weights is empty.")
                composite_metric_sources = [str(k) for k in weights_dict]
                composite_metric_weights = [float(v) for v in weights_dict.values()]
            else:
                # Parallel-list form (legacy).
                composite_metric_sources = [str(s) for s in (raw_sources or [])]
                if not composite_metric_sources:
                    raise ValueError(
                        "linear_probe_eval.composite_metric: provide either a "
                        "weights dict {metric_key: weight} or both `sources` "
                        "and `weights` lists."
                    )
                if raw_weights is not None:
                    composite_metric_weights = [float(w) for w in raw_weights]
                    if len(composite_metric_weights) != len(composite_metric_sources):
                        raise ValueError(
                            f"linear_probe_eval.composite_metric.weights has "
                            f"{len(composite_metric_weights)} entries but sources "
                            f"has {len(composite_metric_sources)}."
                        )

        callbacks.append(
            LinearProbeEvalCallback(
                tasks=lp_tasks,
                every_n_epochs=int(lp_eval_cfg.get("every_n_epochs", 5)),
                output_dir=str(output_dir / "lp_eval"),
                run_at_fit_start=bool(lp_eval_cfg.get("run_at_fit_start", False)),
                run_at_epoch0=bool(lp_eval_cfg.get("run_at_epoch0", False)),
                fail_on_error=bool(lp_eval_cfg.get("fail_on_error", True)),
                composite_metric_name=composite_metric_name,
                composite_metric_sources=composite_metric_sources,
                composite_metric_weights=composite_metric_weights,
            )
        )

    # ── Logger ────────────────────────────────────────────────────────────
    wandb_cfg = cfg.get("wandb", {})
    pl_logger = None

    if not wandb_cfg.get("disabled", False):
        try:
            pl_logger = WandbLogger(
                project=wandb_cfg.get("project", "oceanpath-pretrain"),
                entity=wandb_cfg.get("entity", None),
                name=str(cfg.exp_name),
                group=wandb_cfg.get("group", None),
                tags=list(wandb_cfg.get("tags", [])),
                save_dir=str(output_dir),
                offline=bool(wandb_cfg.get("offline", False)),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception as e:
            logger.warning("W&B init failed: %s. Training without W&B.", e)
            pl_logger = None

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer_accelerator = _trainer_setting(cfg, "accelerator", "auto")
    trainer_devices = _trainer_setting(cfg, "devices", "auto")
    if isinstance(trainer_devices, ListConfig):
        trainer_devices = list(trainer_devices)
    trainer_strategy = _trainer_setting(cfg, "strategy", "auto")
    trainer_num_nodes = int(_trainer_setting(cfg, "num_nodes", 1))

    trainer = L.Trainer(
        max_epochs=int(t_cfg.max_epochs),
        accelerator=str(trainer_accelerator),
        devices=trainer_devices,
        strategy=trainer_strategy,
        num_nodes=trainer_num_nodes,
        precision=str(t_cfg.precision),
        gradient_clip_val=float(t_cfg.gradient_clip_val),
        accumulate_grad_batches=int(t_cfg.accumulate_grad_batches),
        callbacks=callbacks,
        logger=pl_logger,
        default_root_dir=str(output_dir),
        enable_checkpointing=True,
        deterministic=bool(_maybe_get(t_cfg, "deterministic", False)),
        log_every_n_steps=int(_maybe_get(t_cfg, "log_every_n_steps", 10)),
        use_distributed_sampler=False,  # PretrainDataModule handles DDP sampler.
    )

    # Optional compile. Keep off for first debugging run.
    if bool(_maybe_get(t_cfg, "compile_model", False)) and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile...")
        module = torch.compile(module)

    # ── Train ─────────────────────────────────────────────────────────────
    t_start = time.time()
    trainer.fit(module, datamodule=dm)
    elapsed = time.time() - t_start

    logger.info("Training complete in %.1f minutes", elapsed / 60)

    # ── Save metadata ─────────────────────────────────────────────────────
    best_ckpt = checkpoint_cb.best_model_path
    best_score = checkpoint_cb.best_model_score
    best_score_value = (
        float(best_score.detach().cpu().item())
        if hasattr(best_score, "detach")
        else (float(best_score) if best_score is not None else None)
    )

    metadata = {
        "ssl_method": str(t_cfg.ssl_method),
        "arch": str(cfg.model.arch),
        "encoder": str(cfg.encoder.name),
        "dataset": str(cfg.data.name),
        "mmap_dir": str(mmap_dir),
        "in_dim": in_dim,
        "embed_dim": int(model_cfg.get("embed_dim", 512)),
        "fixed_n": int(t_cfg.fixed_n),
        "batch_size": global_loss_batch,
        "global_loss_batch_size": global_loss_batch,
        "per_rank_batch_size": per_rank_batch,
        "configured_world_size": configured_world_size,
        "accumulate_grad_batches": int(t_cfg.accumulate_grad_batches),
        "optimizer_effective_batch_size": optimizer_batch,
        "monitor_metric": monitor_metric,
        "monitor_mode": monitor_mode,
        "best_checkpoint": str(best_ckpt),
        "best_monitor_score": best_score_value,
        "best_val_loss": best_score_value if monitor_metric == "val/loss" else None,
        "max_epochs": int(t_cfg.max_epochs),
        "actual_epochs": int(trainer.current_epoch),
        "total_time_minutes": elapsed / 60,
        "seed": int(t_cfg.seed),
        "split_manifest": _maybe_get(cfg, "pretrain_split_manifest", None),
    }

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Metadata saved to %s", meta_path)

    # ── Extract aggregator-only checkpoint ────────────────────────────────
    if best_ckpt and Path(best_ckpt).exists():
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        agg_state = ckpt.get("aggregator_state_dict", {})

        if agg_state:
            agg_path = output_dir / "aggregator_weights.pt"
            torch.save(agg_state, str(agg_path))
            logger.info(
                "Aggregator weights saved to %s (%s params)",
                agg_path,
                f"{sum(v.numel() for v in agg_state.values()):,}",
            )
            logger.info("Use downstream: training.aggregator_weights_path=%s", agg_path)
        else:
            logger.warning("No aggregator_state_dict found in best checkpoint.")

    # ── Cleanup ───────────────────────────────────────────────────────────
    if pl_logger is not None:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass

    del module, trainer, dm
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Pretraining pipeline complete.")


if __name__ == "__main__":
    main()
