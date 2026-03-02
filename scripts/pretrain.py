"""
SSL pretraining orchestrator.

Pretrains a MIL aggregator on unlabeled slide features using one of:
VICReg, SimCLR, BYOL, DINO, or JEPA.

Quality is monitored via RankMe and alpha-ReQ callbacks (no labels needed).

After pretraining, the aggregator checkpoint is used by Stage 5:
    python scripts/train.py ... training.aggregator_weights_path=outputs/pretrain/.../best.ckpt

Usage:
    # VICReg (simplest, recommended starting point)
    python scripts/pretrain.py platform=local data=blca encoder=uni2 model=abmil

    # BYOL with TransMIL
    python scripts/pretrain.py ... model=transmil pretrain_training.ssl_method=byol

    # DINO with more prototypes
    python scripts/pretrain.py ... pretrain_training.ssl_method=dino pretrain_training.ssl.n_prototypes=8192

    # Dry run
    python scripts/pretrain.py ... dry_run=true
"""

import gc
import json
import logging
import time
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from oceanpath.data.pretrain_datamodule import PretrainDataModule
from oceanpath.ssl.callbacks import SSLQualityCallback
from oceanpath.ssl.pretrain_module import SSLPretrainModule

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
    return Path(cfg.platform.output_root) / "pretrain" / cfg.exp_name


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
        return str(Path(mmap_root) / cfg.data.name / cfg.encoder.name)

    # Legacy fallback for older configs.
    features_root = OmegaConf.select(
        cfg,
        "platform.features_root",
        default=OmegaConf.select(cfg, "platform.feature_root", default=None),
    )
    if features_root:
        return str(Path(features_root) / cfg.data.name / cfg.encoder.name / "mmap")

    raise ValueError(
        "Could not resolve mmap directory from cfg.mmap_dir, cfg.data.mmap_dir, or platform paths"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


@hydra.main(config_path="../configs", config_name="pretrain", version_base="1.3")
def main(cfg: DictConfig) -> None:
    _setup_logging(cfg)

    t_cfg = cfg.pretrain_training
    output_dir = _get_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Save resolved config ──────────────────────────────────────────────
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, str(config_path))
    logger.info(f"Config saved to {config_path}")

    logger.info(
        f"SSL Pretraining: method={t_cfg.ssl_method}, "
        f"arch={cfg.model.arch}, data={cfg.data.name}, "
        f"encoder={cfg.encoder.name}"
    )

    # ── Dry run: validate config and exit ─────────────────────────────────
    if cfg.get("dry_run", False):
        logger.info("[DRY RUN] Config validated. Exiting.")
        print(OmegaConf.to_yaml(cfg))
        return

    # ── Seed ──────────────────────────────────────────────────────────────
    L.seed_everything(t_cfg.seed, workers=True)

    # ── Data ──────────────────────────────────────────────────────────────
    mmap_dir = _get_mmap_dir(cfg)
    logger.info(f"Mmap directory: {mmap_dir}")

    pretrain_csv = OmegaConf.select(cfg, "pretrain_csv", default=None)
    if pretrain_csv is None:
        pretrain_csv = OmegaConf.select(cfg, "data.csv_path", default=None)

    pretrain_filename_column = OmegaConf.select(cfg, "pretrain_filename_column", default=None)
    if pretrain_filename_column is None:
        pretrain_filename_column = OmegaConf.select(cfg, "data.filename_column", default="slide_id")

    # ── Batching strategy ────────────────────────────────────────────────
    batching_cfg_raw = OmegaConf.select(t_cfg, "batching", default={})
    batching_cfg_dict = (
        OmegaConf.to_container(batching_cfg_raw, resolve=True) if batching_cfg_raw else {}
    )
    batching_strategy = batching_cfg_dict.pop("strategy", "pad_to_max_in_batch")

    dm = PretrainDataModule(
        mmap_dir=mmap_dir,
        csv_path=pretrain_csv,
        filename_column=pretrain_filename_column,
        split_manifest_path=OmegaConf.select(cfg, "pretrain_split_manifest", default=None),
        augmentation_cfg=OmegaConf.to_container(t_cfg.augmentation, resolve=True),
        coords_aware=t_cfg.coords_aware,
        batch_size=t_cfg.batch_size,
        max_instances=t_cfg.max_instances,
        dataset_max_instances=t_cfg.dataset_max_instances,
        dataset_pre_cap_mode=OmegaConf.select(t_cfg, "dataset_pre_cap_mode", default="random"),
        num_workers=t_cfg.num_workers,
        prefetch_factor=OmegaConf.select(t_cfg, "prefetch_factor", default=2),
        pin_memory=OmegaConf.select(t_cfg, "pin_memory", default=None),
        persistent_workers=OmegaConf.select(t_cfg, "persistent_workers", default=None),
        val_frac=t_cfg.val_frac,
        seed=t_cfg.seed,
        batching_strategy=batching_strategy,
        batching_cfg=batching_cfg_dict,
        force_float32=OmegaConf.select(t_cfg, "force_float32", default=False),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    in_dim = cfg.encoder.feature_dim
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    ssl_cfg = OmegaConf.to_container(t_cfg.ssl, resolve=True)

    module = SSLPretrainModule(
        ssl_method=t_cfg.ssl_method,
        arch=cfg.model.arch,
        in_dim=in_dim,
        model_cfg=model_cfg,
        ssl_cfg=ssl_cfg,
        lr=t_cfg.lr,
        weight_decay=t_cfg.weight_decay,
        warmup_epochs=t_cfg.warmup_epochs,
        max_epochs=t_cfg.max_epochs,
        lr_scheduler=t_cfg.lr_scheduler,
    )

    logger.info(
        f"Model params: {sum(p.numel() for p in module.parameters()):,} total, "
        f"{sum(p.numel() for p in module.parameters() if p.requires_grad):,} trainable"
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        # Checkpoint: save best by val loss
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="best-{epoch:03d}-{val/loss:.4f}",
            monitor=t_cfg.monitor_metric,
            mode=t_cfg.monitor_mode,
            save_top_k=t_cfg.save_top_k,
            save_last=t_cfg.save_last,
            auto_insert_metric_name=False,
        ),
        # LR monitoring
        LearningRateMonitor(logging_interval="epoch"),
        # SSL quality: RankMe + alpha-ReQ
        SSLQualityCallback(
            compute_every_n_epochs=t_cfg.rankme_every_n_epochs,
            max_samples=t_cfg.rankme_max_samples,
        ),
        # Progress bar
        RichProgressBar(),
    ]

    # ── Logger ────────────────────────────────────────────────────────────
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("offline", False):
        pl_logger = None
    else:
        try:
            pl_logger = WandbLogger(
                project=wandb_cfg.get("project", "oceanpath-pretrain"),
                entity=wandb_cfg.get("entity"),
                name=cfg.exp_name,
                group=wandb_cfg.get("group"),
                tags=list(wandb_cfg.get("tags", [])),
                save_dir=str(output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception as e:
            logger.warning(f"W&B init failed: {e}. Training without W&B.")
            pl_logger = None

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=t_cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=t_cfg.precision,
        gradient_clip_val=t_cfg.gradient_clip_val,
        accumulate_grad_batches=t_cfg.accumulate_grad_batches,
        callbacks=callbacks,
        logger=pl_logger,
        default_root_dir=str(output_dir),
        enable_checkpointing=True,
        deterministic=t_cfg.get("deterministic", False),
        log_every_n_steps=10,
    )

    # ── Compile (optional) ────────────────────────────────────────────────
    if t_cfg.get("compile_model", False) and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile...")
        module = torch.compile(module)

    # ── Train ─────────────────────────────────────────────────────────────
    t_start = time.time()
    trainer.fit(module, datamodule=dm)
    elapsed = time.time() - t_start

    logger.info(f"Training complete in {elapsed / 60:.1f} minutes")

    # ── Save metadata ─────────────────────────────────────────────────────
    best_ckpt = trainer.checkpoint_callback.best_model_path
    best_score = trainer.checkpoint_callback.best_model_score

    metadata = {
        "ssl_method": t_cfg.ssl_method,
        "arch": cfg.model.arch,
        "encoder": cfg.encoder.name,
        "dataset": cfg.data.name,
        "in_dim": in_dim,
        "embed_dim": model_cfg.get("embed_dim", 512),
        "best_checkpoint": str(best_ckpt),
        "best_val_loss": float(best_score) if best_score is not None else None,
        "max_epochs": t_cfg.max_epochs,
        "actual_epochs": trainer.current_epoch,
        "total_time_minutes": elapsed / 60,
        "seed": t_cfg.seed,
        "split_manifest": OmegaConf.select(cfg, "pretrain_split_manifest", default=None),
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {meta_path}")

    # ── Extract aggregator-only checkpoint ────────────────────────────────
    if best_ckpt and Path(best_ckpt).exists():
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        agg_state = ckpt.get("aggregator_state_dict", {})
        if agg_state:
            agg_path = output_dir / "aggregator_weights.pt"
            torch.save(agg_state, str(agg_path))
            logger.info(
                f"Aggregator weights saved to {agg_path} "
                f"({sum(v.numel() for v in agg_state.values()):,} params)"
            )
            logger.info(f"Use in Stage 5: training.aggregator_weights_path={agg_path}")

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
