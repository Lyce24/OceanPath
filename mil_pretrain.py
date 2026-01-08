from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    Callback,
)
from lightning.pytorch.loggers import WandbLogger, CSVLogger

# CHANGED: use memmap datamodule
from data.mmap_datamodule import MemmapSSLDataModule

from modules.mil_pretrain import BagSSLModule, SSLConfig


# =============================================================================
# Callbacks
# =============================================================================
class SSLMetricsCallback(Callback):
    """Log additional SSL-specific metrics."""

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: BagSSLModule) -> None:
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0]["lr"]
            pl_module.log("train/lr", lr)

        if hasattr(pl_module, "ema_decay"):
            pl_module.log("train/ema_decay", pl_module.ema_decay)


class RepresentationCollapseMonitor(Callback):
    """Monitor for representation collapse."""

    def __init__(self, variance_threshold: float = 0.01):
        self.variance_threshold = variance_threshold

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: BagSSLModule) -> None:
        metrics = trainer.callback_metrics
        repr_std = metrics.get("val/ssl/repr_std")

        if repr_std is not None and float(repr_std) < self.variance_threshold:
            print(f"\n⚠️  WARNING: Representation std ({float(repr_std):.4f}) below threshold!")
            print("   Consider: reducing augmentation, checking loss, or using VICReg\n")


# =============================================================================
# Training Function
# =============================================================================
def train_ssl(
    config: SSLConfig,
    wandb_run: Optional[Any] = None,
) -> Tuple[BagSSLModule, str]:
    """
    Train bag-level SSL model using MemmapSSLDataModule.
    """
    # CHANGED: workers=True so numpy/random in dataloader workers get seeded too
    pl.seed_everything(config.seed, workers=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Data module
    aug_config = config.get_augmentation_config()

    # CHANGED: MemmapSSLDataModule expects data_dir (memmap folder)
    # You need config.data_dir pointing to a folder containing:
    #   features.bin and index_arrays.npz
    dm = MemmapSSLDataModule(
        data_dir="./src/univ2_mmap/",  # <-- NEW: memmap directory
        augmentation_config=aug_config,
        augmentation_strength=config.augmentation_strength,
        asymmetric=(config.ssl_method.lower() == "byol"),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_instances=config.max_instances,
        pin_memory=config.pin_memory,
        use_padded_collate=getattr(config, "use_padded_collate", False),
        val_fraction=config.val_fraction,
        seed=config.seed,
        return_key=False,
        verbose=True,
        prefetch_factor=2,  # recommended default=1
    )

    dm.prepare_data()
    dm.setup()

    # Infer input_dim from data
    if hasattr(dm, "feature_dim") and dm.feature_dim is not None:
        if config.input_dim != dm.feature_dim:
            print(f"[train_ssl] Updating input_dim: {config.input_dim} → {dm.feature_dim}")
            config.input_dim = dm.feature_dim

    # Print configuration
    print(f"\n{'='*60}")
    print("SSL Training Configuration")
    print(f"{'='*60}")
    print(f"SSL Method:    {config.ssl_method.upper()}")
    print(f"Aggregator:    {config.aggregator.upper()}")
    print(f"Train slides:  {dm.num_train_samples}")
    print(f"Val slides:    {dm.num_val_samples}")
    print(f"Input dim:     {config.input_dim}")
    print(f"Embed dim:     {config.embed_dim}")
    print(f"Proj dim:      {config.proj_dim}")
    print(f"Batch size:    {config.batch_size}")
    print(f"Max epochs:    {config.max_epochs}")
    print(f"Max instances: {config.max_instances}")
    print(f"Prefetch:      {getattr(config, 'prefetch_factor', 1)}")
    print(f"{'='*60}\n")

    # Model
    module = BagSSLModule(config)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename=f"ssl_{config.ssl_method}_{config.aggregator}_{{epoch:03d}}_{{val/loss:.4f}}",
        monitor="val/loss",
        mode="min",
        save_top_k=config.save_top_k,
        save_last=True,
    )

    callbacks = [
        checkpoint_callback,
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=config.early_stopping_patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        SSLMetricsCallback(),
        RepresentationCollapseMonitor(variance_threshold=0.01),
    ]

    # Loggers
    loggers = [CSVLogger(save_dir=config.checkpoint_dir, name="logs")]

    if config.use_wandb:
        if wandb_run is not None:
            loggers.append(WandbLogger(experiment=wandb_run))
        else:
            try:
                import wandb
                run_name = config.wandb_name or f"ssl_{config.ssl_method}_{config.aggregator}"
                wandb_run = wandb.init(
                    project=config.wandb_project,
                    name=run_name,
                    config=config.to_dict(),
                )
                loggers.append(WandbLogger(experiment=wandb_run))
            except ImportError:
                print("wandb not installed, skipping wandb logging")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        callbacks=callbacks,
        logger=loggers,
        precision="16-mixed",  # keep; memmap returns fp16 if stored fp16
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        log_every_n_steps=config.log_every_n_steps,
        enable_progress_bar=True,
        deterministic=False,
    )

    # Train
    trainer.fit(module, dm)

    # Load best checkpoint
    best_path = checkpoint_callback.best_model_path
    if best_path:
        print(f"\n✓ Loading best checkpoint: {best_path}")
        module = BagSSLModule.load_from_checkpoint(best_path, config=config)

    return module, best_path


# =============================================================================
# Feature Extraction (Memmap)
# =============================================================================
@torch.no_grad()
def extract_ssl_features_memmap(
    module: BagSSLModule,
    data_dir: str,
    max_instances: int = 8000,
    precision: int = 16,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Extract bag-level representations using the memmap store directly.
    Reads slide_ids from index_arrays.npz.
    Uses contiguous-window subsampling when length > max_instances (I/O-friendly).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device).eval()

    index = np.load(os.path.join(data_dir, "index_arrays.npz"), allow_pickle=True)
    offsets = index["offsets"]
    lengths = index["lengths"]
    slide_ids = index["slide_ids"]

    feat_dim = int(index["feat_dim"].item() if np.ndim(index["feat_dim"]) == 0 else index["feat_dim"])
    dtype_arr = index["dtype"]
    dtype_str = str(dtype_arr.item() if np.ndim(dtype_arr) == 0 else dtype_arr[0])
    total_patches = int(index["total_patches"].item() if np.ndim(index["total_patches"]) == 0 else index["total_patches"])

    data = np.memmap(
        os.path.join(data_dir, "features.bin"),
        dtype=dtype_str,
        mode="r",
        shape=(total_patches, feat_dim),
    )

    valid = [i for i in range(len(offsets)) if int(offsets[i]) >= 0 and int(lengths[i]) > 0]

    from tqdm import tqdm
    it = tqdm(valid, desc="Extracting(memmap)") if verbose else valid

    out: Dict[str, np.ndarray] = {}

    for i in it:
        offset = int(offsets[i])
        length = int(lengths[i])
        sid = str(slide_ids[i])

        k = min(length, max_instances) if max_instances else length
        if k < length:
            # I/O-friendly contiguous window
            start = np.random.randint(0, length - k + 1)
            x_np = np.array(data[offset + start : offset + start + k], copy=True)
        else:
            x_np = np.array(data[offset : offset + length], copy=True)

        bag = torch.from_numpy(x_np)
        bag = bag.half() if precision == 16 else bag.float()

        # model expects [B, N, D] like your old extractor
        bag = bag.unsqueeze(0).to(device, non_blocking=True)

        z = module(bag)
        out[sid] = z.detach().cpu().numpy().squeeze()

    if verbose:
        print(f"✓ Extracted features for {len(out)} slides (memmap)")

    return out


# =============================================================================
# Example config + run
# =============================================================================
# NOTE: You MUST add `data_dir` to SSLConfig (memmap folder), e.g.:
#   ./src/univ2_mmap/
# containing features.bin + index_arrays.npz
config = SSLConfig(
    aggregator="abmil",
    ssl_method="vicreg",
    temperature=0.1,
    max_epochs=100,
    batch_size=128,
    lr=1e-3,
    num_workers=12,
    augmentation_strength="medium",
    use_wandb=True,
    wandb_project="wsi-ssl",
    checkpoint_dir="checkpoints/ssl",
    seed=42,
)

module, best_path = train_ssl(config)

print(f"\n{'='*60}")
print("Training complete!")
print(f"Best checkpoint: {best_path}")
print(f"{'='*60}")
