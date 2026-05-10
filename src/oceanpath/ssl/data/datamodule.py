"""
Lightning DataModule for SSL pretraining.

Two batching modes, dispatched on ssl_method:

* **Dual-view** (default; vicreg, jepa, lejepa):
  ``stack_collate`` produces ``view1`` / ``view2`` tensors of shape
  ``[B, fixed_n, D]``.

* **Multi-crop** (lejepa_mc):
  ``multicrop_stack_collate`` produces ``global_views`` / ``local_views``
  tensors of shape ``[V_g, B, N_g, D]`` and ``[V_l, B, N_l, D]``.

Both collators perform a trivial torch.stack — every per-sample shape is
already finalized by the augmentor, so collation costs are minimal.

Supports random split or explicit manifest (train_ids/val_ids JSON).
DDP handled via DistributedSampler. Per-worker RNG seeded so augmentation
is independent across data-loader workers.
"""

import json
import logging
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from oceanpath.data.mmap_builder import validate_mmap_dir
from oceanpath.ssl.data.augmentation import (
    WSIDualViewAugmentor,
    WSIMultiCropAugmentor,
    build_augmentor,
    build_multicrop_augmentor,
)
from oceanpath.ssl.data.dataset import PretrainDataset

logger = logging.getLogger(__name__)

# Methods supported by the data module. Every entry must either map to a
# dual-view augmentor (default behaviour) or a multi-crop augmentor.
_VALID_SSL_METHODS = frozenset(
    {
        "default",
        "vicreg",
        "jepa",  # NEW (slide-level JEPA — aliased to pooled_jepa augmentor)
        "lejepa",
        "lejepa_mc",  # NEW (multi-crop)
    }
)

# Subset that produces the multi-crop sample format.
_MULTICROP_SSL_METHODS = frozenset({"lejepa_mc"})


# =============================================================================
# Collators
# =============================================================================


def stack_collate(batch: list[dict]) -> dict:
    """Stack pre-finalized fixed-size dual-view samples into a batch.

    The dataset finalizes each view to exactly ``fixed_n`` patches (subsample
    or zero-pad) and emits per-view lengths. Collation is therefore a trivial
    stack — no padding logic, no allocation-then-fill, no [B, N] mask tensor
    held in the worker output.

    Masks are derived from lengths via a single broadcast compare. This is
    cheaper than storing [B, N] float32 masks in worker output and avoids
    desynchronized mask/length state.

    Output contract
    ═══════════════════════════════════════════════════════════════════════════
    Per-sample input (from PretrainDataset, dual-view path):
        view1_features, view2_features : [fixed_n, D] tensors
        view1_coords, view2_coords     : [fixed_n, C] tensors (optional)
        view1_length, view2_length     : int (real patches before padding)
        slide_id                       : str

    Batch output:
        view1, view2                   : [B, fixed_n, D] tensors
        mask1, mask2                   : [B, fixed_n]    float tensors (from lengths)
        lengths1, lengths2             : [B]             int32 tensors
        coords1, coords2               : [B, fixed_n, C] tensors (if coords present)
        slide_ids                      : list[str]
        Augmentation diagnostics (only if augmentor returned stats, [B] float32):
            crop_area_frac{1,2}, mask_ratio{1,2}     — what was attempted
            mask_fraction{1,2}, crop_or_better_fraction{1,2},
            view_or_better_fraction{1,2}, full_refill_fraction{1,2},
            replacement_fraction{1,2}                — what the final view became
    """
    if not batch:
        raise ValueError("stack_collate received an empty batch")

    # Cheap contract check: same shape across all samples. Fails fast with
    # a clear message if the dataset wasn't given fixed_n.
    ref_shape = batch[0]["view1_features"].shape
    for i, s in enumerate(batch):
        if s["view1_features"].shape != ref_shape or s["view2_features"].shape != ref_shape:
            raise RuntimeError(
                f"stack_collate requires uniform [fixed_n, D] shape across samples. "
                f"Sample 0 is {tuple(ref_shape)}, sample {i} differs. "
                f"Make sure PretrainDataset was constructed with fixed_n set."
            )

    view1 = torch.stack([s["view1_features"] for s in batch], dim=0)
    view2 = torch.stack([s["view2_features"] for s in batch], dim=0)

    B, N, _ = view1.shape
    lengths1 = torch.full((B,), N, dtype=torch.int32)
    lengths2 = torch.full((B,), N, dtype=torch.int32)
    mask1 = torch.ones(B, N, dtype=torch.float32)
    mask2 = torch.ones(B, N, dtype=torch.float32)

    out = {
        "view1": view1,
        "view2": view2,
        "mask1": mask1,
        "mask2": mask2,
        "lengths1": lengths1,
        "lengths2": lengths2,
        "slide_ids": [s["slide_id"] for s in batch],
    }
    if "view1_coords" in batch[0]:
        out["coords1"] = torch.stack([s["view1_coords"] for s in batch], dim=0)
        out["coords2"] = torch.stack([s["view2_coords"] for s in batch], dim=0)
    # Per-view augmentation diagnostics (only present when the view generator
    # surfaced them — currently the dual-view augmentor with
    # return_stats=True). Stacked into [B] float tensors for `aug/*` logging.
    if "view1_crop_area_frac" in batch[0]:
        for key in (
            "crop_area_frac",
            "mask_ratio",
            "mask_fraction",
            "crop_or_better_fraction",
            "view_or_better_fraction",
            "full_refill_fraction",
            "replacement_fraction",
        ):
            out[f"{key}1"] = torch.tensor([s[f"view1_{key}"] for s in batch], dtype=torch.float32)
            out[f"{key}2"] = torch.tensor([s[f"view2_{key}"] for s in batch], dtype=torch.float32)
    return out


def multicrop_stack_collate(batch: list[dict]) -> dict:
    """Stack multi-crop samples into a batch.

    Output contract
    ═══════════════════════════════════════════════════════════════════════════
    Per-sample input (from PretrainDataset, multi-crop path):
        global_features : list[Tensor[N_g, D]] of length V_g (uniform shape)
        local_features  : list[Tensor[N_l, D]] of length V_l (uniform shape)
        global_coords, local_coords : optional lists of [N, 2] int tensors
        num_global, num_local : ints (must agree across the batch)
        slide_id : str

    Batch output:
        global_views   : Tensor[V_g, B, N_g, D]    float
        local_views    : Tensor[V_l, B, N_l, D]    float (or empty if V_l == 0)
        global_masks   : Tensor[V_g, B, N_g]       float (all ones — fixed_n)
        local_masks    : Tensor[V_l, B, N_l]       float
        global_lengths : Tensor[V_g, B]            int32
        local_lengths  : Tensor[V_l, B]            int32
        global_coords  : Tensor[V_g, B, N_g, 2]    int32 (if present)
        local_coords   : Tensor[V_l, B, N_l, 2]    int32 (if present)
        num_global, num_local : ints
        slide_ids      : list[str]

    The output layout — leading view axis followed by batch — matches the
    LeJEPAMCLoss expectation (a list/tensor of V views, each [B, D]).
    """
    if not batch:
        raise ValueError("multicrop_stack_collate received an empty batch")

    V_g = batch[0]["num_global"]
    V_l = batch[0]["num_local"]

    # Sanity: every sample in the batch must agree on V_g / V_l. The
    # augmentor is constructed once per epoch so this should always hold;
    # the check guards against accidental mid-epoch reconfig.
    for i, s in enumerate(batch):
        if s["num_global"] != V_g or s["num_local"] != V_l:
            raise RuntimeError(
                f"multicrop_stack_collate: sample 0 has "
                f"V_g={V_g}, V_l={V_l}; sample {i} has "
                f"V_g={s['num_global']}, V_l={s['num_local']}"
            )

    # ---- Global views ----
    # Stack per-view across batch first, then stack views into [V, B, N, D].
    # Two stacks (vs one over a flat list) makes the per-view shape contract
    # explicit and produces a clearer traceback if a view's shape diverges.
    per_view_global = []
    for v in range(V_g):
        per_view_global.append(torch.stack([s["global_features"][v] for s in batch], dim=0))
    global_views = torch.stack(per_view_global, dim=0)  # [V_g, B, N_g, D]

    # ---- Local views ----
    if V_l > 0:
        per_view_local = []
        for v in range(V_l):
            per_view_local.append(torch.stack([s["local_features"][v] for s in batch], dim=0))
        local_views = torch.stack(per_view_local, dim=0)  # [V_l, B, N_l, D]
    else:
        # Empty placeholder so downstream code can unconditionally index.
        D = global_views.shape[-1]
        local_views = global_views.new_zeros((0, len(batch), 0, D))

    # ---- Lengths and masks (all-ones because fixed_n) ----
    B = len(batch)
    N_g = global_views.shape[2]
    N_l = local_views.shape[2] if V_l > 0 else 0

    global_lengths = torch.full((V_g, B), N_g, dtype=torch.int32)
    global_masks = torch.ones((V_g, B, N_g), dtype=torch.float32)

    local_lengths = (
        torch.full((V_l, B), N_l, dtype=torch.int32)
        if V_l > 0
        else torch.zeros((0, B), dtype=torch.int32)
    )
    local_masks = (
        torch.ones((V_l, B, N_l), dtype=torch.float32)
        if V_l > 0
        else torch.zeros((0, B, 0), dtype=torch.float32)
    )

    out = {
        "global_views": global_views,
        "local_views": local_views,
        "global_masks": global_masks,
        "local_masks": local_masks,
        "global_lengths": global_lengths,
        "local_lengths": local_lengths,
        "num_global": V_g,
        "num_local": V_l,
        "slide_ids": [s["slide_id"] for s in batch],
    }

    # ---- Optional coords ----
    if "global_coords" in batch[0]:
        per_view_gc = []
        for v in range(V_g):
            per_view_gc.append(torch.stack([s["global_coords"][v] for s in batch], dim=0))
        out["global_coords"] = torch.stack(per_view_gc, dim=0)

    if "local_coords" in batch[0] and V_l > 0:
        per_view_lc = []
        for v in range(V_l):
            per_view_lc.append(torch.stack([s["local_coords"][v] for s in batch], dim=0))
        out["local_coords"] = torch.stack(per_view_lc, dim=0)

    return out


def _cuda_pin_memory_available() -> bool:
    """Return True only when CUDA is genuinely usable for pinned transfer."""
    try:
        if not torch.cuda.is_available():
            return False
        _ = torch.empty(1).pin_memory()
        return True
    except Exception:
        return False


# =============================================================================
# View generator dispatch
# =============================================================================


def _build_view_generator(
    ssl_method: str | None,
    augmentation_cfg: dict | None,
    fixed_n: int,
    manifest_min_patches: int | None = None,
):
    """Build the augmentor matching the configured SSL method.

    Returns either a ``WSIDualViewAugmentor`` (for dual-view methods) or a
    ``WSIMultiCropAugmentor`` (for ``lejepa_mc``).
    """
    method = (ssl_method or "default").lower()
    if method not in _VALID_SSL_METHODS:
        raise ValueError(f"Unknown ssl_method '{ssl_method}'. Valid: {sorted(_VALID_SSL_METHODS)}")

    if method in _MULTICROP_SSL_METHODS:
        cfg = dict(augmentation_cfg or {})
        # Pull MC-specific keys out of augmentation_cfg.
        num_global = int(cfg.pop("num_global_views", 2))
        num_local = int(cfg.pop("num_local_views", 2))
        global_fixed_n = int(cfg.pop("global_fixed_n", fixed_n))
        local_fixed_n = int(cfg.pop("local_fixed_n", 1024))
        # Crop knobs (per-view).
        global_crop_area_range = tuple(cfg.pop("global_crop_area_range", (0.5, 1.0)))
        local_crop_area_range = tuple(cfg.pop("local_crop_area_range", (0.25, 0.60)))
        global_crop_aspect_range = tuple(cfg.pop("global_crop_aspect_range", (0.5, 2.0)))
        local_crop_aspect_range = tuple(cfg.pop("local_crop_aspect_range", (0.5, 2.0)))
        global_crop_min_keep_frac = float(cfg.pop("global_crop_min_keep_frac", 0.40))
        local_crop_min_keep_frac = float(cfg.pop("local_crop_min_keep_frac", 0.40))
        # Mask knobs (per-view).
        global_mask_ratio_range = tuple(cfg.pop("global_mask_ratio_range", (0.10, 0.25)))
        local_mask_ratio_range = tuple(cfg.pop("local_mask_ratio_range", (0.05, 0.20)))
        global_mask_min_keep_frac = float(cfg.pop("global_mask_min_keep_frac", 0.60))
        local_mask_min_keep_frac = float(cfg.pop("local_mask_min_keep_frac", 0.60))
        # Floor knobs (per-view).
        global_min_tokens = int(cfg.pop("global_min_tokens", 256))
        local_min_tokens_raw = cfg.pop("local_min_tokens", None)
        local_min_tokens = int(local_min_tokens_raw) if local_min_tokens_raw is not None else None
        d4_mode = str(cfg.pop("d4_mode", "shared"))
        # Inter-global coupling — when <1.0 + V_g=2, the augmentor runs the
        # global pair through the dual-view split sampler so MC globals are
        # bit-equivalent to LeJEPA-2C / VICReg / JEPA. base.yaml's dual-view
        # `split_overlap` is the sensible inherited default.
        split_overlap = float(cfg.pop("split_overlap", 1.0))

        # Drop dual-view-only knobs. These are EXPECTED here because
        # `lejepa_mc.yaml` inherits `defaults: - base` and base.yaml's
        # augmentation block is dual-view-shaped; Hydra deep-merge brings the
        # legacy `crop_*` / `mask_*` keys in alongside the MC `global_*` /
        # `local_*` knobs. Logging at DEBUG keeps real misconfiguration
        # visible (the unknown-key warning below) without spamming every run.
        for key in (
            "crop_area_range",
            "mask_ratio_range",
            "crop_aspect_range",
            "crop_min_keep_frac",
            "crop_max_tries",
            "mask_min_keep_frac",
            "min_tokens",
            "fixed_n",
            "crop_prob",
        ):
            if key in cfg:
                logger.debug(
                    "augmentation_cfg.%s inherited from base.yaml is dual-view-only; "
                    "ignored by lejepa_mc.",
                    key,
                )
                cfg.pop(key)

        if cfg:
            logger.warning(
                "augmentation_cfg has unknown keys for lejepa_mc: %s (silently dropped)",
                sorted(cfg.keys()),
            )

        return build_multicrop_augmentor(
            num_global=num_global,
            num_local=num_local,
            global_fixed_n=global_fixed_n,
            local_fixed_n=local_fixed_n,
            global_crop_area_range=global_crop_area_range,
            local_crop_area_range=local_crop_area_range,
            global_crop_aspect_range=global_crop_aspect_range,
            local_crop_aspect_range=local_crop_aspect_range,
            global_crop_min_keep_frac=global_crop_min_keep_frac,
            local_crop_min_keep_frac=local_crop_min_keep_frac,
            global_mask_ratio_range=global_mask_ratio_range,
            local_mask_ratio_range=local_mask_ratio_range,
            global_mask_min_keep_frac=global_mask_min_keep_frac,
            local_mask_min_keep_frac=local_mask_min_keep_frac,
            global_min_tokens=global_min_tokens,
            local_min_tokens=local_min_tokens,
            d4_mode=d4_mode,
            split_overlap=split_overlap,
            manifest_min_patches=manifest_min_patches,
        )

    # Dual-view path. Map "jepa" → "pooled_jepa" (slide-level JEPA shares
    # augmentation policy with vicreg/lejepa; the augmentor factory accepts
    # both names but we keep the alias explicit here for clarity).
    cfg_overrides = dict(augmentation_cfg or {})
    if "fixed_n" in cfg_overrides and cfg_overrides["fixed_n"] != fixed_n:
        logger.warning(
            "augmentation_cfg.fixed_n=%d ignored; using datamodule.fixed_n=%d",
            cfg_overrides["fixed_n"],
            fixed_n,
        )
    cfg_overrides["fixed_n"] = fixed_n  # always set, never via setdefault

    aug_method = "pooled_jepa" if method == "jepa" else method

    return build_augmentor(
        method=aug_method,
        manifest_min_patches=manifest_min_patches,
        **cfg_overrides,
    )


def _log_view_generator(view_gen, ssl_method: str) -> None:
    """Type-aware augmentor logging."""
    if isinstance(view_gen, WSIMultiCropAugmentor):
        gs = view_gen.global_spec
        ls = view_gen.local_spec
        logger.info(
            "Multi-crop augmentor (method=%s): "
            "%d global views @ fixed_n=%d (crop=%s, mask=%s); "
            "%d local views @ fixed_n=%d (crop=%s, mask=%s); "
            "d4=%s",
            ssl_method,
            view_gen.num_global,
            gs.fixed_n,
            gs.crop_area_range,
            gs.mask_ratio_range,
            view_gen.num_local,
            ls.fixed_n,
            ls.crop_area_range,
            ls.mask_ratio_range,
            view_gen.d4_mode,
        )
    elif isinstance(view_gen, WSIDualViewAugmentor):
        cfg = view_gen.cfg
        logger.info(
            "Dual-view augmentor (method=%s): fixed_n=%d d4=%s "
            "split_overlap=%.2f crop_area=%s crop_aspect=%s "
            "crop_min_keep=%.2f mask_ratio=%s mask_min_keep=%.2f "
            "min_tokens=%d",
            ssl_method,
            cfg.fixed_n,
            cfg.d4_mode,
            cfg.split_overlap,
            cfg.crop_area_range,
            cfg.crop_aspect_range,
            cfg.crop_min_keep_frac,
            cfg.mask_ratio_range,
            cfg.mask_min_keep_frac,
            cfg.min_tokens,
        )
    else:
        logger.info("Augmentor (method=%s): %r", ssl_method, view_gen)


# =============================================================================
# DataModule
# =============================================================================


class PretrainDataModule(L.LightningDataModule):
    """Lightning DataModule for SSL pretraining with fixed-N batching.

    Parameters
    ----------
    mmap_dir : str
        Path to mmap directory (features_*.bin, coords_*.bin, index_arrays.npz).
    fixed_n : int
        Patch count per view for dual-view methods. Multi-crop methods derive
        per-view fixed_n from the augmentor's specs.
    batch_size : int
        Samples per batch.
    csv_path : str or None
        Optional pretrain.csv — only slides listed here are used.
    filename_column : str
        Column name in pretrain.csv containing slide IDs.
    split_manifest_path : str or None
        Optional JSON {train_ids, val_ids}. If provided, overrides random split.
    ssl_method : str or None
        SSL method name. Valid: vicreg | jepa | lejepa | lejepa_mc | default.
    augmentation_cfg : dict or None
        Field-level overrides applied to the augmentor config. For multi-crop
        methods, accepts ``num_global_views``, ``num_local_views``,
        ``global_fixed_n``, ``local_fixed_n``, and per-view crop/mask ranges.
    dataset_max_instances : int or None
        Slide-level token budget applied BEFORE augmentation.
    dataset_pre_cap_mode : str
        "spatial_stratified" | "random" | "contiguous" | "head".
    num_workers, prefetch_factor : int
        DataLoader tuning.
    pin_memory : bool or None
        None = auto-detect CUDA safety.
    persistent_workers : bool or None
        None = enabled iff num_workers > 0.
    val_frac : float
        Validation fraction when no manifest is provided.
    seed : int
        Split + augmentation seed.
    force_float32 : bool
        Cast features to fp32 in the dataset (default: keep native dtype).
    coords_aware : bool or None
        Whether the aggregator should receive view coords.
    manifest_min_patches : int or None
        Asserted >= max view fixed_n at augmentor construction.
    """

    def __init__(
        self,
        mmap_dir: str,
        fixed_n: int = 2048,
        batch_size: int = 128,
        csv_path: str | None = None,
        filename_column: str = "slide_id",
        split_manifest_path: str | None = None,
        ssl_method: str | None = "vicreg",
        augmentation_cfg: dict | None = None,
        dataset_max_instances: int | None = 12000,
        dataset_pre_cap_mode: str = "spatial_stratified",
        num_workers: int = 8,
        prefetch_factor: int = 4,
        pin_memory: bool | None = None,
        persistent_workers: bool | None = None,
        val_frac: float = 0.1,
        seed: int = 42,
        force_float32: bool = False,
        max_instances: int | None = None,
        coords_aware: bool | None = None,
        manifest_min_patches: int | None = 2048,
    ):
        if max_instances is not None:
            fixed_n = int(max_instances)
        if manifest_min_patches is not None and manifest_min_patches < fixed_n:
            raise ValueError(
                f"manifest_min_patches={manifest_min_patches} < fixed_n={fixed_n}. "
                "This would allow replacement oversampling."
            )

        if dataset_max_instances is not None and dataset_max_instances < fixed_n:
            raise ValueError(
                f"dataset_max_instances={dataset_max_instances} < fixed_n={fixed_n}. "
                "Pre-cap cannot be smaller than fixed_n."
            )
        if fixed_n < 1:
            raise ValueError(f"fixed_n must be >= 1, got {fixed_n}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be >= 1, got {prefetch_factor}")
        if dataset_max_instances is not None and dataset_max_instances < 1:
            raise ValueError(
                f"dataset_max_instances must be >= 1 or None, got {dataset_max_instances}"
            )
        if not (0.0 < float(val_frac) < 1.0):
            raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")

        super().__init__()
        self.save_hyperparameters()
        self.manifest_min_patches = manifest_min_patches
        self.mmap_dir = mmap_dir
        self.fixed_n = fixed_n
        self.batch_size = batch_size
        self.csv_path = csv_path
        self.filename_column = filename_column
        self.split_manifest_path = split_manifest_path
        self.ssl_method = ssl_method
        self.augmentation_cfg = augmentation_cfg or {}
        self.dataset_max_instances = dataset_max_instances
        self.dataset_pre_cap_mode = dataset_pre_cap_mode
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.val_frac = val_frac
        self.seed = seed
        self.force_float32 = force_float32
        self.coords_aware = coords_aware

        self.train_dataset: PretrainDataset | None = None
        self.val_dataset: PretrainDataset | None = None
        self._feat_dim: int | None = None

    @property
    def feat_dim(self) -> int:
        if self._feat_dim is None:
            self._feat_dim = validate_mmap_dir(self.mmap_dir)["feat_dim"]
        return self._feat_dim

    # ── Slide ID loading / splitting ──────────────────────────────────────

    def _load_slide_ids(self) -> list[str]:
        idx = np.load(str(Path(self.mmap_dir) / "index_arrays.npz"), allow_pickle=True)
        mmap_ids = set(idx["slide_ids"].tolist())

        if self.csv_path is not None and Path(self.csv_path).exists():
            df = pd.read_csv(self.csv_path)
            csv_ids = set(df[self.filename_column].astype(str).tolist())
            slide_ids = sorted(mmap_ids & csv_ids)
            missing = len(csv_ids - mmap_ids)
            if missing:
                logger.warning("%d slides listed in CSV but missing from mmap", missing)
        else:
            slide_ids = sorted(mmap_ids)
            if self.csv_path is not None:
                logger.warning(
                    "CSV not found at %s — using all %d mmap slides", self.csv_path, len(slide_ids)
                )

        logger.info("Loaded %d slide IDs for pretraining", len(slide_ids))
        return slide_ids

    def _split_slide_ids(self, slide_ids: list[str]) -> tuple[list[str], list[str]]:
        n = len(slide_ids)
        if n == 0:
            raise ValueError("No slide IDs are available for pretraining after filtering.")

        split_manifest_path = getattr(self, "split_manifest_path", None)
        if split_manifest_path is not None:
            mf = Path(split_manifest_path)
            if mf.exists():
                payload = json.loads(mf.read_text())
                slide_set = set(slide_ids)
                train_ids = [s for s in payload.get("train_ids", []) if s in slide_set]
                val_ids = [s for s in payload.get("val_ids", []) if s in slide_set]
                if train_ids and val_ids:
                    overlap = set(train_ids) & set(val_ids)
                    if overlap:
                        raise ValueError(
                            f"Split manifest {mf} has {len(overlap)} overlapping "
                            f"train/val slide IDs. First 5: {sorted(overlap)[:5]}"
                        )
                    logger.info(
                        "Using split manifest %s (%d train, %d val)",
                        mf,
                        len(train_ids),
                        len(val_ids),
                    )
                    return train_ids, val_ids
                logger.warning("Manifest %s resolved to empty split — falling back to random", mf)
            else:
                logger.warning("Manifest %s not found — falling back to random split", mf)

        rng = np.random.default_rng(getattr(self, "seed", 42))
        val_frac = float(getattr(self, "val_frac", 0.1))
        if not (0.0 < val_frac < 1.0):
            raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")

        n_val = max(1, int(n * val_frac))
        if n > 1:
            n_val = min(n_val, n - 1)
        perm = rng.permutation(n)
        val_ids = [slide_ids[i] for i in perm[:n_val]]
        train_ids = [slide_ids[i] for i in perm[n_val:]]
        logger.info(
            "Random split: %d train, %d val (val_frac=%.2f, seed=%d)",
            len(train_ids),
            len(val_ids),
            val_frac,
            getattr(self, "seed", 42),
        )
        return train_ids, val_ids

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return

        slide_ids = self._load_slide_ids()
        train_ids, val_ids = self._split_slide_ids(slide_ids)

        view_gen = _build_view_generator(
            ssl_method=self.ssl_method,
            augmentation_cfg=self.augmentation_cfg,
            fixed_n=self.fixed_n,
            manifest_min_patches=self.manifest_min_patches,
        )
        _log_view_generator(view_gen, str(self.ssl_method))

        # Multi-crop's largest view can exceed datamodule.fixed_n
        # (e.g. global_fixed_n=4096 with fixed_n=2048). The pre-cap must
        # not be smaller than the largest view, or the augmentor will be
        # asked to produce a view bigger than the available index pool.
        # Dual-view augmentors do not expose `fixed_n` at the object level,
        # so we fall back to self.fixed_n (which equals their cfg.fixed_n
        # by construction — see `_build_view_generator`).
        max_view_n = int(getattr(view_gen, "fixed_n", self.fixed_n))
        if self.dataset_max_instances is not None and self.dataset_max_instances < max_view_n:
            raise ValueError(
                f"dataset_max_instances={self.dataset_max_instances} < "
                f"max view fixed_n={max_view_n} (ssl_method={self.ssl_method!r}). "
                "Pre-cap cannot be smaller than the largest generated view."
            )

        # coords_aware: True/False/None (None = legacy auto = True). The
        # augmentor still uses coords internally (D4 + spatial crop) regardless;
        # this flag only gates whether per-view coords reach the aggregator.
        coords_aware_resolved = True if self.coords_aware is None else bool(self.coords_aware)
        ds_kwargs = {
            "mmap_dir": self.mmap_dir,
            "fixed_n": self.fixed_n,
            "view_generator": view_gen,
            "dataset_max_instances": self.dataset_max_instances,
            "pre_cap_mode": self.dataset_pre_cap_mode,
            "force_float32": self.force_float32,
            "coords_aware": coords_aware_resolved,
        }
        self.train_dataset = PretrainDataset(slide_ids=train_ids, **ds_kwargs)
        self.val_dataset = PretrainDataset(slide_ids=val_ids, **ds_kwargs)

        for name, ds in [("train", self.train_dataset), ("val", self.val_dataset)]:
            sizes = ds.get_bag_sizes()
            if len(sizes) > 0:
                logger.info(
                    "Bag sizes (%s): mean=%.0f median=%.0f max=%d",
                    name,
                    sizes.mean(),
                    np.median(sizes),
                    sizes.max(),
                )

    # ── DataLoader construction ───────────────────────────────────────────

    def _resolve_pin_memory(self) -> bool:
        return _cuda_pin_memory_available() if self.pin_memory is None else bool(self.pin_memory)

    def _resolve_persistent_workers(self) -> bool:
        if self.persistent_workers is None:
            return self.num_workers > 0
        return bool(self.persistent_workers) and self.num_workers > 0

    def _loader_kwargs(self) -> dict:
        kw = {
            "num_workers": self.num_workers,
            "pin_memory": self._resolve_pin_memory(),
            "persistent_workers": self._resolve_persistent_workers(),
        }
        if self.num_workers > 0:
            kw["prefetch_factor"] = self.prefetch_factor
            kw["worker_init_fn"] = self._worker_init_fn
        return kw

    @staticmethod
    def _worker_init_fn(worker_id: int) -> None:
        """Seed per-worker RNG once at worker birth.

        With persistent_workers=True this runs only at worker construction;
        self.rng state then advances naturally across epochs, producing
        fresh augmentation. Do NOT reseed per epoch.
        """
        info = torch.utils.data.get_worker_info()
        if info is None:
            return
        ds = info.dataset
        if hasattr(ds, "rng"):
            ds.rng = np.random.default_rng(info.seed & 0xFFFF_FFFF)

    def _collate_fn(self):
        """Pick the collator matching the configured ssl_method."""
        if str(self.ssl_method or "").lower() in _MULTICROP_SSL_METHODS:
            return multicrop_stack_collate
        return stack_collate

    def train_dataloader(self) -> DataLoader:
        sampler = None
        shuffle = True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                self.train_dataset,
                shuffle=True,
                drop_last=True,
            )
            shuffle = False

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=self._collate_fn(),
            drop_last=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                self.val_dataset,
                shuffle=False,
            )

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            collate_fn=self._collate_fn(),
            drop_last=False,
            **self._loader_kwargs(),
        )
