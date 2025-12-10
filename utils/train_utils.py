from __future__ import annotations

from typing import Dict, List
import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from torch.amp import autocast
from torch.amp import GradScaler
import random
from typing import Dict, List, Tuple, Callable, Optional, Any
from torch.nn.utils import clip_grad_norm_
import json
import tqdm

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

def _avail_ram_bytes() -> int:
    """Cross-platform-ish available RAM in bytes."""
    if _HAS_PSUTIL:
        return psutil.virtual_memory().available
    # Fallbacks (Linux/Unix)
    try:
        page = os.sysconf('SC_PAGE_SIZE')
        avail_pages = os.sysconf('SC_AVPHYS_PAGES')
        return int(page) * int(avail_pages)
    except Exception:
        # last resort: be conservative (1 GiB)
        return 1 << 30

# -------------------
# Feature Loading
# -------------------
def load_features_mmap(
    feature_dir: str,
    data_combined = None,
    id_col=None,
    verbose: bool = True,
    precision: int = 32,
) -> Tuple[Dict[str, torch.Tensor | Tuple[int, int]], int]:
    """
    Opens a memory-mapped feature array and prepares per-slide access.

    Returns:
        preloaded_features: dict mapping
            - if RAM is sufficient: {sid -> torch.Tensor [N_s, D]} (CPU, zero-copy view onto memmap)
            - if RAM is insufficient: {sid -> (start_idx, end_idx)} for lazy loading
        feature_dim: int

    Notes:
        * Tensors are CPU and zero-copy (share the memmap buffer). They will
          only move to GPU if you call .to('cuda').
        * We keep the memmap alive by storing a reference on each tensor as
          `tensor._source_memmap`. Do not delete that attribute.
        * The memmap is opened read-only; avoid in-place ops on these tensors.
          If you must mutate, do `tensor = tensor.clone()`.
    """
    # --- Prepare paths ---
    base = os.path.join(feature_dir, f"combined_mmap_{precision}")
    meta_p = os.path.join(base, "metadata.json")
    feat_p = os.path.join(base, "features.npy")

    if not os.path.exists(meta_p):
        raise FileNotFoundError(f"metadata.json not found: {meta_p}")
    if not os.path.exists(feat_p):
        raise FileNotFoundError(f"features.npy not found: {feat_p}")

    if verbose:
        print(f"[load_features_mmap] Loading mmap store from: {base}")

    # --- Load metadata ---
    with open(meta_p, "r") as f:
        metadata = json.load(f)

    if "slides" not in metadata or "feature_dim" not in metadata:
        raise KeyError("metadata.json missing required keys: 'slides' and 'feature_dim'")

    slides_meta: Dict[str, dict] = metadata["slides"]
    D = int(metadata["feature_dim"])

    # --- Collect valid IDs ---
    available_ids = set(slides_meta.keys())
    if data_combined is not None and id_col is not None:
        requested_ids = set(map(str, data_combined[id_col].unique().tolist()))
        valid_ids = [sid for sid in requested_ids if sid in available_ids]
        missing = len(requested_ids) - len(valid_ids)
        if missing > 0 and verbose:
            print(f"[load_features_mmap] Warning: {missing} requested slides not found in metadata; skipped.")
    else:
        valid_ids = list(available_ids)

    # --- Compute total_patches & sanity checks ---
    # Prefer explicit metadata if provided; else infer from end_idx
    total_patches = int(metadata.get(
        "total_patches",
        max([v.get("end_idx", 0) for v in slides_meta.values()] or [0])
    ))
    if total_patches <= 0:
        raise ValueError("total_patches computed as 0; check metadata['slides'][sid]['end_idx'].")

    # --- Open memmap with correct dtype ---
    np_dtype = {32: np.float32, 16: np.float16}.get(precision, None)
    if np_dtype is None:
        raise ValueError(f"Unsupported precision {precision}. Use 16 or 32.")

    try:
        features_mmap = np.memmap(feat_p, dtype=np_dtype, mode="c", shape=(total_patches, D))
    except Exception as e:
        raise RuntimeError(f"Failed to open memmap at {feat_p}: {e}")

    if verbose:
        print(f"[load_features_mmap] {len(valid_ids)} WSIs | feature_dim={D} | total_patches={total_patches}")

    # --- Estimate RAM needs ONLY for valid_ids (if we were to preload) ---
    bytes_per_elem = np.dtype(np_dtype).itemsize
    est_bytes = 0
    spans: Dict[str, Tuple[int, int]] = {}
    for sid in valid_ids:
        m = slides_meta[sid]
        if m.get("dummy", False):
            spans[sid] = (0, 0)
            continue
        s = int(m.get("start_idx", -1))
        e = int(m.get("end_idx", -1))
        if s < 0 or e < 0 or e < s or e > total_patches:
            raise ValueError(f"Invalid span for slide {sid}: (start_idx={s}, end_idx={e})")
        spans[sid] = (s, e)
        est_bytes += (e - s) * D * bytes_per_elem

    avail_bytes = _avail_ram_bytes()

    # Heuristic: keep at least 30% of free RAM as headroom
    preload_ok = (est_bytes > 0) and (est_bytes < 0.7 * avail_bytes)

    if verbose:
        est_gb = est_bytes / (1024**3)
        avail_gb = avail_bytes / (1024**3)
        print(f"[load_features_mmap] Est preload for selected slides: {est_gb:.2f} GB | Avail RAM: {avail_gb:.2f} GB "
              f"| decision: {'PRELOAD' if preload_ok else 'INDEX-ONLY'}")

    # --- Branch: index-only (low-RAM) ---
    if not preload_ok:
        preloaded_features: Dict[str, Tuple[int, int]] = {}
        for sid in tqdm.tqdm(valid_ids, disable=not verbose):
            s, e = spans[sid]
            if s == e == 0 and slides_meta.get(sid, {}).get("dummy", False):
                # represent dummy explicitly
                preloaded_features[sid] = (0, 0)
                if verbose:
                    print(f"[load_features_mmap] Dummy slide {sid} (no features).")
            else:
                preloaded_features[sid] = (s, e)

        if verbose:
            print(f"[load_features_mmap] Prepared {len(preloaded_features)} slides with indices only. "
                  f"Use mmap slices later to load per-batch.")
        return preloaded_features, D, preload_ok

    # --- Preload as zero-copy CPU tensors (share memmap buffer) ---
    preloaded_features: Dict[str, torch.Tensor] = {}
    for sid in tqdm.tqdm(valid_ids, disable=not verbose):
        m = slides_meta[sid]
        if m.get("dummy", False):
            feats = torch.zeros((1, D), dtype=torch.float32)  # small sentinel
            if verbose:
                print(f"[load_features_mmap] Dummy slide {sid}: created 1xD zeros.")
        else:
            s, e = spans[sid]
            # Avoid .astype(..., copy=False); memmap already has correct dtype
            arr = features_mmap[s:e]  # zero-copy slice view
            # Important: from_numpy shares buffer (still CPU).
            feats = torch.from_numpy(arr)
            feats.requires_grad_(False)        # belt-and-suspenders
            
            # Keep memmap alive by attaching a strong reference to the tensor
            feats._source_memmap = features_mmap  # DO NOT DELETE THIS ATTRIBUTE

        preloaded_features[sid] = feats

    return preloaded_features, D, preload_ok

def prepare_labels(
    data_combined: pd.DataFrame,
    id_col: str,
    task: str,
    verbose: bool = True,
    cohorts: List[str] | None = None,
    patient_id_col: str = None
) -> Tuple[pd.DataFrame, Dict[str, Any], torch.Tensor | None, int]:
    """
    Filters and prepares labels from a DataFrame for a given task.

    This optimized version uses vectorized pandas operations instead of iterrows
    for significantly improved performance.

    Args:
        data_combined: The input DataFrame containing all data.
        id_col: The name of the column containing unique sample identifiers.
        task: The name of the target task. Can be a column name for classification
              or "survival".
        verbose: If True, prints detailed processing information.
        cohorts: An optional list of cohort prefixes to filter the data by.

    Returns:
        A tuple containing:
        - df (pd.DataFrame): The filtered DataFrame with only valid samples.
        - labels (Dict): A dictionary mapping sample IDs to their processed labels.
        - class_weights (torch.Tensor | None): Computed class weights for classification tasks.
        - n_classes (int): The number of unique classes for the task.
    """
    ## 1. Initial Filtering by Cohort and Task
    if cohorts:
        cohort_mask = data_combined["Cohort"].str.startswith(tuple(cohorts))
        df = data_combined.loc[cohort_mask].copy()
    else:
        df = data_combined.copy()

    # Define columns required for the task to check for NaNs
    if task == "survival":
        required_cols = ['Overall Survival (months)', 'Vital Status']
    else:
        required_cols = [task]

    # Ensure all required columns exist
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' for task '{task}' not found in DataFrame.")

    # Drop rows with missing values in required columns
    df.dropna(subset=required_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if verbose:
        print(f"Filtered to {len(df)} samples with valid labels for task '{task}'.")

    if df.empty:
        print("Warning: DataFrame is empty after filtering. No labels to prepare.")
        return df, {}, None, 0

    ## 2. Task-Specific Label and Weight Processing
    labels = {}
    class_weights = None
    n_classes = 0
    patient_id_mapping = dict(zip(df[id_col], df[patient_id_col])) if patient_id_col else {}

    # --- Survival Task ---
    if task == "survival":
        n_classes = 1 # Often represents a single risk score output
        
        # Vectorized creation of time and event tensors
        times = torch.log1p(torch.tensor(df['Overall Survival (months)'].values, dtype=torch.float32))
        events = torch.tensor((df['Vital Status'] == 'Dead').values.astype(float), dtype=torch.float32)
        
        # Create labels dictionary efficiently using zip
        labels = dict(zip(df[id_col], zip(times, events)))

        if verbose:
            event_count = int(events.sum())
            print(f"Prepared survival labels for {len(labels)} samples ({event_count} events).")

    # --- Classification Task ---
    else:
        # Use pd.factorize for efficient and robust label encoding
        labels_encoded, unique_classes = pd.factorize(df[task], sort=True)
        n_classes = len(unique_classes)

        if n_classes < 2:
            raise ValueError(f"Task '{task}' has {n_classes} unique class(es). At least 2 are required.")

        # Create labels dictionary efficiently using zip
        labels_tensor = torch.tensor(labels_encoded, dtype=torch.long)
        labels = dict(zip(df[id_col], labels_tensor))

        # Calculate class weights in a vectorized way
        class_counts = torch.tensor(np.bincount(labels_encoded), dtype=torch.float32)
        total_samples = len(df)
        weights = total_samples / (n_classes * class_counts)
        class_weights = weights.clone().detach().float()

        if verbose:
            print(f"Detected {n_classes} classes: {unique_classes.tolist()}")
            print(f"Computed class weights: {class_weights.tolist()}")
            counts_series = pd.Series(labels_encoded).value_counts().sort_index()
            dist_str = ", ".join([f"'{c}' ({counts_series[i]})" for i, c in enumerate(unique_classes)])
            print(f"Class distribution -> {dist_str}")

    return df, labels, class_weights, n_classes, patient_id_mapping

# -----------------------------
# Repro
# -----------------------------
SEED = 42
def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _to_device(d) -> torch.device:
    return d if isinstance(d, torch.device) else torch.device(d)

def _worker_init_fn(worker_id):
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# -----------------------------
# Dataset (returns with or without key)
# -----------------------------
class WSIBagDataset(Dataset):
    """
    features_dict: {id: Tensor [N,C] (float16)}
    labels_dict:   {id: torch.long scalar OR int}
    """
    def __init__(
        self,
        ids: List[str],
        features_dict: Dict[str, torch.Tensor],
        labels_dict: Dict[str, torch.Tensor | int],
        bag_size: Optional[int] = None,
        replacement: bool = True,
        return_key: bool = False,
        cpu_cast_float32: bool = False,
        preloaded: bool = True,
        feature_dir: Optional[str] = None,
        precision: int = 16,
        mmap_mode: str = "c",
    ):
        self.ids = list(ids)
        self.X = features_dict
        self.y = labels_dict
        self.bag_size = bag_size
        self.replacement = replacement
        self.return_key = return_key
        self.cpu_cast_float32 = cpu_cast_float32
        self.preloaded = preloaded
        self.precision = int(precision)
        if self.precision not in (16, 32):
            raise ValueError("precision must be 16 or 32")
        self._mode = "preloaded" if preloaded else "mmap"
        self._feature_dim = None
        
        if not preloaded:
            if feature_dir is None:
                raise ValueError("preloaded=False requires feature_dir (dir with features.npy & metadata.json).")
            meta_p = os.path.join(feature_dir, f"combined_mmap_{precision}", "metadata.json")
            feat_p = os.path.join(feature_dir, f"combined_mmap_{precision}", "features.npy")
            if not os.path.exists(meta_p):
                raise FileNotFoundError(f"metadata.json not found: {meta_p}")
            if not os.path.exists(feat_p):
                raise FileNotFoundError(f"features.npy not found: {feat_p}")
            with open(meta_p, "r") as f:
                meta = json.load(f)
            if "feature_dim" not in meta:
                raise KeyError("metadata.json missing 'feature_dim'.")
            self.feature_dim = int(meta["feature_dim"])

            total_patches = int(meta.get("total_patches", 0))
            if total_patches <= 0:
                raise ValueError("metadata.json must include a positive 'total_patches' in lazy mode.")

            self._np_dtype = {16: np.float16, 32: np.float32}.get(int(precision))
            if self._np_dtype is None:
                raise ValueError("precision must be 16 or 32.")
            self._feat_path = feat_p
            self._shape0 = total_patches
            self._mmap_mode = mmap_mode
            # open memmap (will be re-opened in workers)
            self._features_mmap = np.memmap(
                self._feat_path, dtype=self._np_dtype, mode=self._mmap_mode,
                shape=(self._shape0, self.feature_dim)
            )            
            
    def __len__(self) -> int:
        return len(self.ids)

    # ---------- multiprocessing: re-open memmap in workers ----------
    def __getstate__(self):
        st = self.__dict__.copy()
        st["_features_mmap"] = None  # don't pickle live handle
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        if not self.preloaded and self._features_mmap is None:
            self._features_mmap = np.memmap(
                self._feat_path, dtype=self._np_dtype, mode=self._mmap_mode,
                shape=(self._shape0, self.feature_dim)
            )

    def _sample_instances(self, bag: torch.Tensor) -> torch.Tensor:
        N = bag.size(0)
        if self.bag_size is None or N == self.bag_size:
            return bag
        if N > self.bag_size:
            idx = torch.randperm(N)[: self.bag_size]
            return bag[idx]
        if not self.replacement:
            return bag
        extra = torch.randint(0, N, (self.bag_size - N,))
        idx = torch.cat([torch.arange(N), extra], dim=0)
        return bag[idx]

    def __getitem__(self, i: int):
        key = self.ids[i]
        if self.preloaded:
            bag = self.X[key]
        else:
            s, e = self.X[key]
            bag = self._features_mmap[s:e]
            
        if isinstance(bag, np.ndarray):
            bag = torch.from_numpy(bag)
        bag = bag.to(dtype=torch.float16, copy=False)  # features are preloaded fp16

        bag = self._sample_instances(bag)

        # Cast to float32 if staying on CPU (safer; many CPU ops don't support fp16)
        if self.cpu_cast_float32:
            bag = bag.float()

        lab = self.y[key]
        if torch.is_tensor(lab):
            lab = int(lab.item())
        else:
            lab = int(lab)

        if self.return_key:
            return bag, lab, key
        return bag, lab

def make_stratified_group_splitter(y, groups=None, n_splits=5, seed=42):
    if groups is None:
        print("No groups provided, using StratifiedKFold.")
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    groups = np.asarray(groups)
    # If every group is unique, grouping has no effect → use StratifiedKFold
    if np.unique(groups).size == groups.size:
        print("All groups are unique, using StratifiedKFold.")
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

def adamw_param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower() or "bn" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def train_one_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
    class_weights: torch.Tensor = None,
    use_amp: bool = False,
    scaler: GradScaler = None,
    max_grad_norm: float = None,
    scheduler: Optional[Callable] = None,
    accum_steps: int = 1,
    epoch: int = 0,
    updates_per_epoch: int = 0
) -> float:
    """
    Gradient accumulation: performs one optimizer step every `accum_steps` batches.
    Set accum_steps = effective_batch_size / (physical batch size).
    """
    assert accum_steps >= 1
    model.train()
    total_loss = 0.0
    scaler = scaler or GradScaler(enabled=use_amp)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer.zero_grad(set_to_none=True)
    amp_enabled = bool(use_amp and device.type == "cuda")
    global_update = epoch * updates_per_epoch  # start-of-epoch offset    
    
    for batch_idx, (feats, labs) in enumerate(train_loader):
        feats = feats.to(device, non_blocking=True)
        labs  = labs.to(device, non_blocking=True).long()

        with autocast(device_type=device.type, enabled=amp_enabled):
            if getattr(model, "encoder_type", None) == "CLAM" or getattr(model, "encoder_type", None) == "DSMIL":
                logits, log_dict = model(feats, return_raw_attention=True, labels=labs)
                instance_loss = log_dict.get("instance_loss", -1) if isinstance(log_dict, dict) else -1
            else:
                logits, _ = model(feats)
                instance_loss = -1

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"Invalid logits at batch {batch_idx}. Sanitizing.")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

            if task != "survival":
                loss = loss_fn(logits, labs)
                if instance_loss is not None and instance_loss != -1 and model.encoder_type == "CLAM":
                    loss = 0.7 * loss + 0.3 * instance_loss
                elif instance_loss is not None and instance_loss != -1 and model.encoder_type == "DSMIL":
                    loss = 0.5 * loss + 0.5 * instance_loss
            else:
                raise NotImplementedError("Survival task not implemented in this function.")

            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"Invalid loss ({loss.item()}) at batch {batch_idx}")

        # For reporting: accumulate the real (unscaled) loss value
        total_loss += float(torch.clamp(loss, min=0.0).detach().cpu())

        # Scale loss for accumulation
        loss_for_backward = torch.clamp(loss, min=0.0) / accum_steps
        scaler.scale(loss_for_backward).backward()

        # Step when we've accumulated enough micro-batches, or at the very end
        do_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(train_loader))
        if do_step:
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            if scheduler is not None:
                # per-update stepping
                global_update += 1
                if hasattr(scheduler, "step_update"):
                    scheduler.step_update(global_update)
                else:
                    scheduler.step(global_update)

    avg_loss = total_loss / max(1, len(train_loader))
    return avg_loss
