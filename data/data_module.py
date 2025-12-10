from __future__ import annotations

import os
import json
import random
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False


# ============================================================
# RAM helper
# ============================================================

def _avail_ram_bytes() -> int:
    """Cross-platform-ish available RAM in bytes."""
    if _HAS_PSUTIL:
        return psutil.virtual_memory().available
    # Fallbacks (Linux/Unix)
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page) * int(avail_pages)
    except Exception:
        # last resort: be conservative (1 GiB)
        return 1 << 30


# ============================================================
# Feature Loading (memmap + optional preload)
# ============================================================

def load_features_mmap(
    feature_dir: str,
    df: Optional[pd.DataFrame] = None,
    id_col: Optional[str] = None,
    verbose: bool = True,
    precision: int = 16,  # 16 or 32
) -> Tuple[Dict[str, torch.Tensor | Tuple[int, int]], int, bool]:
    """
    Opens a memory-mapped feature array and prepares per-slide access.

    Returns:
        preloaded_features: dict mapping
            - if RAM is sufficient:
                {sid -> torch.Tensor [N_s, D]} (CPU, zero-copy view onto memmap)
            - if RAM is insufficient:
                {sid -> (start_idx, end_idx)} for lazy loading
        feature_dim: int
        preload_ok: bool (True if we actually preloaded tensors)

    Notes:
        * Tensors are CPU and zero-copy (share the memmap buffer). They will
          only move to GPU if you call .to('cuda').
        * We keep the memmap alive by storing a reference on each tensor as
          `tensor._source_memmap`. Do not delete that attribute.
        * The memmap is opened read-only; avoid in-place ops on these tensors.
    """
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

    # --- Collect valid IDs from metadata & df ---
    available_ids = set(slides_meta.keys())
    if df is not None and id_col is not None:
        requested_ids = set(map(str, df[id_col].astype(str).unique().tolist()))
        valid_ids = [sid for sid in requested_ids if sid in available_ids]
        missing = len(requested_ids) - len(valid_ids)
        if missing > 0 and verbose:
            print(f"[load_features_mmap] Warning: {missing} requested slides not found in metadata; skipped.")
    else:
        valid_ids = list(available_ids)

    # --- Compute total_patches & sanity checks ---
    total_patches = int(metadata.get(
        "total_patches",
        max([v.get("end_idx", 0) for v in slides_meta.values()] or [0])
    ))
    if total_patches <= 0:
        raise ValueError("total_patches computed as 0; check metadata['slides'][sid]['end_idx'].")

    # --- Open memmap with correct dtype ---
    np_dtype = {32: np.float32, 16: np.float16}.get(int(precision), None)
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
    preload_ok = (est_bytes > 0) and (est_bytes < 0.7 * avail_bytes)

    if verbose:
        est_gb = est_bytes / (1024**3)
        avail_gb = avail_bytes / (1024**3)
        print(
            f"[load_features_mmap] Est preload for selected slides: {est_gb:.2f} GB | "
            f"Avail RAM: {avail_gb:.2f} GB | decision: {'PRELOAD' if preload_ok else 'INDEX-ONLY'}"
        )

    # --- Branch: index-only (low-RAM) ---
    if not preload_ok:
        preloaded_features: Dict[str, Tuple[int, int]] = {}
        for sid in valid_ids:
            s, e = spans[sid]
            if s == e == 0 and slides_meta.get(sid, {}).get("dummy", False):
                preloaded_features[sid] = (0, 0)
                if verbose:
                    print(f"[load_features_mmap] Dummy slide {sid} (no features).")
            else:
                preloaded_features[sid] = (s, e)

        if verbose:
            print(
                f"[load_features_mmap] Prepared {len(preloaded_features)} slides with indices only. "
                f"Use mmap slices later to load per-batch."
            )
        return preloaded_features, D, False

    # --- Preload as zero-copy CPU tensors (share memmap buffer) ---
    preloaded_features: Dict[str, torch.Tensor] = {}
    for sid in valid_ids:
        m = slides_meta[sid]
        if m.get("dummy", False):
            feats = torch.zeros((1, D), dtype=torch.float32)  # small sentinel
            if verbose:
                print(f"[load_features_mmap] Dummy slide {sid}: created 1xD zeros.")
        else:
            s, e = spans[sid]
            arr = features_mmap[s:e]  # zero-copy memmap slice
            feats = torch.from_numpy(arr)  # shares buffer
            feats.requires_grad_(False)
            feats._source_memmap = features_mmap  # keep memmap alive

        preloaded_features[sid] = feats

    return preloaded_features, D, True


# ============================================================
# Label prep
# ============================================================

def prepare_labels(
    df: pd.DataFrame,
    id_col: str,
    task: str,
    verbose: bool = True,
    cohorts: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[torch.Tensor], int]:
    """
    Filters and prepares labels from a DataFrame for a given task.

    Args:
        df: Input DataFrame containing all data (must include `id_col`, `task`, etc.).
        id_col: Column containing unique sample identifiers.
        task: Column name for classification or "survival".
        cohorts: Optional list of cohort prefixes to keep (based on df["Cohort"]).

    Returns:
        df_filtered: filtered DataFrame (only valid rows for the task).
        labels: dict {id -> label}
                classification: id -> class index (torch.long)
                survival: id -> (time_tensor, event_tensor)
        class_weights: class weights for classification, else None
        n_classes: number of unique classes (1 for survival).
    """
    # 1. Cohort filtering
    if cohorts:
        if "Cohort" not in df.columns:
            raise ValueError("Cohort filtering requested but 'Cohort' column not found in DataFrame.")
        cohort_mask = df["Cohort"].str.startswith(tuple(cohorts))
        df = df.loc[cohort_mask].copy()
    else:
        df = df.copy()

    # 2. Required columns
    if task == "survival":
        required_cols = ["Overall Survival (months)", "Vital Status"]
    else:
        required_cols = [task]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' for task '{task}' not found in DataFrame.")

    # 3. Drop rows with missing values in required columns
    df.dropna(subset=required_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if verbose:
        print(f"Filtered to {len(df)} samples with valid labels for task '{task}'.")

    if df.empty:
        print("Warning: DataFrame is empty after filtering. No labels to prepare.")
        return df, {}, None, 0

    labels: Dict[str, Any] = {}
    class_weights: Optional[torch.Tensor] = None
    n_classes = 0

    # Survival case
    if task == "survival":
        n_classes = 1
        times = torch.log1p(torch.tensor(df["Overall Survival (months)"].values, dtype=torch.float32))
        events = torch.tensor((df["Vital Status"] == "Dead").values.astype(float), dtype=torch.float32)
        labels = dict(zip(df[id_col].astype(str), zip(times, events)))

        if verbose:
            event_count = int(events.sum())
            print(f"Prepared survival labels for {len(labels)} samples ({event_count} events).")

    else:
        # Classification
        labels_encoded, unique_classes = pd.factorize(df[task], sort=True)
        n_classes = len(unique_classes)
        if n_classes < 2:
            raise ValueError(f"Task '{task}' has {n_classes} unique class(es). At least 2 are required.")

        labels_tensor = torch.tensor(labels_encoded, dtype=torch.long)
        labels = dict(zip(df[id_col].astype(str), labels_tensor))

        class_counts = torch.tensor(np.bincount(labels_encoded), dtype=torch.float32)
        total_samples = len(df)
        weights = total_samples / (n_classes * class_counts)
        class_weights = weights.clone().detach().float()

        if verbose:
            print(f"Detected {n_classes} classes: {unique_classes.tolist()}")
            print(f"Computed class weights: {class_weights.tolist()}")
            counts_series = pd.Series(labels_encoded).value_counts().sort_index()
            dist_str = ", ".join(
                [f"'{c}' ({counts_series[i]})" for i, c in enumerate(unique_classes)]
            )
            print(f"Class distribution -> {dist_str}")

    return df, labels, class_weights, n_classes

# ============================================================
# Dataset: WSIBagDataset
# ============================================================

class WSIBagDataset(Dataset):
    """
    Bag-level WSI dataset for MIL.

    features_dict:
        - preloaded mode: {id -> Tensor [N_patches, C]}
        - mmap-index mode: {id -> (start_idx, end_idx)}

    labels_dict:
        - classification: {id -> class_idx (torch.long or int)}
        - survival:       {id -> (time_tensor, event_tensor)}
    """
    def __init__(
        self,
        ids: List[str],
        features_dict: Dict[str, torch.Tensor | Tuple[int, int]],
        labels_dict: Dict[str, Any],
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
        self.preloaded = bool(preloaded)
        self.precision = int(precision)
        if self.precision not in (16, 32):
            raise ValueError("precision must be 16 or 32")

        self._mode = "preloaded" if self.preloaded else "mmap"
        self.feature_dim: Optional[int] = None

        # Lazy memmap mode: we open features.npy inside the dataset
        if not self.preloaded:
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

            self._np_dtype = {16: np.float16, 32: np.float32}.get(self.precision)
            if self._np_dtype is None:
                raise ValueError("precision must be 16 or 32.")

            self._feat_path = feat_p
            self._shape0 = total_patches
            self._mmap_mode = mmap_mode

            # Open memmap (will be reopened in workers due to __getstate__/__setstate__)
            self._features_mmap = np.memmap(
                self._feat_path,
                dtype=self._np_dtype,
                mode=self._mmap_mode,
                shape=(self._shape0, self.feature_dim),
            )

    def __len__(self) -> int:
        return len(self.ids)

    # multiprocessing: re-open memmap in workers
    def __getstate__(self):
        st = self.__dict__.copy()
        if not self.preloaded:
            st["_features_mmap"] = None  # don't pickle live handle
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        if not self.preloaded and self._features_mmap is None:
            self._features_mmap = np.memmap(
                self._feat_path,
                dtype=self._np_dtype,
                mode=self._mmap_mode,
                shape=(self._shape0, self.feature_dim),
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

        # Load bag either from preloaded tensor or via memmap index range
        if self.preloaded:
            bag = self.X[key]
        else:
            s, e = self.X[key]
            bag = self._features_mmap[s:e]

        if isinstance(bag, np.ndarray):
            bag = torch.from_numpy(bag)

        # Cast to fp16 (features are stored as fp16 or fp32; we standardize)
        bag = bag.to(dtype=torch.float16, copy=False)

        # Possibly down/up-sample patches
        bag = self._sample_instances(bag)

        # Cast to fp32 if staying on CPU and you want safer ops
        if self.cpu_cast_float32:
            bag = bag.float()

        lab = self.y[key]
        # Survival: (time, event) as tensors -> keep as is
        if isinstance(lab, tuple):
            label_out = lab
        else:
            if torch.is_tensor(lab):
                label_out = int(lab.item())
            else:
                label_out = int(lab)

        if self.return_key:
            return bag, label_out, key
        return bag, label_out


# ============================================================
# Lightning DataModule with numeric fold column
# ============================================================

"""
CSV example:

id,label,fold
1,0,0
2,1,0
3,0,1
4,0,-1  # optional: no fold assigned (use for test set)
5,1,1

For fold k, train on fold >= 0 and fold != k, validate on fold == k.
Fold == -1 is treated as test.
"""

class WSIMILDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        feature_dir: str,
        id_col: str,
        target_col: str,
        cohorts: Optional[List[str]] = None,
        # dataloader / dataset
        batch_size: int = 1,
        num_workers: int = 4,
        bag_size: Optional[int] = None,
        replacement: bool = True,
        return_key: bool = False,
        precision: int = 16,
        preloaded: Optional[bool] = None,  # if None -> use load_features_mmap decision
        cpu_cast_float32: bool = False,
        # misc
        pin_memory: bool = True,
        verbose: bool = True,
        current_fold: int = 0,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.feature_dir = feature_dir
        self.id_col = id_col
        self.target_col = target_col
        self.cohorts = cohorts

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bag_size = bag_size
        self.replacement = replacement
        self.return_key = return_key
        self.precision = int(precision)
        self._forced_preloaded = preloaded
        self.cpu_cast_float32 = cpu_cast_float32

        self.pin_memory = pin_memory
        self.verbose = verbose
        self.current_fold = int(current_fold)

        # Populated in setup()
        self.data_df: Optional[pd.DataFrame] = None
        self.labels_dict: Dict[str, Any] = {}
        self.class_weights: Optional[torch.Tensor] = None
        self.n_classes: int = 0

        self.features_dict: Dict[str, Any] = {}
        self.feature_dim: Optional[int] = None
        self.preloaded: bool = True  # effective mode used by WSIBagDataset

        self.train_ids: List[str] = []
        self.val_ids: List[str] = []
        self.test_ids: List[str] = []

        self.train_dataset: Optional[WSIBagDataset] = None
        self.val_dataset: Optional[WSIBagDataset] = None
        self.test_dataset: Optional[WSIBagDataset] = None

    def prepare_data(self):
        # Nothing to download here.
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Build datasets and splits. Called on trainer.fit()/validate()/test().
        """
        # 1) Load CSV
        if self.data_df is None:
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"CSV not found: {self.csv_path}")
            self.data_df = pd.read_csv(self.csv_path)

        # 2) Prepare labels (cohort filtering + NaN removal)
        df_filtered, labels_dict, class_weights, n_classes = prepare_labels(
            df=self.data_df,
            id_col=self.id_col,
            task=self.target_col,
            verbose=self.verbose,
            cohorts=self.cohorts,
        )
        self.data_df = df_filtered
        self.labels_dict = labels_dict
        self.class_weights = class_weights
        self.n_classes = n_classes

        if self.data_df.empty:
            raise RuntimeError("No samples left after label filtering.")

        # 3) Load features (preloaded or index-only) restricted to df_filtered
        if self.verbose:
            print(f"[WSIMILDataModule] Loading features from {self.feature_dir}")

        feats, feature_dim, preload_ok = load_features_mmap(
            feature_dir=self.feature_dir,
            df=self.data_df,
            id_col=self.id_col,
            verbose=self.verbose,
            precision=self.precision,
        )
        self.features_dict = feats
        self.feature_dim = feature_dim

        # Decide effective preloaded flag
        if self._forced_preloaded is None:
            self.preloaded = preload_ok
        else:
            self.preloaded = bool(self._forced_preloaded)

        if self.verbose:
            mode = "PRELOADED" if self.preloaded else "MMAP-INDEX"
            print(f"[WSIMILDataModule] Dataset mode: {mode} | feature_dim={self.feature_dim}")

        # 4) Keep only rows that actually have features
        valid_ids = set(self.features_dict.keys())
        self.data_df = self.data_df[self.data_df[self.id_col].astype(str).isin(valid_ids)].copy()
        # Filter labels_dict to valid_ids
        self.labels_dict = {str(k): v for k, v in self.labels_dict.items() if str(k) in valid_ids}

        if self.verbose:
            print(f"[WSIMILDataModule] After matching to features: {len(self.data_df)} slides")

        if self.data_df.empty:
            raise RuntimeError("No samples left after matching labels with features.")

        # 5) Build splits based on numeric 'fold' column
        self._build_splits()

        # 6) Construct datasets
        if stage in (None, "fit", "validate"):
            self.train_dataset = WSIBagDataset(
                ids=self.train_ids,
                features_dict=self.features_dict,
                labels_dict=self.labels_dict,
                bag_size=self.bag_size,
                replacement=self.replacement,
                return_key=self.return_key,
                cpu_cast_float32=self.cpu_cast_float32,
                preloaded=self.preloaded,
                feature_dir=self.feature_dir,
                precision=self.precision,
            )
            self.val_dataset = WSIBagDataset(
                ids=self.val_ids,
                features_dict=self.features_dict,
                labels_dict=self.labels_dict,
                bag_size=self.bag_size,
                replacement=False,
                return_key=self.return_key,
                cpu_cast_float32=self.cpu_cast_float32,
                preloaded=self.preloaded,
                feature_dir=self.feature_dir,
                precision=self.precision,
            )

        if stage in (None, "test"):
            test_ids = self.test_ids if len(self.test_ids) > 0 else self.val_ids
            self.test_dataset = WSIBagDataset(
                ids=test_ids,
                features_dict=self.features_dict,
                labels_dict=self.labels_dict,
                bag_size=self.bag_size,
                replacement=False,
                return_key=self.return_key,
                cpu_cast_float32=self.cpu_cast_float32,
                preloaded=self.preloaded,
                feature_dir=self.feature_dir,
                precision=self.precision,
            )

    def _build_splits(self):
        """
        Use numeric 'fold' column:

        - train: fold >= 0 and fold != current_fold
        - val:   fold == current_fold
        - test:  fold == -1
        """
        if "k_fold" not in self.data_df.columns:
            raise ValueError("Expected a 'k_fold' column in CSV for k-fold splitting.")

        df = self.data_df.copy()
        df["k_fold"] = df["k_fold"].astype(int)

        k = self.current_fold

        is_val = df["k_fold"] == k
        is_train = (df["k_fold"] >= 0) & (df["k_fold"] != k)
        is_test = df["k_fold"] == -1

        train_ids = df.loc[is_train, self.id_col].astype(str).tolist()
        val_ids = df.loc[is_val, self.id_col].astype(str).tolist()
        test_ids = df.loc[is_test, self.id_col].astype(str).tolist()

        self.train_ids = [sid for sid in train_ids if sid in self.features_dict]
        self.val_ids = [sid for sid in val_ids if sid in self.features_dict]
        self.test_ids = [sid for sid in test_ids if sid in self.features_dict]

        if self.verbose:
            print(
                f"[WSIMILDataModule] Split sizes (fold={k}) -> "
                f"train={len(self.train_ids)}, val={len(self.val_ids)}, test={len(self.test_ids)}"
            )

    # ------------------------------
    # Dataloaders
    # ------------------------------

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not built. Call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not built. Call setup('fit'/'validate') first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is not built. Call setup('test') first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    # ------------------------------
    # Convenience getters
    # ------------------------------

    @property
    def class_weight_tensor(self) -> Optional[torch.Tensor]:
        return self.class_weights

    @property
    def num_classes(self) -> int:
        return self.n_classes

    @property
    def dim_features(self) -> Optional[int]:
        return self.feature_dim

if __name__ == "__main__":
    # ---- user-configurable paths ----
    csv_path = "../src/blca_k_fold.csv"              # <-- change to your CSV
    feature_dir = "../../BLCA_AI/combined_features/wsi_processed_no_artifacts/features_uni_v1"                 # <-- change to your feature root
    id_col = "De ID"
    target_col = "Binary WHO 2022"
    precision = 16
    current_fold = 0  # test fold 0

    # ---- instantiate datamodule ----
    dm = WSIMILDataModule(
        csv_path=csv_path,
        feature_dir=feature_dir,
        id_col=id_col,
        target_col=target_col,
        cohorts=None,             # or ["TCGA", "RHO", ...] if you use Cohort column
        batch_size=1,
        num_workers=0,            # set >0 later; 0 is safer for quick test
        bag_size=None,            # full bag; or an int for patch subsampling
        replacement=True,
        return_key=True,          # so we can see the slide IDs
        precision=precision,
        preloaded=None,           # None = auto (RAM-based), True/False to force
        pin_memory=False,
        verbose=True,
        current_fold=current_fold,
    )

    # ---- setup & build loaders ----
    print(f"\n=== Setting up DataModule for fold={current_fold} ===")
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    print("\n=== Inspect one batch from train_dataloader ===")
    for batch_idx, batch in enumerate(train_loader):
        # Because return_key=True, batch = (bag, label, key)
        bags, labels, keys = batch

        print(f"Batch {batch_idx}:")
        print(f"  bags.shape   = {bags.shape}")   # [B, N_patches, C] if you collate properly, or [N_patches, C] with B=1
        print(f"  labels       = {labels}")
        print(f"  keys         = {keys}")

        break  # just one batch

    print("\n=== Inspect one batch from val_dataloader ===")
    for batch_idx, batch in enumerate(val_loader):
        bags, labels, keys = batch
        print(f"Val batch {batch_idx}:")
        print(f"  bags.shape   = {bags.shape}")
        print(f"  labels       = {labels}")
        print(f"  keys         = {keys}")
        break

    # ---- optional: test loader sanity ----
    print("\n=== Setting up test stage ===")
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    for batch_idx, batch in enumerate(test_loader):
        bags, labels, keys = batch
        print(f"Test batch {batch_idx}:")
        print(f"  bags.shape   = {bags.shape}")
        print(f"  labels       = {labels}")
        print(f"  keys         = {keys}")
        break

    print("\nSanity check complete.")