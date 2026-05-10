#!/usr/bin/env python3
"""
Sklearn-based linear probing for slide embeddings stored in OceanPath mmap format.

Design goals
============
- Keep the probe simple and stable: StandardScaler + LogisticRegression.
- Respect patient-level leakage rules by splitting on patient_id, never slide_id.
- Support the two downstream patterns in the proposal:
    1) patient-level grouped cross-validation (default)
    2) external locked test evaluation (train/tune on source cohort, test once on target cohort)
- Work directly on mmap-packed slide embeddings produced by the existing mmap_builder.
- Aggregate predictions to patient level by averaging slide probabilities.

Expected mmap format
====================
The embedding mmap must use the same index_arrays.npz / features_*.bin layout as
OceanPath's build_mmap(). Each slide should usually have exactly one row
(length == 1). If a slide has multiple rows, set --non_singleton_policy to
mean or first.

Typical usage
=============
1) Patient-level 5-fold CV:

python scripts/linear_probing_sklearn.py \
    --mmap_dir mmap/tcga_nsclc_embeddings \
    --manifest_csv data/tcga_nsclc_manifest.csv \
    --label_column label \
    --patient_column patient_id \
    --output_dir outputs/linear_probe/tcga_nsclc \
    --protocol grouped_cv \
    --n_splits 5

2) External transfer (train TCGA -> test CPTAC):

python scripts/linear_probing_sklearn.py \
    --mmap_dir mmap/tcga_nsclc_embeddings \
    --manifest_csv data/tcga_nsclc_manifest.csv \
    --test_mmap_dir mmap/cptac_nsclc_embeddings \
    --test_manifest_csv data/cptac_nsclc_manifest.csv \
    --label_column label \
    --patient_column patient_id \
    --output_dir outputs/linear_probe/nsclc_tcga_to_cptac \
    --protocol external_test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from warnings import simplefilter

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

simplefilter("ignore", ConvergenceWarning)

logger = logging.getLogger("linear_probe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear probing on mmap-backed slide embeddings.")

    # Core data
    parser.add_argument(
        "--mmap_dir", type=str, required=True, help="Train/primary embedding mmap dir."
    )
    parser.add_argument(
        "--manifest_csv", type=str, required=True, help="Train/primary manifest CSV."
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for outputs.")

    # Optional external test cohort
    parser.add_argument(
        "--test_mmap_dir", type=str, default=None, help="External test embedding mmap dir."
    )
    parser.add_argument(
        "--test_manifest_csv", type=str, default=None, help="External test manifest CSV."
    )

    # Columns
    parser.add_argument(
        "--slide_id_column", type=str, default=None, help="Explicit slide_id column in manifest."
    )
    parser.add_argument(
        "--filename_column",
        type=str,
        default="filename",
        help="Filename column used to derive slide_id when slide_id_column is absent.",
    )
    parser.add_argument("--label_column", type=str, required=True, help="Integer label column.")
    parser.add_argument(
        "--patient_column", type=str, required=True, help="Patient/group ID column."
    )
    parser.add_argument(
        "--split_column",
        type=str,
        default=None,
        help="Optional manifest split column for protocol=predefined_split.",
    )

    # Evaluation protocol
    parser.add_argument(
        "--protocol",
        type=str,
        default="grouped_cv",
        choices=["grouped_cv", "external_test", "predefined_split"],
        help="Evaluation protocol.",
    )
    parser.add_argument("--n_splits", type=int, default=5, help="Outer grouped CV folds.")
    parser.add_argument(
        "--inner_splits",
        type=int,
        default=3,
        help="Inner grouped CV folds for C selection. Set 0 or 1 to disable tuning.",
    )
    parser.add_argument(
        "--primary_metric",
        type=str,
        default=None,
        choices=[None, "auroc", "auprc", "balanced_acc", "macro_f1", "qwk"],
        help="Model-selection metric. Defaults by task type.",
    )

    # Probe hyperparameters
    parser.add_argument(
        "--c_grid",
        type=float,
        nargs="+",
        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        help="Grid of LogisticRegression C values.",
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        default="balanced",
        help="class_weight passed to LogisticRegression, e.g. balanced or none.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=5000,
        help="Max iterations for LogisticRegression.",
    )
    parser.add_argument(
        "--multi_class_mode",
        type=str,
        default="auto",
        choices=["auto", "multinomial", "ovr"],
        help="Multiclass handling for LogisticRegression.",
    )

    # Robustness / IO
    parser.add_argument(
        "--non_singleton_policy",
        type=str,
        default="error",
        choices=["error", "mean", "first"],
        help="How to handle slides whose mmap entry has length > 1.",
    )
    parser.add_argument(
        "--allow_missing_mmap",
        action="store_true",
        help="Drop manifest rows missing from mmap instead of failing.",
    )
    parser.add_argument(
        "--allow_missing_manifest",
        action="store_true",
        help="Ignore mmap slide_ids missing from manifest instead of failing.",
    )
    # Optional per-cohort test column overrides
    parser.add_argument(
        "--test_slide_id_column", type=str, default=None, help="slide_id column for test manifest."
    )
    parser.add_argument(
        "--test_filename_column", type=str, default=None, help="filename column for test manifest."
    )
    parser.add_argument(
        "--test_label_column", type=str, default=None, help="label column for test manifest."
    )
    parser.add_argument(
        "--test_patient_column", type=str, default=None, help="patient column for test manifest."
    )

    parser.add_argument(
        "--n_boot",
        type=int,
        default=0,
        help="Bootstrap resamples for 95%% CIs. 0 = skip, 1000 = recommended for final reporting.",
    )
    parser.add_argument(
        "--label_fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of labeled training groups to use for fitting/tuning the probe. "
            "Validation and test sets remain full-size."
        ),
    )
    parser.add_argument(
        "--label_fraction_seed",
        type=int,
        default=None,
        help="Seed for deterministic label-fraction subsampling. Defaults to --seed.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for tie-breaks / fallbacks."
    )
    parser.add_argument(
        "--save_model", action="store_true", help="Save final sklearn model with joblib."
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Consolidated factory for the probe args Namespace.
#
# The probe's `run_grouped_cv` / `run_external_test` / `run_predefined_split`
# all consume an `argparse.Namespace`. Three call sites previously built that
# Namespace independently (Hydra cfg in `scripts/linear_probing.py`, dataclass
# in `LinearProbeEvalCallback`, CLI in `parse_args`) — drift between them was
# a real bug source. This factory is the single source of truth: the two
# non-argparse callers now go through it; argparse remains the CLI entry
# (its returned Namespace already matches by construction).
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_C_GRID: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
_DEFAULT_TRAIN_SPLIT_NAMES: tuple[str, ...] = ("train", "tr")
_DEFAULT_EVAL_SPLIT_NAMES: tuple[str, ...] = ("test", "te")


def make_probe_args(
    *,
    # Core IO
    mmap_dir: str | None = None,
    manifest_csv: str = "",
    output_dir: str = "",
    # External test cohort
    test_mmap_dir: str | None = None,
    test_manifest_csv: str | None = None,
    test_slide_id_column: str | None = None,
    test_filename_column: str | None = None,
    test_label_column: str | None = None,
    test_patient_column: str | None = None,
    # Manifest columns
    slide_id_column: str | None = None,
    filename_column: str = "filename",
    label_column: str = "label",
    patient_column: str = "patient_id",
    split_column: str | None = None,
    # Protocol
    protocol: str = "grouped_cv",
    n_splits: int = 5,
    inner_splits: int = 3,
    primary_metric: str | None = None,
    # Probe hyperparameters
    c_grid: list[float] | tuple[float, ...] | None = None,
    class_weight: str = "balanced",
    max_iter: int = 5000,
    multi_class_mode: str = "auto",
    # Robustness
    non_singleton_policy: str = "error",
    allow_missing_mmap: bool = False,
    allow_missing_manifest: bool = False,
    aggregate_to_patient: bool = True,
    # Predefined-split partition names
    train_split_names: list[str] | tuple[str, ...] | None = None,
    eval_split_names: list[str] | tuple[str, ...] | None = None,
    # Bootstrap / LDE / output
    n_boot: int = 0,
    label_fraction: float | None = None,
    label_fraction_seed: int | None = None,
    seed: int = 42,
    save_model: bool = False,
    verbose: bool = False,
) -> argparse.Namespace:
    """Build the args Namespace for `run_grouped_cv` / `run_external_test`
    / `run_predefined_split`. Single source of truth — kwargs mirror the
    `parse_args` CLI exactly so adding a new flag here also covers the CLI
    path automatically (and vice-versa).
    """
    return argparse.Namespace(
        mmap_dir=mmap_dir,
        manifest_csv=manifest_csv,
        output_dir=output_dir,
        test_mmap_dir=test_mmap_dir,
        test_manifest_csv=test_manifest_csv,
        test_slide_id_column=test_slide_id_column,
        test_filename_column=test_filename_column,
        test_label_column=test_label_column,
        test_patient_column=test_patient_column,
        slide_id_column=slide_id_column,
        filename_column=filename_column,
        label_column=label_column,
        patient_column=patient_column,
        split_column=split_column,
        protocol=protocol,
        n_splits=n_splits,
        inner_splits=inner_splits,
        primary_metric=primary_metric,
        c_grid=list(c_grid if c_grid is not None else _DEFAULT_C_GRID),
        class_weight=class_weight,
        max_iter=max_iter,
        multi_class_mode=multi_class_mode,
        non_singleton_policy=non_singleton_policy,
        allow_missing_mmap=allow_missing_mmap,
        allow_missing_manifest=allow_missing_manifest,
        aggregate_to_patient=aggregate_to_patient,
        train_split_names=list(
            train_split_names if train_split_names is not None else _DEFAULT_TRAIN_SPLIT_NAMES
        ),
        eval_split_names=list(
            eval_split_names if eval_split_names is not None else _DEFAULT_EVAL_SPLIT_NAMES
        ),
        n_boot=n_boot,
        label_fraction=label_fraction,
        label_fraction_seed=label_fraction_seed,
        seed=seed,
        save_model=save_model,
        verbose=verbose,
    )


@dataclass
class EmbeddingStore:
    slide_ids: list[str]
    embeddings: np.ndarray


def setup_logging(output_dir: Path, verbose: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    logger.handlers.clear()
    logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "linear_probe.log")
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)


def _resolve_slide_ids(
    df: pd.DataFrame, slide_id_column: str | None, filename_column: str
) -> pd.Series:
    if slide_id_column is not None:
        if slide_id_column not in df.columns:
            raise ValueError(
                f"slide_id_column='{slide_id_column}' not found. Available: {list(df.columns)}"
            )
        return df[slide_id_column].astype(str).map(_normalize_slide_id)

    if filename_column not in df.columns:
        raise ValueError(
            f"Neither slide_id_column nor filename_column='{filename_column}' is available. "
            f"Columns: {list(df.columns)}"
        )

    # Normalize separators and strip known file extensions only.
    # Preserves directory structure and non-extension dots (e.g. TCGA UUIDs).
    return df[filename_column].astype(str).map(_normalize_slide_id)


_KNOWN_EXTENSIONS = frozenset(
    {
        # Feature / data files
        ".h5",
        ".pt",
        ".npy",
        ".npz",
        ".parquet",
        # WSI formats
        ".svs",
        ".tiff",
        ".tif",
        ".ndpi",
        ".mrxs",
        ".vsi",
        ".scn",
        ".sdpc",
        ".bif",
    }
)


def _normalize_slide_id(x: str) -> str:
    """Normalize a slide identifier.

    - Converts backslashes to forward slashes
    - Strips known file extensions (.h5, .svs, .tiff, etc.)
    - Does NOT strip arbitrary suffixes (e.g. TCGA UUIDs like .8923A151-...)
    """
    x = str(x).strip().replace("\\", "/")
    for suffix in _KNOWN_EXTENSIONS:
        if x.lower().endswith(suffix):
            x = x[: -len(suffix)]
            break
    return x


def load_manifest(
    csv_path: str | Path,
    slide_id_column: str | None,
    filename_column: str,
    label_column: str,
    patient_column: str,
    split_column: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {label_column, patient_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest {csv_path} missing required columns: {sorted(missing)}")

    df = df.copy()
    df["slide_id"] = _resolve_slide_ids(df, slide_id_column, filename_column)
    df["label"] = df[label_column].astype(int)
    df["patient_id"] = df[patient_column].astype(str)
    if split_column is not None:
        if split_column not in df.columns:
            raise ValueError(
                f"split_column='{split_column}' not found in {csv_path}. Columns: {list(df.columns)}"
            )
        df["split"] = df[split_column].astype(str)

    dup = df["slide_id"].duplicated(keep=False)
    if dup.any():
        dup_ids = df.loc[dup, "slide_id"].head(10).tolist()
        raise ValueError(
            f"Manifest {csv_path} has duplicate slide_id rows. First examples: {dup_ids}"
        )

    return df


def load_embedding_mmap(
    mmap_dir: str | Path,
    non_singleton_policy: str = "error",
) -> EmbeddingStore:
    from oceanpath.data.mmap_builder import validate_mmap_dir

    mmap_dir = Path(mmap_dir)
    validate_mmap_dir(mmap_dir)

    idx = np.load(str(mmap_dir / "index_arrays.npz"), allow_pickle=True)
    slide_ids = idx["slide_ids"].tolist()
    if len(slide_ids) != len(set(slide_ids)):
        raise ValueError(f"Embedding mmap {mmap_dir} contains duplicate slide_ids.")
    lengths = idx["lengths"].astype(int)
    feat_chunk_ids = idx["feat_chunk_ids"].astype(int)
    feat_offsets = idx["feat_offsets"].astype(int)
    feat_dim = int(idx["feat_dim"])
    feat_dtype = np.dtype(str(idx["feat_dtype"][0]))
    elem_size = feat_dtype.itemsize
    n_feat_chunks = int(idx["num_feat_chunks"])

    feat_mmaps = [
        np.memmap(str(mmap_dir / f"features_{i:03d}.bin"), dtype=feat_dtype, mode="r")
        for i in range(n_feat_chunks)
    ]

    embeddings = np.empty((len(slide_ids), feat_dim), dtype=np.float32)
    non_singletons: list[tuple[str, int]] = []

    for i, sid in enumerate(slide_ids):
        n_rows = int(lengths[i])
        chunk_id = int(feat_chunk_ids[i])
        byte_offset = int(feat_offsets[i])
        mm = feat_mmaps[chunk_id]
        start_elem = byte_offset // elem_size
        n_elems = n_rows * feat_dim
        arr = mm[start_elem : start_elem + n_elems].reshape(n_rows, feat_dim)

        if n_rows == 1:
            vec = np.asarray(arr[0], dtype=np.float32)
        else:
            non_singletons.append((sid, n_rows))
            if non_singleton_policy == "error":
                raise ValueError(
                    f"Embedding mmap contains slide '{sid}' with length={n_rows}. "
                    f"Expected one embedding row per slide. Use --non_singleton_policy mean|first if intentional."
                )
            if non_singleton_policy == "mean":
                vec = np.asarray(arr.mean(axis=0), dtype=np.float32)
            elif non_singleton_policy == "first":
                vec = np.asarray(arr[0], dtype=np.float32)
            else:
                raise ValueError(f"Unknown non_singleton_policy: {non_singleton_policy}")

        if not np.isfinite(vec).all():
            raise ValueError(
                f"Embedding mmap contains non-finite values for slide '{sid}'. "
                f"Rebuild the embedding cache before probing."
            )
        embeddings[i] = vec

    if non_singletons:
        logger.warning(
            "Found %d non-singleton mmap entries. Policy='%s'. First examples: %s",
            len(non_singletons),
            non_singleton_policy,
            non_singletons[:5],
        )

    return EmbeddingStore(slide_ids=slide_ids, embeddings=embeddings)


def merge_manifest_and_embeddings(
    manifest_df: pd.DataFrame,
    store: EmbeddingStore,
    allow_missing_mmap: bool = False,
    allow_missing_manifest: bool = False,
) -> pd.DataFrame:
    embed_df = pd.DataFrame({"slide_id": store.slide_ids})
    embed_df["embedding"] = list(store.embeddings)

    merged = manifest_df.merge(embed_df, on="slide_id", how="left", validate="one_to_one")

    missing_in_mmap = merged["embedding"].isna()
    if missing_in_mmap.any():
        missing_ids = merged.loc[missing_in_mmap, "slide_id"].head(10).tolist()
        msg = (
            f"{missing_in_mmap.sum()} manifest slide_ids were not found in mmap. "
            f"First examples: {missing_ids}"
        )
        if allow_missing_mmap:
            logger.warning(msg + " -- dropping them.")
            merged = merged.loc[~missing_in_mmap].copy()
        else:
            raise ValueError(msg)

    mmap_only = sorted(set(store.slide_ids) - set(manifest_df["slide_id"].tolist()))
    if mmap_only:
        msg = f"{len(mmap_only)} mmap slide_ids are missing from the manifest. First examples: {mmap_only[:10]}"
        if allow_missing_manifest:
            logger.warning(msg + " -- ignoring them.")
        else:
            raise ValueError(msg)

    if merged.empty:
        raise ValueError(
            "No rows remain after merging manifest and embeddings. "
            "Check slide_id conventions and allow_missing_* settings."
        )

    merged["embedding"] = merged["embedding"].map(lambda x: np.asarray(x, dtype=np.float32))
    return merged.reset_index(drop=True)


def infer_task_type(y: np.ndarray) -> str:
    n_classes = len(np.unique(y))
    return "binary" if n_classes == 2 else "multiclass"


def _ensure_trainable_labels(y: np.ndarray, context: str) -> None:
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(
            f"{context} contains only one class {classes.tolist()}. "
            "The linear probe cannot be trained on a single-class split."
        )


def _validate_label_space(train_y: np.ndarray, eval_y: np.ndarray, context: str) -> None:
    train_classes = set(np.unique(train_y).tolist())
    eval_classes = set(np.unique(eval_y).tolist())
    unseen = sorted(eval_classes - train_classes)
    if unseen:
        raise ValueError(
            f"{context} contains labels not present in the training split: {unseen}. "
            f"Training classes: {sorted(train_classes)}"
        )


def _assert_binary_labels_01(y: np.ndarray, context: str = "") -> None:
    classes = np.unique(y)
    if set(classes.tolist()) != {0, 1}:
        raise ValueError(f"Binary labels must be encoded as {{0, 1}}, got {classes}. {context}")


def default_primary_metric(task_type: str) -> str:
    if task_type == "binary":
        return "auroc"
    return "macro_f1"


def make_probe(
    C: float,
    task_type: str,
    class_weight: str | None,
    max_iter: int,
    multi_class_mode: str,
    seed: int = 42,
) -> Pipeline:
    cw = None if class_weight in (None, "none", "None") else class_weight
    kwargs: dict = {
        "C": C,
        "class_weight": cw,
        "solver": "lbfgs",
        "max_iter": max_iter,
        "random_state": seed,
    }

    # sklearn >= 1.5 deprecated multi_class (always multinomial).
    # Only pass it for older versions or explicit non-default values.
    import sklearn

    sklearn_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if task_type != "binary" and sklearn_version < (1, 5):
        if multi_class_mode == "auto":
            multi_class_mode = "multinomial"
        kwargs["multi_class"] = multi_class_mode

    clf = LogisticRegression(**kwargs)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def pick_group_cv(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int = 42):
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(f"Need at least {n_splits} unique groups, found {len(unique_groups)}.")

    try:
        from sklearn.model_selection import StratifiedGroupKFold  # type: ignore

        counts = pd.Series(y).value_counts()
        if counts.min() >= n_splits:
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        logger.warning(
            "Falling back to GroupKFold because the rarest class has only %d examples (< n_splits=%d).",
            int(counts.min()),
            n_splits,
        )
    except Exception:
        logger.warning("StratifiedGroupKFold unavailable. Falling back to GroupKFold.")

    # Seed the GroupKFold fallback so folds aren't a deterministic function of
    # patient_id arrival order. sklearn ≥ 1.4 supports shuffle/random_state on
    # GroupKFold; older versions raise — fall back to the unshuffled default
    # there so we don't crash, and surface a one-time warning.
    try:
        return GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    except TypeError:
        logger.warning(
            "Installed sklearn's GroupKFold does not support shuffle=/random_state=; "
            "fold assignment will be deterministic on patient_id arrival order. "
            "Upgrade to sklearn>=1.4 for seed-controlled shuffling."
        )
        return GroupKFold(n_splits=n_splits)


@dataclass
class PatientPredictions:
    patient_ids: np.ndarray
    y_true: np.ndarray
    prob: np.ndarray
    y_pred: np.ndarray


def to_slide_level_predictions(
    slide_df: pd.DataFrame,
    prob: np.ndarray,
    task_type: str,
) -> PatientPredictions:
    """Slide-level evaluation: one prediction per slide, no patient collapse.

    Returns a PatientPredictions where `patient_ids` actually holds slide
    identifiers, so downstream metric / output code is unchanged.
    """
    sid = (
        slide_df["slide_id"].to_numpy()
        if "slide_id" in slide_df.columns
        else np.arange(len(slide_df))
    )
    y_true = slide_df["label"].to_numpy(dtype=int)

    if task_type == "binary":
        prob = np.asarray(prob, dtype=np.float64).reshape(-1)
        y_pred = (prob >= 0.5).astype(int)
    else:
        prob = np.asarray(prob, dtype=np.float64)
        y_pred = prob.argmax(axis=1)

    return PatientPredictions(
        patient_ids=sid,
        y_true=y_true,
        prob=prob,
        y_pred=y_pred,
    )


def reduce_predictions(
    slide_df: pd.DataFrame,
    prob: np.ndarray,
    task_type: str,
    aggregate_to_patient: bool,
) -> PatientPredictions:
    if aggregate_to_patient:
        return aggregate_to_patient_level(slide_df, prob, task_type)
    return to_slide_level_predictions(slide_df, prob, task_type)


def aggregate_to_patient_level(
    slide_df: pd.DataFrame,
    prob: np.ndarray,
    task_type: str,
) -> PatientPredictions:
    bad = slide_df.groupby("patient_id")["label"].nunique()
    bad = bad[bad > 1]
    if len(bad) > 0:
        raise ValueError(
            f"Found patients with conflicting slide labels. "
            f"Examples: {bad.head().to_dict()}. "
            f"Set aggregate_to_patient=false to evaluate at slide level."
        )

    df = slide_df[["patient_id", "label", "slide_id"]].copy()

    if task_type == "binary":
        prob = np.asarray(prob, dtype=np.float64).reshape(-1)
        df["prob_1"] = prob

        patient = (
            df.groupby("patient_id", sort=False)
            .agg(
                label=("label", "first"), prob_1=("prob_1", "mean"), n_slides=("slide_id", "count")
            )
            .reset_index()
        )
        y_prob = patient["prob_1"].to_numpy(dtype=np.float64)
        y_pred = (y_prob >= 0.5).astype(int)
        return PatientPredictions(
            patient_ids=patient["patient_id"].to_numpy(),
            y_true=patient["label"].to_numpy(dtype=int),
            prob=y_prob,
            y_pred=y_pred,
        )

    prob = np.asarray(prob, dtype=np.float64)
    prob_cols = [f"prob_{i}" for i in range(prob.shape[1])]
    for i, c in enumerate(prob_cols):
        df[c] = prob[:, i]

    agg_spec: dict[str, tuple[str, str]] = {
        "label": ("label", "first"),
        "n_slides": ("slide_id", "count"),
    }
    for c in prob_cols:
        agg_spec[c] = (c, "mean")

    patient = df.groupby("patient_id", sort=False).agg(**agg_spec).reset_index()
    y_prob = patient[prob_cols].to_numpy(dtype=np.float64)
    y_pred = y_prob.argmax(axis=1)
    return PatientPredictions(
        patient_ids=patient["patient_id"].to_numpy(),
        y_true=patient["label"].to_numpy(dtype=int),
        prob=y_prob,
        y_pred=y_pred,
    )


def compute_metrics_from_patient_preds(pp: PatientPredictions, task_type: str) -> dict[str, Any]:
    y_true = pp.y_true
    y_pred = pp.y_pred
    out: dict[str, Any] = {
        "n_patients": len(y_true),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if task_type == "binary":
        y_prob = np.asarray(pp.prob, dtype=np.float64).reshape(-1)
        out["auroc"] = _safe_binary_auroc(y_true, y_prob)
        out["auprc"] = _safe_binary_auprc(y_true, y_prob)
    else:
        y_prob = np.asarray(pp.prob, dtype=np.float64)
        out["qwk"] = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
        out["auroc_macro_ovr"] = _safe_multiclass_auroc(y_true, y_prob)

    return out


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float | None, float | None]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            vals.append(metric_fn(y_true[idx], y_prob[idx]))
        except ValueError:
            continue

    if len(vals) == 0:
        return None, None

    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _safe_binary_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return None


def _safe_binary_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return None


def _safe_multiclass_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        return None


def get_primary_metric_value(metrics: dict[str, Any], primary_metric: str) -> float:
    value = metrics.get(primary_metric)
    if value is None:
        return float("-inf")
    fv = float(value)
    # QWK can be NaN when the probe collapses to one class (no prediction variance).
    # Treat as "missing" so model selection sees it as worse than any finite score.
    if math.isnan(fv):
        return float("-inf")
    return fv


def embeddings_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["embedding"].to_list()).astype(np.float32, copy=False)


def _validate_label_fraction(label_fraction: float | None) -> float:
    if label_fraction is None:
        return 1.0
    fraction = float(label_fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"label_fraction must be in (0, 1], got {label_fraction!r}.")
    return fraction


def _label_fraction_seed(args: argparse.Namespace, fallback_seed: int) -> int:
    seed = getattr(args, "label_fraction_seed", None)
    if seed is None:
        seed = fallback_seed
    return int(seed)


def _group_label_for_sampling(labels: pd.Series) -> int:
    counts = labels.astype(int).value_counts()
    max_count = int(counts.max())
    winners = sorted(int(x) for x in counts[counts == max_count].index.tolist())
    return winners[0]


def _labels_are_categorical(labels: pd.Series) -> bool:
    """True when labels are clean integer-valued (binary/multiclass/ordinal)."""
    arr = pd.to_numeric(labels, errors="coerce")
    if arr.isna().any():
        return False
    return bool(np.allclose(arr.to_numpy(), arr.astype(int).to_numpy()))


def subsample_labeled_training_data(
    train_df: pd.DataFrame,
    label_fraction: float | None,
    seed: int,
    context: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Subsample labeled training data at patient/group level.

    All slides for a selected patient are kept together. The validation/test
    cohorts are intentionally not handled here, so downstream evaluation still
    measures performance on the full held-out split.
    """
    fraction = _validate_label_fraction(label_fraction)
    full_slides = len(train_df)
    full_groups = int(train_df["patient_id"].nunique())
    full_classes = int(train_df["label"].nunique())
    categorical = _labels_are_categorical(train_df["label"])

    base_info: dict[str, Any] = {
        "label_fraction": fraction,
        "label_fraction_seed": int(seed),
        "label_fraction_stratified": bool(categorical),
        "train_full_slides": full_slides,
        "train_full_patients": full_groups,
        "train_full_classes": full_classes,
    }

    if train_df.empty:
        raise ValueError(f"{context} is empty; cannot train a linear probe.")

    if fraction >= 1.0:
        info = {
            **base_info,
            "train_labeled_slides": full_slides,
            "train_labeled_patients": full_groups,
            "train_labeled_classes": full_classes,
            "train_labeled_slide_fraction": 1.0,
            "train_labeled_patient_fraction": 1.0,
            "train_mixed_label_patients_full": int(
                (train_df.groupby("patient_id")["label"].nunique() > 1).sum()
            ),
            "train_mixed_label_patients_labeled": int(
                (train_df.groupby("patient_id")["label"].nunique() > 1).sum()
            ),
        }
        return train_df.reset_index(drop=True), info

    rng = np.random.default_rng(seed)

    if not categorical:
        # Continuous / regression labels: cannot stratify by class. Fall back to
        # uniform random patient-level subsampling.
        all_ids = np.array(
            sorted(train_df["patient_id"].astype(str).unique().tolist()),
            dtype=object,
        )
        rng.shuffle(all_ids)
        n_take = max(1, math.ceil(len(all_ids) * fraction))
        selected = {str(x) for x in all_ids[:n_take]}
        sampled_df = train_df[train_df["patient_id"].astype(str).isin(selected)].reset_index(
            drop=True
        )

        if sampled_df.empty:
            raise ValueError(
                f"{context}: label_fraction={fraction:g} selected zero rows. "
                "Use a larger fraction or check patient_id values."
            )

        info = {
            **base_info,
            "train_labeled_slides": len(sampled_df),
            "train_labeled_patients": int(sampled_df["patient_id"].nunique()),
            "train_labeled_classes": int(sampled_df["label"].nunique()),
            "train_labeled_slide_fraction": float(len(sampled_df) / max(1, full_slides)),
            "train_labeled_patient_fraction": float(
                sampled_df["patient_id"].nunique() / max(1, full_groups)
            ),
            "train_mixed_label_patients_full": int(
                (train_df.groupby("patient_id")["label"].nunique() > 1).sum()
            ),
            "train_mixed_label_patients_labeled": int(
                (sampled_df.groupby("patient_id")["label"].nunique() > 1).sum()
            ),
        }
        logger.info(
            "%s | label_fraction=%g (continuous, unstratified) | labeled=%d/%d slides, %d/%d patients",
            context,
            fraction,
            info["train_labeled_slides"],
            full_slides,
            info["train_labeled_patients"],
            full_groups,
        )
        return sampled_df, info

    group_rows: list[dict[str, Any]] = []
    for patient_id, group in train_df.groupby("patient_id", sort=True):
        labels = group["label"].astype(int)
        group_rows.append(
            {
                "patient_id": str(patient_id),
                "sample_label": _group_label_for_sampling(labels),
                "n_slides": len(group),
                "mixed_labels": bool(labels.nunique() > 1),
            }
        )

    group_df = pd.DataFrame(group_rows)
    selected_ids: list[str] = []

    # Per-class quotas keep low-label splits trainable. For tiny fractions
    # this deliberately rounds up via `max(1, ceil(n_class * fraction))` per
    # class, AND the missing-class backfill loop below adds at most one extra
    # patient per absent class. As a result, the *actual* labeled patient
    # fraction is systematically >= the requested `fraction` — most
    # noticeably at very small fractions (e.g. 0.01 with K classes can yield
    # K patients out of N rather than the expected 0.01*N).
    #
    # The realized fraction is captured in the returned `info` dict as
    # `mean_train_labeled_patient_fraction` (see `_summary_record_info`); use
    # THAT key as the x-axis in label-efficiency plots, not the requested
    # `fraction` argument. `scripts/collect_lp_results.py` honors this.
    for _, class_df in group_df.groupby("sample_label", sort=True):
        ids = class_df["patient_id"].astype(str).to_numpy(dtype=object)
        rng.shuffle(ids)
        n_take = max(1, math.ceil(len(ids) * fraction))
        selected_ids.extend(str(x) for x in ids[:n_take])

    selected = set(selected_ids)
    sampled_df = train_df[train_df["patient_id"].astype(str).isin(selected)].reset_index(drop=True)

    missing_labels = sorted(
        set(train_df["label"].astype(int)) - set(sampled_df["label"].astype(int))
    )
    for label in missing_labels:
        candidates = sorted(
            train_df.loc[train_df["label"].astype(int) == label, "patient_id"]
            .astype(str)
            .unique()
            .tolist()
        )
        unselected = [pid for pid in candidates if pid not in selected]
        pool = unselected or candidates
        if pool:
            selected.add(str(rng.choice(pool)))

    sampled_df = train_df[train_df["patient_id"].astype(str).isin(selected)].reset_index(drop=True)

    if sampled_df.empty:
        raise ValueError(
            f"{context}: label_fraction={fraction:g} selected zero rows. "
            "Use a larger fraction or check patient_id values."
        )

    info = {
        **base_info,
        "train_labeled_slides": len(sampled_df),
        "train_labeled_patients": int(sampled_df["patient_id"].nunique()),
        "train_labeled_classes": int(sampled_df["label"].nunique()),
        "train_labeled_slide_fraction": float(len(sampled_df) / max(1, full_slides)),
        "train_labeled_patient_fraction": float(
            sampled_df["patient_id"].nunique() / max(1, full_groups)
        ),
        "train_mixed_label_patients_full": int(group_df["mixed_labels"].sum()),
        "train_mixed_label_patients_labeled": int(
            group_df[group_df["patient_id"].isin(selected)]["mixed_labels"].sum()
        ),
    }

    logger.info(
        "%s | label_fraction=%g | labeled=%d/%d slides, %d/%d patients",
        context,
        fraction,
        info["train_labeled_slides"],
        full_slides,
        info["train_labeled_patients"],
        full_groups,
    )

    return sampled_df, info


def select_best_c(
    train_df: pd.DataFrame,
    task_type: str,
    c_grid: list[float],
    inner_splits: int,
    primary_metric: str,
    class_weight: str | None,
    max_iter: int,
    multi_class_mode: str,
    seed: int = 42,
    X_train: np.ndarray | None = None,
    aggregate_to_patient: bool = True,
) -> tuple[float, pd.DataFrame]:
    if inner_splits is None or inner_splits <= 1 or len(c_grid) == 1:
        chosen = float(c_grid[0])
        return chosen, pd.DataFrame({"C": [chosen], "mean_inner_score": [np.nan]})

    if X_train is None:
        X_train = embeddings_matrix(train_df)
    y = train_df["label"].to_numpy(dtype=int)
    _ensure_trainable_labels(y, context="Inner CV training pool")
    groups = train_df["patient_id"].to_numpy()

    # Graceful fallback for low-label regimes: when there are not enough
    # patients/groups or the rarest class is too thin to honor `inner_splits`,
    # skip the inner CV and pick the geometric midpoint of the C grid.
    n_groups = int(np.unique(groups).size)
    min_class_count = int(pd.Series(y).value_counts().min())
    if n_groups < inner_splits or min_class_count < inner_splits:
        sorted_grid = sorted(float(c) for c in c_grid)
        chosen = float(sorted_grid[len(sorted_grid) // 2])
        logger.warning(
            "Skipping inner CV: n_groups=%d, min_class_count=%d, inner_splits=%d. "
            "Falling back to grid midpoint C=%g.",
            n_groups,
            min_class_count,
            inner_splits,
            chosen,
        )
        return chosen, pd.DataFrame(
            {
                "C": [chosen],
                "mean_inner_score": [np.nan],
                "fallback": [True],
                "n_groups": [n_groups],
                "min_class_count": [min_class_count],
            }
        )

    splitter = pick_group_cv(y, groups, inner_splits, seed=seed)

    rows: list[dict[str, Any]] = []
    c_fold_scores: dict[float, list[float]] = {float(C): [] for C in c_grid}

    # Sort C descending for warm-start: large C (less regularisation) first,
    # so the LBFGS solution shrinks smoothly toward stronger regularisation.
    sorted_c = sorted(c_grid, reverse=True)

    for inner_fold, (tr_idx, va_idx) in enumerate(splitter.split(X_train, y, groups)):
        va_df = train_df.iloc[va_idx].reset_index(drop=True)
        tr_y = y[tr_idx]
        va_y = y[va_idx]
        _ensure_trainable_labels(tr_y, context=f"Inner fold {inner_fold} train split")
        _validate_label_space(tr_y, va_y, context=f"Inner fold {inner_fold} validation split")

        # Fit scaler once per fold; reuse across C values
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train[tr_idx])
        X_va_scaled = scaler.transform(X_train[va_idx])

        # Extract a properly-configured LogisticRegression via make_probe,
        # then enable warm_start to amortise LBFGS across the C sweep.
        probe = make_probe(
            sorted_c[0], task_type, class_weight, max_iter, multi_class_mode, seed=seed
        )
        clf = probe.named_steps["clf"]
        clf.warm_start = True

        for C in sorted_c:
            clf.C = C
            clf.fit(X_tr_scaled, tr_y)
            prob = clf.predict_proba(X_va_scaled)
            if task_type == "binary":
                prob = prob[:, 1]
            pp = reduce_predictions(va_df, prob, task_type, aggregate_to_patient)
            metrics = compute_metrics_from_patient_preds(pp, task_type)
            score = get_primary_metric_value(metrics, primary_metric)
            c_fold_scores[float(C)].append(score)
            rows.append(
                {
                    "C": float(C),
                    "inner_fold": int(inner_fold),
                    "primary_metric": primary_metric,
                    "score": score,
                }
            )

    best_c = None
    best_score = float("-inf")
    for C in c_grid:
        fold_scores = c_fold_scores[float(C)]
        mean_score = float(np.mean(fold_scores)) if fold_scores else float("-inf")
        logger.info("Inner CV | C=%g | %s=%.4f", C, primary_metric, mean_score)

        if mean_score > best_score or (
            math.isclose(mean_score, best_score) and (best_c is None or best_c > C)
        ):
            best_score = mean_score
            best_c = float(C)

    if best_c is None:
        # No C value produced a finite score (e.g., every inner fold's probe
        # collapsed to one class, making the metric undefined). Fall back to the
        # geometric midpoint of the C grid so the outer fold still completes.
        sorted_grid = sorted(float(c) for c in c_grid)
        best_c = float(sorted_grid[len(sorted_grid) // 2])
        logger.warning(
            "Inner CV produced no finite scores for any C; falling back to grid midpoint C=%g.",
            best_c,
        )

    summary = pd.DataFrame(rows)
    return best_c, summary


def summarize_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not fold_metrics:
        return {}

    numeric_keys = sorted(
        {
            k
            for fm in fold_metrics
            for k, v in fm.items()
            if isinstance(v, (int, float, np.floating))
            and k not in {"fold", "best_C", "n_patients"}
        }
    )
    summary: dict[str, Any] = {
        "n_folds": len(fold_metrics),
    }
    for k in numeric_keys:
        # Drop None and non-finite values (NaN, ±inf). A fold's metric can be
        # NaN when the probe degenerates to constant predictions (e.g., QWK
        # divides by zero when one column of the confusion matrix is empty);
        # silently averaging NaN poisons every aggregate. Track how many folds
        # contributed via n_valid_<metric> so degenerate folds stay visible.
        vals: list[float] = []
        for fm in fold_metrics:
            v = fm.get(k)
            if v is None:
                continue
            fv = float(v)
            if not math.isfinite(fv):
                continue
            vals.append(fv)
        if not vals:
            continue
        summary[f"mean_{k}"] = float(np.mean(vals))
        summary[f"std_{k}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        if len(vals) < len(fold_metrics):
            summary[f"n_valid_{k}"] = len(vals)
    return summary


def run_grouped_cv(args: argparse.Namespace, df: pd.DataFrame) -> dict[str, Any]:
    X = embeddings_matrix(df)
    y = df["label"].to_numpy(dtype=int)
    groups = df["patient_id"].to_numpy()
    task_type = infer_task_type(y)
    if task_type == "binary":
        _assert_binary_labels_01(y, context="(grouped_cv train data)")
    primary_metric = args.primary_metric or default_primary_metric(task_type)
    seed = getattr(args, "seed", 42)
    label_fraction = _validate_label_fraction(getattr(args, "label_fraction", 1.0))
    base_label_fraction_seed = _label_fraction_seed(args, seed)
    aggregate_to_patient = getattr(args, "aggregate_to_patient", True)
    splitter = pick_group_cv(y, groups, args.n_splits, seed=seed)

    fold_metrics: list[dict[str, Any]] = []
    all_patient_preds: list[pd.DataFrame] = []
    all_inner_search: list[pd.DataFrame] = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y, groups)):
        full_train_df = df.iloc[tr_idx].reset_index(drop=True)
        train_df, label_fraction_info = subsample_labeled_training_data(
            train_df=full_train_df,
            label_fraction=label_fraction,
            seed=base_label_fraction_seed + fold,
            context=f"Outer fold {fold} train split",
        )
        val_df = df.iloc[va_idx].reset_index(drop=True)
        X_train = embeddings_matrix(train_df)
        X_val = X[va_idx]
        tr_y = train_df["label"].to_numpy(dtype=int)
        va_y = val_df["label"].to_numpy(dtype=int)
        _ensure_trainable_labels(tr_y, context=f"Outer fold {fold} train split")
        _validate_label_space(tr_y, va_y, context=f"Outer fold {fold} validation split")

        best_c, inner_search_df = select_best_c(
            train_df=train_df,
            task_type=task_type,
            c_grid=list(args.c_grid),
            inner_splits=args.inner_splits,
            primary_metric=primary_metric,
            class_weight=args.class_weight,
            max_iter=args.max_iter,
            multi_class_mode=args.multi_class_mode,
            seed=seed,
            X_train=X_train,
            aggregate_to_patient=aggregate_to_patient,
        )
        inner_search_df = inner_search_df.copy()
        inner_search_df["outer_fold"] = fold
        all_inner_search.append(inner_search_df)

        model = make_probe(
            C=best_c,
            task_type=task_type,
            class_weight=args.class_weight,
            max_iter=args.max_iter,
            multi_class_mode=args.multi_class_mode,
            seed=seed,
        )
        model.fit(X_train, train_df["label"].to_numpy(dtype=int))
        prob = model.predict_proba(X_val)
        if task_type == "binary":
            prob = prob[:, 1]

        patient_preds = reduce_predictions(val_df, prob, task_type, aggregate_to_patient)
        metrics = compute_metrics_from_patient_preds(patient_preds, task_type)
        metrics.update(
            {
                "fold": int(fold),
                "best_C": float(best_c),
                "primary_metric": primary_metric,
                **label_fraction_info,
            }
        )
        fold_metrics.append(metrics)

        fold_pred_df = pd.DataFrame(
            {
                "outer_fold": fold,
                "patient_id": patient_preds.patient_ids,
                "label": patient_preds.y_true,
                "pred": patient_preds.y_pred,
            }
        )
        if task_type == "binary":
            fold_pred_df["prob_1"] = patient_preds.prob
        else:
            for c in range(patient_preds.prob.shape[1]):
                fold_pred_df[f"prob_{c}"] = patient_preds.prob[:, c]
        all_patient_preds.append(fold_pred_df)

        if getattr(args, "save_model", False):
            model_path = Path(args.output_dir) / f"linear_probe_fold{fold}.joblib"
            joblib.dump(model, model_path)
            logger.info("Saved fold %d model -> %s", fold, model_path)

        key_score = get_primary_metric_value(metrics, primary_metric)
        logger.info(
            "Outer fold %d | best_C=%g | %s=%.4f",
            fold,
            best_c,
            primary_metric,
            key_score,
        )

    summary = summarize_fold_metrics(fold_metrics)

    n_boot = getattr(args, "n_boot", 0)
    if n_boot > 0 and task_type == "binary" and all_patient_preds:
        oof_df = pd.concat(all_patient_preds, ignore_index=True)
        oof_y = oof_df["label"].to_numpy(dtype=int)
        oof_prob = oof_df["prob_1"].to_numpy(dtype=np.float64)
        auroc_ci = bootstrap_metric_ci(oof_y, oof_prob, roc_auc_score, n_boot=n_boot, seed=seed)
        summary["oof_auroc_ci_low"] = auroc_ci[0]
        summary["oof_auroc_ci_high"] = auroc_ci[1]
        auprc_ci = bootstrap_metric_ci(
            oof_y, oof_prob, average_precision_score, n_boot=n_boot, seed=seed
        )
        summary["oof_auprc_ci_low"] = auprc_ci[0]
        summary["oof_auprc_ci_high"] = auprc_ci[1]

    summary.update(
        {
            "protocol": "grouped_cv",
            "task_type": task_type,
            "primary_metric": primary_metric,
            "label_fraction": label_fraction,
            "label_fraction_seed": base_label_fraction_seed,
            "fold_metrics": fold_metrics,
        }
    )

    pd.DataFrame(fold_metrics).to_csv(Path(args.output_dir) / "fold_metrics.csv", index=False)
    pd.concat(all_patient_preds, ignore_index=True).to_csv(
        Path(args.output_dir) / "patient_predictions_oof.csv", index=False
    )
    if all_inner_search:
        pd.concat(all_inner_search, ignore_index=True).to_csv(
            Path(args.output_dir) / "inner_cv_search.csv", index=False
        )

    return summary


def run_external_test(
    args: argparse.Namespace, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict[str, Any]:
    seed = getattr(args, "seed", 42)
    label_fraction = _validate_label_fraction(getattr(args, "label_fraction", 1.0))
    label_fraction_seed = _label_fraction_seed(args, seed)
    full_train_df = train_df.reset_index(drop=True)
    train_df, label_fraction_info = subsample_labeled_training_data(
        train_df=full_train_df,
        label_fraction=label_fraction,
        seed=label_fraction_seed,
        context="External-test train split",
    )

    X_train = embeddings_matrix(train_df)
    X_test = embeddings_matrix(test_df)
    y_train = train_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)
    _ensure_trainable_labels(y_train, context="External-test train split")
    _validate_label_space(y_train, y_test, context="External-test evaluation split")
    task_type = infer_task_type(y_train)
    if task_type == "binary":
        _assert_binary_labels_01(y_train, context="(external_test train data)")
        _assert_binary_labels_01(y_test, context="(external_test test data)")
    primary_metric = args.primary_metric or default_primary_metric(task_type)
    aggregate_to_patient = getattr(args, "aggregate_to_patient", True)

    best_c, inner_search_df = select_best_c(
        train_df=train_df,
        task_type=task_type,
        c_grid=list(args.c_grid),
        inner_splits=args.inner_splits,
        primary_metric=primary_metric,
        class_weight=args.class_weight,
        max_iter=args.max_iter,
        multi_class_mode=args.multi_class_mode,
        seed=seed,
        X_train=X_train,
        aggregate_to_patient=aggregate_to_patient,
    )

    model = make_probe(
        C=best_c,
        task_type=task_type,
        class_weight=args.class_weight,
        max_iter=args.max_iter,
        multi_class_mode=args.multi_class_mode,
        seed=seed,
    )
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)
    if task_type == "binary":
        prob = prob[:, 1]

    patient_preds = reduce_predictions(test_df, prob, task_type, aggregate_to_patient)
    metrics = compute_metrics_from_patient_preds(patient_preds, task_type)

    n_boot = getattr(args, "n_boot", 0)
    if n_boot > 0 and task_type == "binary":
        y_prob = np.asarray(patient_preds.prob, dtype=np.float64).reshape(-1)
        auroc_ci = bootstrap_metric_ci(
            patient_preds.y_true, y_prob, roc_auc_score, n_boot=n_boot, seed=seed
        )
        metrics["auroc_ci_low"] = auroc_ci[0]
        metrics["auroc_ci_high"] = auroc_ci[1]
        auprc_ci = bootstrap_metric_ci(
            patient_preds.y_true, y_prob, average_precision_score, n_boot=n_boot, seed=seed
        )
        metrics["auprc_ci_low"] = auprc_ci[0]
        metrics["auprc_ci_high"] = auprc_ci[1]

    metrics.update(
        {
            "protocol": "external_test",
            "task_type": task_type,
            "primary_metric": primary_metric,
            "best_C": float(best_c),
            "label_fraction": label_fraction,
            "label_fraction_seed": label_fraction_seed,
            "n_train_patients": int(train_df["patient_id"].nunique()),
            "n_train_patients_full": int(full_train_df["patient_id"].nunique()),
            "n_train_slides": len(train_df),
            "n_train_slides_full": len(full_train_df),
            "n_test_patients": int(test_df["patient_id"].nunique()),
            **label_fraction_info,
        }
    )

    patient_pred_df = pd.DataFrame(
        {
            "patient_id": patient_preds.patient_ids,
            "label": patient_preds.y_true,
            "pred": patient_preds.y_pred,
        }
    )
    if task_type == "binary":
        patient_pred_df["prob_1"] = patient_preds.prob
    else:
        for c in range(patient_preds.prob.shape[1]):
            patient_pred_df[f"prob_{c}"] = patient_preds.prob[:, c]

    inner_search_df.to_csv(Path(args.output_dir) / "inner_cv_search.csv", index=False)
    patient_pred_df.to_csv(Path(args.output_dir) / "patient_predictions_test.csv", index=False)

    if args.save_model:
        joblib.dump(model, Path(args.output_dir) / "linear_probe_model.joblib")
        logger.info(
            "Saved fitted probe model → %s", Path(args.output_dir) / "linear_probe_model.joblib"
        )

    return metrics


def run_predefined_split(args: argparse.Namespace, df: pd.DataFrame) -> dict[str, Any]:
    if "split" not in df.columns:
        raise ValueError("protocol=predefined_split requires --split_column.")

    train_names = getattr(args, "train_split_names", None) or ["train", "tr"]
    eval_names = getattr(args, "eval_split_names", None) or ["test", "te"]

    train_mask = df["split"].str.lower().isin(train_names)
    test_mask = df["split"].str.lower().isin(eval_names)

    if not train_mask.any() or not test_mask.any():
        unique_vals = sorted(df["split"].str.lower().unique())
        raise ValueError(
            f"Predefined split manifest must contain rows matching "
            f"train_split_names={train_names} and eval_split_names={eval_names}. "
            f"Found split values: {unique_vals}"
        )

    train_df = df.loc[train_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)
    return run_external_test(args, train_df, test_df)


def save_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    def convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=convert)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    setup_logging(output_dir, args.verbose)

    logger.info("Loading primary manifest: %s", args.manifest_csv)
    manifest_df = load_manifest(
        csv_path=args.manifest_csv,
        slide_id_column=args.slide_id_column,
        filename_column=args.filename_column,
        label_column=args.label_column,
        patient_column=args.patient_column,
        split_column=args.split_column if args.protocol == "predefined_split" else None,
    )
    logger.info("Loading primary embedding mmap: %s", args.mmap_dir)
    store = load_embedding_mmap(args.mmap_dir, non_singleton_policy=args.non_singleton_policy)
    df = merge_manifest_and_embeddings(
        manifest_df=manifest_df,
        store=store,
        allow_missing_mmap=args.allow_missing_mmap,
        allow_missing_manifest=args.allow_missing_manifest,
    )
    logger.info(
        "Primary cohort after merge: %d slides | %d patients | %d classes",
        len(df),
        df["patient_id"].nunique(),
        df["label"].nunique(),
    )

    if args.protocol == "grouped_cv":
        summary = run_grouped_cv(args, df)
    elif args.protocol == "external_test":
        if not args.test_mmap_dir or not args.test_manifest_csv:
            raise ValueError(
                "protocol=external_test requires both --test_mmap_dir and --test_manifest_csv."
            )
        logger.info("Loading external test manifest: %s", args.test_manifest_csv)
        test_manifest_df = load_manifest(
            csv_path=args.test_manifest_csv,
            slide_id_column=getattr(args, "test_slide_id_column", None) or args.slide_id_column,
            filename_column=getattr(args, "test_filename_column", None) or args.filename_column,
            label_column=getattr(args, "test_label_column", None) or args.label_column,
            patient_column=getattr(args, "test_patient_column", None) or args.patient_column,
            split_column=None,
        )
        logger.info("Loading external test embedding mmap: %s", args.test_mmap_dir)
        test_store = load_embedding_mmap(
            args.test_mmap_dir,
            non_singleton_policy=args.non_singleton_policy,
        )
        test_df = merge_manifest_and_embeddings(
            manifest_df=test_manifest_df,
            store=test_store,
            allow_missing_mmap=args.allow_missing_mmap,
            allow_missing_manifest=args.allow_missing_manifest,
        )
        logger.info(
            "External cohort after merge: %d slides | %d patients | %d classes",
            len(test_df),
            test_df["patient_id"].nunique(),
            test_df["label"].nunique(),
        )
        summary = run_external_test(args, df, test_df)
    elif args.protocol == "predefined_split":
        summary = run_predefined_split(args, df)
    else:
        raise ValueError(f"Unknown protocol: {args.protocol}")

    save_summary(output_dir, summary)
    logger.info("Summary saved → %s", output_dir / "summary.json")
    logger.info("Done.")


if __name__ == "__main__":
    main()
