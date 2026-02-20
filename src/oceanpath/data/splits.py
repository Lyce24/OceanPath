"""
Patient-safe, config-driven split generation for MIL training.

Supports:
  (A) Holdout — fixed train/val/test ratios
  (B) K-fold CV — stratified, optionally grouped by patient
  (C) Custom holdout/k-fold — domain-specific test set rules (e.g., "grade == 2 → test")
  (D) Monte Carlo CV — repeated random holdout
  (E) Nested CV — outer folds (test) x inner folds (train/val)

Key design decisions:
  - Patient grouping is OPTIONAL. If no patient_id column exists, each slide
    is treated as independent. When present, no patient appears in multiple splits.
  - Site/scanner balancing is OPTIONAL. When a site column is available, an
    additional pass checks and warns if site distributions are imbalanced.
  - Custom test rules use pandas query expressions — any valid query works.
  - Outputs are Parquet (typed columns, no float precision loss).
  - Integrity hash ties the split to its source manifest.

Output location:
  {platform.splits_root}/{data.name}/{splits.name}/
  ├── splits.parquet          # the actual split assignments
  ├── summary.json            # label/fold distribution stats
  └── .integrity_hash         # SHA256(manifest + splits) for downstream verification
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class SplitConfig:
    """All parameters needed to generate splits."""

    # Scheme
    scheme: str  # holdout | kfold | custom_holdout | custom_kfold | monte_carlo | nested_cv
    name: str  # used in output directory + filename

    # Source
    csv_path: str  # manifest CSV
    output_dir: str  # where to write splits.parquet

    # Column mapping
    filename_column: str = "filename"  # column with slide filename (stem → slide_id)
    label_column: str = "label"  # column for stratification
    group_column: str | None = None  # patient_id column (null → no grouping)
    site_column: str | None = None  # site/scanner column (null → no balancing)

    # Scheme parameters
    seed: int = 42
    n_folds: int = 5
    n_inner_folds: int = 3  # for nested_cv only
    n_repeats: int = 10  # for monte_carlo only
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Custom test/val rules (pandas query expressions)
    test_filter: str | None = None  # e.g., "`grade` == 2"
    val_filter: str | None = None  # rare, but supported

    # Optional: verify features exist
    feature_h5_dir: str | None = None


# ── Main entry point ──────────────────────────────────────────────────────────


@dataclass
class SplitResult:
    """Summary returned after split generation."""

    output_dir: str
    parquet_path: str
    scheme: str
    n_slides: int
    n_groups: int  # number of unique patients/groups
    label_distribution: dict  # label → count
    fold_distribution: dict  # fold/split → count
    integrity_hash: str


def generate_splits(cfg: SplitConfig, force: bool = False) -> SplitResult:
    """
    Generate and persist split assignments.

    Parameters
    ----------
    cfg : SplitConfig
        Complete split configuration.
    force : bool
        Overwrite existing splits.

    Returns
    -------
    SplitResult
        Summary of the generated splits.
    """
    output_dir = Path(cfg.output_dir)
    parquet_path = output_dir / "splits.parquet"

    # Check existing
    if parquet_path.is_file() and not force:
        logger.info(f"Splits already exist: {parquet_path}. Use force=true to overwrite.")
        return _load_existing_result(output_dir, cfg)

    # ── Load + validate manifest ──────────────────────────────────────────
    df = _load_manifest(cfg)
    logger.info(
        f"Manifest loaded: {len(df)} slides, "
        f"{df['group_id'].nunique()} groups, "
        f"labels: {dict(df[cfg.label_column].value_counts().sort_index())}"
    )

    # ── Optional: verify features exist ───────────────────────────────────
    if cfg.feature_h5_dir:
        _verify_features(df, cfg.feature_h5_dir)

    # ── Generate splits based on scheme ───────────────────────────────────
    scheme_fn = {
        "holdout": _generate_holdout,
        "kfold": _generate_kfold,
        "custom_holdout": _generate_custom_holdout,
        "custom_kfold": _generate_custom_kfold,
        "monte_carlo": _generate_monte_carlo,
        "nested_cv": _generate_nested_cv,
    }

    if cfg.scheme not in scheme_fn:
        raise ValueError(f"Unknown scheme '{cfg.scheme}'. Must be one of: {list(scheme_fn.keys())}")

    splits_df = scheme_fn[cfg.scheme](df, cfg)

    # ── Optional: check site balance ──────────────────────────────────────
    if cfg.site_column and cfg.site_column in splits_df.columns:
        _check_site_balance(splits_df, cfg)

    # ── Write outputs ─────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_df.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info(f"Splits written: {parquet_path} ({len(splits_df)} rows)")

    # Write integrity hash
    integrity_hash = _compute_integrity_hash(cfg.csv_path, str(parquet_path))
    (output_dir / ".integrity_hash").write_text(
        json.dumps({"hash": integrity_hash, "csv_path": cfg.csv_path}, indent=2)
    )

    # Write summary
    summary = _build_summary(splits_df, cfg)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"Summary written: {output_dir / 'summary.json'}")

    return SplitResult(
        output_dir=str(output_dir),
        parquet_path=str(parquet_path),
        scheme=cfg.scheme,
        n_slides=len(splits_df),
        n_groups=splits_df["group_id"].nunique()
        if "group_id" in splits_df.columns
        else len(splits_df),
        label_distribution=dict(splits_df[cfg.label_column].value_counts().sort_index()),
        fold_distribution=_get_fold_distribution(splits_df, cfg.scheme),
        integrity_hash=integrity_hash,
    )


# ── Manifest loading ──────────────────────────────────────────────────────────


def _load_manifest(cfg: SplitConfig) -> pd.DataFrame:
    """
    Load manifest CSV and derive slide_id + group_id.

    Always produces columns: slide_id, group_id, filename, {label_column}
    Plus any extra columns from the CSV (site, scanner, etc.)
    """
    csv_path = Path(cfg.csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # ── Validate required columns ─────────────────────────────────────────
    if cfg.filename_column not in df.columns:
        raise ValueError(
            f"Filename column '{cfg.filename_column}' not found in CSV. "
            f"Available: {list(df.columns)}"
        )
    if cfg.label_column not in df.columns:
        raise ValueError(
            f"Label column '{cfg.label_column}' not found in CSV. Available: {list(df.columns)}"
        )

    # ── Derive slide_id from filename ─────────────────────────────────────
    df["slide_id"] = df[cfg.filename_column].astype(str).apply(lambda x: Path(x).stem)

    # Check for duplicate slide_ids
    dupes = df["slide_id"].duplicated()
    if dupes.any():
        n_dupes = dupes.sum()
        logger.warning(
            f"{n_dupes} duplicate slide_ids found. Keeping first occurrence. "
            f"Duplicates: {df.loc[dupes, 'slide_id'].tolist()[:10]}"
        )
        df = df.drop_duplicates(subset="slide_id", keep="first").reset_index(drop=True)

    # ── Derive group_id (patient or slide-level) ─────────────────────────
    if cfg.group_column and cfg.group_column in df.columns:
        df["group_id"] = df[cfg.group_column].astype(str)
        logger.info(
            f"Grouping by '{cfg.group_column}': "
            f"{df['group_id'].nunique()} groups for {len(df)} slides"
        )
    else:
        # No patient column → each slide is its own group
        df["group_id"] = df["slide_id"]
        if cfg.group_column:
            logger.warning(
                f"Group column '{cfg.group_column}' not found in CSV. "
                f"Falling back to slide-level grouping (no patient deduplication)."
            )
        else:
            logger.info("No group column specified — using slide-level grouping.")

    # ── Drop rows with missing labels ─────────────────────────────────────
    n_before = len(df)
    df = df.dropna(subset=[cfg.label_column]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(f"Dropped {n_dropped} rows with missing labels.")

    # Ensure label is a clean type for stratification
    df[cfg.label_column] = df[cfg.label_column].astype(str)

    return df


# ── Scheme: Holdout ───────────────────────────────────────────────────────────


def _generate_holdout(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """Standard train/val/test holdout with stratification."""

    # Validate ratios
    total = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    # Work at group level to prevent leakage
    groups_df = _get_group_df(df, cfg)

    # First split: train+val vs test
    test_frac = cfg.test_ratio
    val_frac_of_trainval = cfg.val_ratio / (cfg.train_ratio + cfg.val_ratio)

    trainval_groups, test_groups = _stratified_group_split(
        groups_df, test_size=test_frac, label_col="label", seed=cfg.seed
    )

    # Second split: train vs val
    _, val_groups = _stratified_group_split(
        trainval_groups, test_size=val_frac_of_trainval, label_col="label", seed=cfg.seed + 1
    )

    # Map back to slides
    df = df.copy()
    df["split"] = "train"
    df.loc[df["group_id"].isin(val_groups["group_id"]), "split"] = "val"
    df.loc[df["group_id"].isin(test_groups["group_id"]), "split"] = "test"

    return _finalize_output(df, cfg)


# ── Scheme: K-Fold ────────────────────────────────────────────────────────────
def _generate_kfold(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """
    Stratified k-fold CV, optionally grouped by patient.

    Update: If cfg.test_ratio > 0, a holdout test set is removed FIRST,
    and CV is performed on the remaining data.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    df = df.copy()

    # 1. Handle Optional Test Holdout
    # -------------------------------
    if cfg.test_ratio and cfg.test_ratio > 0.0:
        logger.info(f"K-Fold: reserving {cfg.test_ratio:.1%} as held-out test set before folding.")

        # Collapse to groups for safe splitting
        groups_df = _get_group_df(df, cfg)

        # Split into (Train+Val) and (Test)
        _, test_groups_df = _stratified_group_split(
            groups_df, test_size=cfg.test_ratio, label_col="label", seed=cfg.seed
        )

        # identify test slides
        test_mask = df["group_id"].isin(test_groups_df["group_id"])

        # Mark test set in the dataframe with a special fold index (e.g., -1)
        df.loc[test_mask, "fold"] = -1

        # Reduce the dataset to only non-test slides for the actual folding
        cv_df = df[~test_mask].copy()
    else:
        # Use all data for CV
        cv_df = df.copy()

    # 2. Run K-Fold on the (remaining) data
    # -------------------------------------
    groups_df = _get_group_df(cv_df, cfg)
    labels = groups_df["label"].values
    has_groups = cfg.group_column and cfg.group_column in cv_df.columns

    if has_groups:
        # Group-aware: no patient leakage
        splitter = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
        fold_gen = splitter.split(
            X=groups_df.index,
            y=labels,
            groups=groups_df["group_id"].values,
        )
    else:
        splitter = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
        fold_gen = splitter.split(X=groups_df.index, y=labels)

    # Assign folds at group level
    group_fold_map = {}
    for fold_idx, (_, val_idx) in enumerate(fold_gen):
        # validation groups for this fold
        current_val_groups = groups_df.iloc[val_idx]["group_id"].values
        for gid in current_val_groups:
            group_fold_map[gid] = fold_idx

    # Map folds back to the original dataframe
    # Only map for rows that aren't already marked as test (-1)
    # We use 'map' on the cv_df subset and update the main df
    cv_folds = cv_df["group_id"].map(group_fold_map)

    # Update the main DataFrame safely
    # We use fillna(-1) to ensure test set remains -1 (or unassigned become -1)
    if "fold" not in df.columns:
        df["fold"] = np.nan

    # Assign the calculated folds to the non-test rows
    df.loc[cv_df.index, "fold"] = cv_folds

    # Sanity check
    unmapped = df["fold"].isna().sum()
    if unmapped > 0:
        logger.warning(
            f"{unmapped} slides not assigned to any fold (check group mapping). Setting to -1."
        )
        df["fold"] = df["fold"].fillna(-1)

    df["fold"] = df["fold"].astype(int)

    # Log distribution
    _log_split_counts(df, "fold")

    return _finalize_output(df, cfg)


# ── Scheme: Custom Holdout ────────────────────────────────────────────────────


def _generate_custom_holdout(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """
    Custom test set (via filter) + holdout train/val on remainder.

    Example: test_filter = "`grade` == 2" → all grade-2 slides are test.
    """
    df = df.copy()
    df["split"] = "train"  # default

    # Apply test filter
    if not cfg.test_filter:
        raise ValueError("custom_holdout requires test_filter to be set")

    test_mask = df.eval(cfg.test_filter)
    # Expand to all slides from the same group (patient safety)
    test_groups = set(df.loc[test_mask, "group_id"])
    test_mask_grouped = df["group_id"].isin(test_groups)
    df.loc[test_mask_grouped, "split"] = "test"

    n_test = test_mask_grouped.sum()
    logger.info(
        f"Custom test filter matched {test_mask.sum()} slides "
        f"({n_test} after group expansion, {len(test_groups)} groups)"
    )

    # Apply val filter if provided
    if cfg.val_filter:
        remaining = df["split"] == "train"
        val_mask = remaining & df.eval(cfg.val_filter)
        val_groups = set(df.loc[val_mask, "group_id"])
        val_mask_grouped = df["group_id"].isin(val_groups) & remaining
        df.loc[val_mask_grouped, "split"] = "val"
    else:
        # Split remaining into train/val by ratio
        remaining_df = df[df["split"] == "train"]
        if len(remaining_df) > 0 and cfg.val_ratio and cfg.val_ratio > 0:
            groups_df = _get_group_df(remaining_df, cfg)
            # val_ratio relative to remaining (not total)
            val_frac = cfg.val_ratio / (cfg.train_ratio + cfg.val_ratio) if cfg.train_ratio else 0.2

            _, val_groups_df = _stratified_group_split(
                groups_df, test_size=val_frac, label_col="label", seed=cfg.seed
            )
            df.loc[
                df["group_id"].isin(val_groups_df["group_id"]) & (df["split"] == "train"), "split"
            ] = "val"

    _log_split_counts(df, "split")
    return _finalize_output(df, cfg)


# ── Scheme: Custom K-Fold ─────────────────────────────────────────────────────


def _generate_custom_kfold(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """
    Custom Test Set + K-Fold on Remainder.

    Logic:
      1. Filter rows using `test_filter`. These become the Fixed Test Set (fold=-1).
      2. Select all remaining rows.
      3. Perform Stratified K-Fold on the remaining rows (folds 0..N).
    """
    df = df.copy()

    # 1. Validate and Apply Test Filter
    if not cfg.test_filter:
        raise ValueError("custom_kfold requires 'test_filter' to be set in config.")

    # Apply query to identify test candidates
    try:
        raw_test_mask = df.eval(cfg.test_filter)
    except Exception as e:
        raise ValueError(f"Failed to evaluate test_filter: '{cfg.test_filter}'. Error: {e}") from e

    # 2. Expand to Group Level (Patient Safety)
    # If one slide from a patient is test, ALL slides from that patient must be test
    test_groups = set(df.loc[raw_test_mask, "group_id"])
    final_test_mask = df["group_id"].isin(test_groups)

    n_test = final_test_mask.sum()
    n_total = len(df)
    logger.info(
        f"Custom K-Fold: Filter '{cfg.test_filter}' selected {n_test}/{n_total} slides "
        f"({len(test_groups)} groups) as Fixed Test."
    )

    # 3. Separate the Data
    # Test pool: assigned fold -1
    df.loc[final_test_mask, "fold"] = -1
    df.loc[final_test_mask, "split"] = "test"

    # CV pool: everything else
    cv_pool_mask = ~final_test_mask
    cv_df = df[cv_pool_mask].copy().reset_index(drop=True)

    if len(cv_df) < cfg.n_folds:
        raise ValueError(
            f"Not enough data remaining for {cfg.n_folds}-fold CV after removing test set. "
            f"Remaining: {len(cv_df)} slides."
        )

    # 4. Generate Folds on the CV Pool
    # We reuse _generate_kfold but force test_ratio=0 so it only makes folds 0..K
    sub_cfg = SplitConfig(
        scheme="kfold",
        name=f"{cfg.name}_sub",
        csv_path=cfg.csv_path,
        output_dir=cfg.output_dir,
        filename_column=cfg.filename_column,
        label_column=cfg.label_column,
        group_column=cfg.group_column,
        site_column=cfg.site_column,
        seed=cfg.seed,
        n_folds=cfg.n_folds,
        test_ratio=0.0,  # CRITICAL: No extra test holdout inside the CV pool
    )

    # Generate the folds on the subset
    cv_result = _generate_kfold(cv_df, sub_cfg)

    # 5. Map Folds Back to Main DataFrame
    # Create a map: slide_id -> fold
    slide_to_fold = dict(zip(cv_result["slide_id"], cv_result["fold"]))

    # Apply map only to the CV rows
    # Rows in final_test_mask are already -1.
    # Rows in cv_pool_mask get their new fold.
    df.loc[cv_pool_mask, "fold"] = df.loc[cv_pool_mask, "slide_id"].map(slide_to_fold).astype(int)
    df.loc[cv_pool_mask, "split"] = "train_val_pool"

    _log_split_counts(df, "fold")
    return _finalize_output(df, cfg)


# ── Scheme: Monte Carlo CV ────────────────────────────────────────────────────


def _generate_monte_carlo(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """Repeated random holdout splits."""
    all_repeats = []

    for repeat_idx in range(cfg.n_repeats):
        repeat_cfg = SplitConfig(
            scheme="holdout",
            name=cfg.name,
            csv_path=cfg.csv_path,
            output_dir=cfg.output_dir,
            filename_column=cfg.filename_column,
            label_column=cfg.label_column,
            group_column=cfg.group_column,
            site_column=cfg.site_column,
            seed=cfg.seed + repeat_idx,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
        )
        repeat_df = _generate_holdout(df.copy(), repeat_cfg)
        repeat_df["repeat"] = repeat_idx
        all_repeats.append(repeat_df)

    result = pd.concat(all_repeats, ignore_index=True)
    logger.info(f"Monte Carlo CV: {cfg.n_repeats} repeats x {len(df)} slides = {len(result)} rows")
    return result


# ── Scheme: Nested CV ─────────────────────────────────────────────────────────


def _generate_nested_cv(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """
    Nested cross-validation: outer folds define test, inner folds define train/val.

    Produces columns: outer_fold, inner_fold
      - outer_fold: which outer fold this slide is in (0..n_folds-1)
      - inner_fold: within outer-train, which inner fold (-1 for outer-test slides)
    """
    # First, generate outer folds
    outer_cfg = SplitConfig(
        scheme="kfold",
        name=cfg.name,
        csv_path=cfg.csv_path,
        output_dir=cfg.output_dir,
        filename_column=cfg.filename_column,
        label_column=cfg.label_column,
        group_column=cfg.group_column,
        site_column=cfg.site_column,
        seed=cfg.seed,
        n_folds=cfg.n_folds,
    )
    outer_df = _generate_kfold(df.copy(), outer_cfg)

    all_rows = []

    for outer_fold in range(cfg.n_folds):
        # Outer test: this fold
        outer_test_mask = outer_df["fold"] == outer_fold
        outer_train_df = outer_df[~outer_test_mask].copy().reset_index(drop=True)

        # Inner folds on outer-train
        inner_cfg = SplitConfig(
            scheme="kfold",
            name=cfg.name,
            csv_path=cfg.csv_path,
            output_dir=cfg.output_dir,
            filename_column=cfg.filename_column,
            label_column=cfg.label_column,
            group_column=cfg.group_column,
            site_column=cfg.site_column,
            seed=cfg.seed + outer_fold + 1000,
            n_folds=cfg.n_inner_folds,
        )
        inner_df = _generate_kfold(outer_train_df, inner_cfg)

        # Build combined rows
        # Inner train/val slides
        for _, row in inner_df.iterrows():
            row_out = row.copy()
            row_out["outer_fold"] = outer_fold
            row_out["inner_fold"] = int(row["fold"])
            all_rows.append(row_out)

        # Outer test slides (inner_fold = -1)
        for _, row in outer_df[outer_test_mask].iterrows():
            row_out = row.copy()
            row_out["outer_fold"] = outer_fold
            row_out["inner_fold"] = -1
            all_rows.append(row_out)

    result = pd.DataFrame(all_rows).reset_index(drop=True)

    # Clean up the temporary "fold" column from inner kfold
    if "fold" in result.columns:
        result = result.drop(columns=["fold"])

    result["outer_fold"] = result["outer_fold"].astype(int)
    result["inner_fold"] = result["inner_fold"].astype(int)

    logger.info(
        f"Nested CV: {cfg.n_folds} outer x {cfg.n_inner_folds} inner folds, "
        f"{len(result)} total rows"
    )
    return result


# ── Helpers: group-level operations ───────────────────────────────────────────


def _get_group_df(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """
    Collapse to one row per group (patient) for group-level splitting.

    Uses majority label per group for stratification.
    """
    # Majority label per group
    group_label = (
        df.groupby("group_id")[cfg.label_column]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={cfg.label_column: "label"})
    )
    return group_label


def _stratified_group_split(
    groups_df: pd.DataFrame,
    test_size: float,
    label_col: str = "label",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split at the group level.

    Returns (group_a, group_b) where group_b has ~test_size fraction.
    """
    from sklearn.model_selection import train_test_split

    a, b = train_test_split(
        groups_df,
        test_size=test_size,
        stratify=groups_df[label_col],
        random_state=seed,
    )
    return a, b


# ── Helpers: site balance checking ────────────────────────────────────────────


def _check_site_balance(df: pd.DataFrame, cfg: SplitConfig) -> None:
    """
    Warn if site/scanner distribution is significantly imbalanced across splits.

    Does NOT enforce balance (that would require more complex splitting),
    just logs a warning so the user knows.
    """
    site_col = cfg.site_column
    if site_col not in df.columns:
        return

    # Determine which column defines the split
    if "fold" in df.columns:
        split_col = "fold"
    elif "split" in df.columns:
        split_col = "split"
    else:
        return

    # Crosstab: site x split
    ct = pd.crosstab(df[site_col], df[split_col], normalize="columns")

    # Check for any site that deviates >20% from its overall proportion
    overall = df[site_col].value_counts(normalize=True)
    max_deviation = 0.0
    worst_site = None
    worst_split = None

    for site in ct.index:
        for split_val in ct.columns:
            deviation = abs(ct.loc[site, split_val] - overall[site])
            if deviation > max_deviation:
                max_deviation = deviation
                worst_site = site
                worst_split = split_val

    if max_deviation > 0.15:
        logger.warning(
            f"Site imbalance detected: site='{worst_site}' deviates by "
            f"{max_deviation:.1%} in {split_col}='{worst_split}'. "
            f"Consider site-aware splitting or verifying your data."
        )
    else:
        logger.info(f"Site balance OK (max deviation: {max_deviation:.1%})")


# ── Helpers: output formatting ────────────────────────────────────────────────


def _finalize_output(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """
    Clean up and standardize the output DataFrame.

    Ensures consistent column ordering and types.
    """
    # Always include these columns
    keep_cols = ["slide_id", "group_id", cfg.filename_column, cfg.label_column]

    # Add scheme-specific columns
    scheme_cols = {
        "holdout": ["split"],
        "kfold": ["fold"],
        "custom_holdout": ["split"],
        "custom_kfold": ["fold", "split"],
        "monte_carlo": ["repeat", "split"],
        "nested_cv": ["outer_fold", "inner_fold"],
    }
    for col in scheme_cols.get(cfg.scheme, []):
        if col in df.columns:
            keep_cols.append(col)

    # Add optional metadata columns if present
    if cfg.site_column and cfg.site_column in df.columns:
        keep_cols.append(cfg.site_column)

    # Rename group_id → patient_id if it was derived from a patient column
    if cfg.group_column and cfg.group_column in df.columns and cfg.group_column != "group_id":
        keep_cols = [c for c in keep_cols if c != "group_id"]
        if cfg.group_column not in keep_cols:
            keep_cols.insert(1, cfg.group_column)
        # Also keep group_id for internal use
        keep_cols.append("group_id")

    # Filter to existing columns only
    keep_cols = [c for c in keep_cols if c in df.columns]
    # Remove duplicates while preserving order
    seen = set()
    keep_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]

    return df[keep_cols].reset_index(drop=True)


def _log_split_counts(df: pd.DataFrame, col: str) -> None:
    """Log value counts for a split/fold column."""
    counts = df[col].value_counts().sort_index()
    logger.info(f"Split distribution ({col}):\n{counts.to_string()}")


def _get_fold_distribution(df: pd.DataFrame, scheme: str) -> dict:
    """Get fold/split distribution for the summary."""
    if "fold" in df.columns:
        return dict(df["fold"].value_counts().sort_index())
    if "split" in df.columns:
        return dict(df["split"].value_counts())
    if "outer_fold" in df.columns:
        return dict(df["outer_fold"].value_counts().sort_index())
    return {}


# ── Integrity hashing ────────────────────────────────────────────────────────


def _compute_integrity_hash(csv_path: str, parquet_path: str) -> str:
    """
    SHA256 of (manifest CSV content + splits Parquet content).

    If either file changes, the hash changes, catching stale splits.
    """
    h = hashlib.sha256()

    for path in [csv_path, parquet_path]:
        with open(path, "rb") as f:
            while chunk := f.read(8 * 1024 * 1024):
                h.update(chunk)

    return h.hexdigest()


def verify_split_integrity(splits_dir: str | Path, csv_path: str | Path) -> bool:
    """
    Verify that splits match their source manifest.

    Call this at training time to catch stale splits.

    Parameters
    ----------
    splits_dir : Path
        Directory containing splits.parquet and .integrity_hash
    csv_path : Path
        The manifest CSV that should match.

    Returns
    -------
    bool
        True if integrity check passes.

    Raises
    ------
    SplitIntegrityError
        If the hash doesn't match.
    """
    splits_dir = Path(splits_dir)
    hash_path = splits_dir / ".integrity_hash"
    parquet_path = splits_dir / "splits.parquet"

    if not hash_path.is_file():
        logger.warning(f"No integrity hash found at {hash_path} — skipping check")
        return True

    if not parquet_path.is_file():
        raise FileNotFoundError(f"Splits file not found: {parquet_path}")

    stored = json.loads(hash_path.read_text())
    expected_hash = stored["hash"]
    actual_hash = _compute_integrity_hash(str(csv_path), str(parquet_path))

    if expected_hash != actual_hash:
        raise SplitIntegrityError(
            f"Split integrity check failed.\n"
            f"  Stored hash:  {expected_hash[:16]}...\n"
            f"  Current hash: {actual_hash[:16]}...\n"
            f"  This means the manifest CSV or splits file has changed.\n"
            f"  Regenerate splits: python scripts/make_splits.py force=true"
        )

    logger.info(f"Split integrity verified: {expected_hash[:16]}...")
    return True


class SplitIntegrityError(Exception):
    """Raised when split integrity verification fails."""


# ── Feature verification ──────────────────────────────────────────────────────


def _verify_features(df: pd.DataFrame, h5_dir: str) -> None:
    """Check that every slide_id in the manifest has a corresponding H5 file."""
    h5_path = Path(h5_dir)
    if not h5_path.is_dir():
        logger.warning(f"Feature H5 dir does not exist: {h5_path} — skipping verification")
        return

    existing = {f.stem for f in h5_path.glob("*.h5")}
    expected = set(df["slide_id"])
    missing = expected - existing

    if missing:
        logger.warning(
            f"{len(missing)}/{len(expected)} slides have no H5 features. "
            f"First 10: {sorted(missing)[:10]}"
        )
    else:
        logger.info(f"Feature verification passed: all {len(expected)} slides have H5 files")


# ── Loading splits (for downstream stages) ────────────────────────────────────


def load_splits(
    splits_dir: str | Path,
    csv_path: str | Path | None = None,
    verify: bool = True,
) -> pd.DataFrame:
    """
    Load split assignments from a splits directory.

    Convenience function for downstream stages (Dataset, DataModule, evaluate).

    Parameters
    ----------
    splits_dir : Path
        Directory containing splits.parquet.
    csv_path : Path, optional
        If provided and verify=True, checks integrity against this manifest.
    verify : bool
        Whether to run integrity check.

    Returns
    -------
    pd.DataFrame
        The splits DataFrame.
    """
    splits_dir = Path(splits_dir)
    parquet_path = splits_dir / "splits.parquet"

    if not parquet_path.is_file():
        raise FileNotFoundError(f"Splits not found: {parquet_path}")

    if verify and csv_path:
        verify_split_integrity(splits_dir, csv_path)

    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded splits: {len(df)} rows from {parquet_path}")
    return df


def get_slide_ids_for_fold(
    splits_df: pd.DataFrame,
    fold: int,
    scheme: str = "kfold",
    outer_fold: int | None = None,
) -> dict[str, list[str]]:
    """
    Get train/val/test slide_id lists for a specific fold iteration.
    """
    # Initialize lists
    train_ids = []
    val_ids = []
    test_ids = []

    if scheme in ["kfold", "custom_kfold"]:
        # Logic:
        # 1. TEST: All rows marked as -1 are ALWAYS test (Fixed Held-out).
        #    (In standard kfold without holdout, this list might be empty, which is fine).
        test_ids = splits_df.loc[splits_df["fold"] == -1, "slide_id"].tolist()

        # 2. VAL: The requested fold index (e.g., 0) is validation.
        val_ids = splits_df.loc[splits_df["fold"] == fold, "slide_id"].tolist()

        # 3. TRAIN: Everything else, EXCLUDING the test set (-1) and current val fold.
        train_mask = (splits_df["fold"] != fold) & (splits_df["fold"] != -1)
        train_ids = splits_df.loc[train_mask, "slide_id"].tolist()

    elif scheme == "holdout" or scheme == "custom_holdout":
        train_ids = splits_df.loc[splits_df["split"] == "train", "slide_id"].tolist()
        val_ids = splits_df.loc[splits_df["split"] == "val", "slide_id"].tolist()
        test_ids = splits_df.loc[splits_df["split"] == "test", "slide_id"].tolist()

    elif scheme == "nested_cv":
        if outer_fold is None:
            raise ValueError("nested_cv requires `outer_fold` to be specified")

        subset = splits_df[splits_df["outer_fold"] == outer_fold]

        # Outer Test
        test_ids = subset.loc[subset["inner_fold"] == -1, "slide_id"].tolist()

        # Inner Val
        val_ids = subset.loc[subset["inner_fold"] == fold, "slide_id"].tolist()

        # Inner Train
        train_mask = (subset["inner_fold"] != fold) & (subset["inner_fold"] != -1)
        train_ids = subset.loc[train_mask, "slide_id"].tolist()

    return {"train": train_ids, "val": val_ids, "test": test_ids}


# ── Existing result loading ───────────────────────────────────────────────────


def _load_existing_result(output_dir: Path, cfg: SplitConfig) -> SplitResult:
    """Load a SplitResult from an existing splits directory."""
    parquet_path = output_dir / "splits.parquet"
    df = pd.read_parquet(parquet_path)

    hash_path = output_dir / ".integrity_hash"
    integrity_hash = ""
    if hash_path.is_file():
        integrity_hash = json.loads(hash_path.read_text()).get("hash", "")

    return SplitResult(
        output_dir=str(output_dir),
        parquet_path=str(parquet_path),
        scheme=cfg.scheme,
        n_slides=len(df),
        n_groups=df["group_id"].nunique() if "group_id" in df.columns else len(df),
        label_distribution=dict(df[cfg.label_column].value_counts().sort_index())
        if cfg.label_column in df.columns
        else {},
        fold_distribution=_get_fold_distribution(df, cfg.scheme),
        integrity_hash=integrity_hash,
    )


def _build_summary(df: pd.DataFrame, cfg: SplitConfig) -> dict:
    """Build a JSON-serializable summary of the splits."""
    summary = {
        "scheme": cfg.scheme,
        "name": cfg.name,
        "seed": cfg.seed,
        "n_slides": len(df),
        "n_groups": df["group_id"].nunique() if "group_id" in df.columns else len(df),
        "label_column": cfg.label_column,
        "label_distribution": dict(df[cfg.label_column].value_counts().sort_index()),
        "group_column": cfg.group_column,
    }

    # Scheme-specific stats
    if "fold" in df.columns:
        summary["n_folds"] = cfg.n_folds
        for fold_val in sorted(df["fold"].unique()):
            fold_df = df[df["fold"] == fold_val]
            summary[f"fold_{fold_val}_count"] = len(fold_df)
            summary[f"fold_{fold_val}_labels"] = dict(
                fold_df[cfg.label_column].value_counts().sort_index()
            )

    if "split" in df.columns:
        for split_val in ["train", "val", "test"]:
            split_df = df[df["split"] == split_val]
            if len(split_df) > 0:
                summary[f"{split_val}_count"] = len(split_df)
                summary[f"{split_val}_labels"] = dict(
                    split_df[cfg.label_column].value_counts().sort_index()
                )

    if cfg.test_filter:
        summary["test_filter"] = cfg.test_filter
    if cfg.val_filter:
        summary["val_filter"] = cfg.val_filter

    return summary
