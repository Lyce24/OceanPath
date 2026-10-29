"""
Thin entry point for split generation.

Usage:
    # Standard 5-fold CV
    python scripts/make_splits.py platform=local data=blca splits=kfold5

    # Holdout 80/10/10
    python scripts/make_splits.py platform=local data=blca splits=holdout_80_10_10

    # Custom: BLCA grade-2 → test, rest → k-fold
    python scripts/make_splits.py platform=local data=blca splits=blca_custom

    # Monte Carlo CV (10 repeats)
    python scripts/make_splits.py platform=local data=blca splits=mccv_10x

    # Nested CV (5-outer x 3-inner)
    python scripts/make_splits.py platform=local data=blca splits=nested_5x3

    # Dry run — load CSV, report stats, don't write
    python scripts/make_splits.py platform=local data=blca splits=kfold5 dry_run=true

    # Override seed
    python scripts/make_splits.py platform=local data=blca splits=kfold5 splits.seed=123

    # Force overwrite
    python scripts/make_splits.py platform=local data=blca splits=kfold5 force=true

    # Verify H5 features exist for every slide
    python scripts/make_splits.py platform=local data=blca splits=kfold5 verify_features=true
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _setup_logging(cfg: DictConfig) -> None:
    level = logging.DEBUG if cfg.get("verbose", False) else logging.INFO
    data_name = OmegaConf.select(cfg, "data.name", default="unknown")
    splits_name = OmegaConf.select(cfg, "splits.name", default="unknown")
    fmt = (
        "%(asctime)s | %(levelname)-7s | "
        f"stage=splits | data={data_name} | scheme={splits_name} | "
        "%(message)s"
    )
    logging.basicConfig(level=level, format=fmt, force=True)


def _build_split_config(cfg: DictConfig):
    """Build the typed SplitConfig from Hydra's composed config."""
    from oceanpath.data.splits import SplitConfig

    s = cfg.splits

    # Resolve output directory:
    # {platform.splits_root}/{data.name}/{splits.name}/
    splits_root = cfg.platform.splits_root
    data_name = cfg.data.name
    splits_name = s.name
    output_dir = f"{splits_root}/{data_name}/{splits_name}"

    # Resolve group column — handle case where it doesn't exist
    group_column = OmegaConf.select(cfg, "splits.group_column", default=None)
    site_column = OmegaConf.select(cfg, "splits.site_column", default=None)

    # Feature H5 dir for optional verification
    feature_h5_dir = None
    if cfg.get("verify_features", False):
        feature_h5_dir = OmegaConf.select(cfg, "data.feature_h5_dir", default=None)

    return SplitConfig(
        scheme=s.scheme,
        name=splits_name,
        csv_path=cfg.data.csv_path,
        output_dir=output_dir,
        filename_column=OmegaConf.select(cfg, "data.filename_column", default="filename"),
        label_column=s.label_column,
        group_column=group_column,
        site_column=site_column,
        seed=s.seed,
        n_folds=s.get("n_folds", 5) or 5,
        n_inner_folds=s.get("n_inner_folds", 3) or 3,
        n_repeats=s.get("n_repeats", 10) or 10,
        train_ratio=s.get("train_ratio", 0.70),
        val_ratio=s.get("val_ratio", 0.10),
        test_ratio=s.get("test_ratio", 0.20),
        test_filter=OmegaConf.select(cfg, "splits.test_filter", default=None),
        val_filter=OmegaConf.select(cfg, "splits.val_filter", default=None),
        feature_h5_dir=feature_h5_dir,
    )


@hydra.main(config_path="../configs", config_name="make_splits", version_base=None)
def main(cfg: DictConfig) -> None:
    _setup_logging(cfg)
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    split_cfg = _build_split_config(cfg)

    # ── Dry run ───────────────────────────────────────────────────────────
    if cfg.get("dry_run", False):
        from oceanpath.data.splits import _get_group_df, _load_manifest

        df = _load_manifest(split_cfg)
        groups_df = _get_group_df(df, split_cfg)

        print(f"\n{'=' * 60}")
        print("  DRY RUN — make_splits")
        print(f"{'=' * 60}")
        print(f"  CSV:             {split_cfg.csv_path}")
        print(f"  Output:          {split_cfg.output_dir}")
        print(f"  Scheme:          {split_cfg.scheme}")
        print(f"  Name:            {split_cfg.name}")
        print(f"  Seed:            {split_cfg.seed}")
        print(f"  Slides:          {len(df)}")
        print(f"  Groups:          {len(groups_df)}")
        print(f"  Label column:    {split_cfg.label_column}")
        print(f"  Label dist:      {dict(df[split_cfg.label_column].value_counts().sort_index())}")
        print(f"  Group column:    {split_cfg.group_column or '(none — slide-level)'}")
        print(f"  Site column:     {split_cfg.site_column or '(none)'}")

        if split_cfg.test_filter:
            try:
                test_mask = df.eval(split_cfg.test_filter)
                print(f"  Test filter:     {split_cfg.test_filter}")
                print(f"  Test matches:    {test_mask.sum()}/{len(df)} slides")
            except Exception as e:
                print(f"  Test filter:     {split_cfg.test_filter} (ERROR: {e})")

        if split_cfg.scheme in ("kfold", "custom_kfold"):
            print(f"  N folds:         {split_cfg.n_folds}")
        if split_cfg.scheme == "nested_cv":
            print(f"  N outer folds:   {split_cfg.n_folds}")
            print(f"  N inner folds:   {split_cfg.n_inner_folds}")
        if split_cfg.scheme == "monte_carlo":
            print(f"  N repeats:       {split_cfg.n_repeats}")
        if split_cfg.scheme in ("holdout", "monte_carlo"):
            print(
                f"  Ratios:          {split_cfg.train_ratio}/{split_cfg.val_ratio}/{split_cfg.test_ratio}"
            )

        print(f"{'=' * 60}\n")
        return

    # ── Generate splits ───────────────────────────────────────────────────
    from oceanpath.data.splits import generate_splits

    result = generate_splits(split_cfg, force=cfg.get("force", False))

    print(f"\n{'=' * 60}")
    print("  make_splits complete")
    print(f"{'=' * 60}")
    print(f"  Output:          {result.output_dir}")
    print(f"  Parquet:         {result.parquet_path}")
    print(f"  Scheme:          {result.scheme}")
    print(f"  Slides:          {result.n_slides}")
    print(f"  Groups:          {result.n_groups}")
    print(f"  Labels:          {result.label_distribution}")
    print(f"  Fold dist:       {result.fold_distribution}")
    print(f"  Integrity hash:  {result.integrity_hash[:16]}...")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
