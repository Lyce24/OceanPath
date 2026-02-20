"""
Stage 9: Cross-experiment comparison with statistical tests.

Loads OOF predictions from multiple training runs and produces:
  - Summary table (AUROC, balanced accuracy per experiment)
  - Pairwise DeLong tests (AUC difference significance)
  - Pairwise McNemar tests (accuracy difference significance)
  - Bootstrap paired CIs for any metric

Without this, model selection happens by eyeballing W&B dashboards.
With this, you get: "ABMIL outperforms TransMIL by 0.03 AUROC
(DeLong p=0.02, 95% CI [0.005, 0.055])."

Usage:
    # Compare two experiments
    python scripts/compare_experiments.py \\
        compare.experiments='{abmil: outputs/train/exp_abmil, transmil: outputs/train/exp_transmil}'

    # Compare all experiments in an output directory
    python scripts/compare_experiments.py \\
        compare.experiment_root=outputs/train

    # Adjust bootstrap iterations
    python scripts/compare_experiments.py ... compare.n_bootstrap=5000

Output:
    outputs/compare/{comparison_name}/
      ├── comparison_report.json
      ├── comparison_table.md
      └── pairwise_tests.json
"""

import json
import logging
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="compare", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Stage 9: Cross-experiment comparison."""

    level = logging.DEBUG if cfg.get("verbose", False) else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(message)s", force=True)

    c = cfg.compare
    start = time.monotonic()

    logger.info("=" * 60)
    logger.info("  Stage 9: Experiment Comparison")
    logger.info("=" * 60)

    # ── Resolve experiment directories ────────────────────────────────────
    experiment_dirs = {}

    # Mode 1: explicit experiment map
    if c.get("experiments"):
        experiment_dirs = dict(c.experiments)
        logger.info(f"Loaded {len(experiment_dirs)} experiments from config")

    # Mode 2: auto-discover from experiment_root
    elif c.get("experiment_root"):
        root = Path(c.experiment_root)
        if not root.is_dir():
            raise FileNotFoundError(f"Experiment root not found: {root}")

        for exp_dir in sorted(root.iterdir()):
            if not exp_dir.is_dir():
                continue
            # Check for OOF predictions or eval output
            has_preds = (exp_dir / "oof_predictions.parquet").is_file() or (
                exp_dir / "eval"
            ).is_dir()
            if has_preds:
                experiment_dirs[exp_dir.name] = str(exp_dir)

        logger.info(f"Discovered {len(experiment_dirs)} experiments in {root}")

    if len(experiment_dirs) < 2:
        raise ValueError(
            f"Need >= 2 experiments to compare, found {len(experiment_dirs)}. "
            f"Provide compare.experiments or compare.experiment_root."
        )

    for name, path in experiment_dirs.items():
        logger.info(f"  {name}: {path}")

    # ── Output directory ──────────────────────────────────────────────────
    comparison_name = c.get("name", "comparison")
    output_dir = Path(c.output_dir) / comparison_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Dry run ───────────────────────────────────────────────────────────
    if cfg.dry_run:
        print(f"\n{'=' * 60}")
        print("  DRY RUN — compare_experiments.py")
        print(f"{'=' * 60}")
        print(f"  Experiments: {len(experiment_dirs)}")
        for name, path in experiment_dirs.items():
            print(f"    {name}: {path}")
        print(f"  Bootstrap:   {c.n_bootstrap} iterations")
        print(f"  Output:      {output_dir}")
        print(f"{'=' * 60}\n")
        return

    # ── Run comparison ────────────────────────────────────────────────────
    from oceanpath.eval.comparison import compare_experiments

    logger.info("Running pairwise comparisons...")
    result = compare_experiments(
        experiment_dirs=experiment_dirs,
        n_bootstrap=c.get("n_bootstrap", 2000),
        alpha=c.get("alpha", 0.05),
        seed=c.get("seed", 42),
    )

    if "error" in result:
        logger.error(f"Comparison failed: {result['error']}")
        return

    # ── Save outputs ──────────────────────────────────────────────────────

    # Full report
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(result, indent=2, default=_json_default))
    logger.info(f"Report → {report_path}")

    # Markdown table
    table_path = output_dir / "comparison_table.md"
    table_path.write_text(result.get("summary_table", ""))
    logger.info(f"Table  → {table_path}")

    # Pairwise tests (separate file for easy parsing)
    pairwise_path = output_dir / "pairwise_tests.json"
    pairwise_path.write_text(
        json.dumps(
            result.get("pairwise", []),
            indent=2,
            default=_json_default,
        )
    )

    elapsed = time.monotonic() - start

    # ── Print summary ─────────────────────────────────────────────────────
    _print_summary(result, output_dir, elapsed)


def _print_summary(result: dict, output_dir: Path, elapsed: float) -> None:
    """Print comparison summary to console."""
    print(f"\n{'=' * 60}")
    print("  Experiment Comparison")
    print(f"{'=' * 60}")
    print(f"  Time: {elapsed:.0f}s")
    print()

    # Ranking table
    ranking = result.get("ranking", [])
    if ranking:
        print(f"  {'Rank':<5s} {'Experiment':<30s} {'AUROC':>8s} {'Bal.Acc':>8s}")
        print(f"  {'─' * 53}")
        for i, r in enumerate(ranking, 1):
            auroc = f"{r['auroc']:.4f}" if r.get("auroc") is not None else "—"
            bacc = (
                f"{r['balanced_accuracy']:.4f}" if r.get("balanced_accuracy") is not None else "—"
            )
            print(f"  {i:<5d} {r['name']:<30s} {auroc:>8s} {bacc:>8s}")

    # Pairwise results
    pairwise = result.get("pairwise", [])
    if pairwise:
        print("\n  Pairwise Tests:")
        print(f"  {'─' * 53}")
        for pw in pairwise:
            pair = pw.get("pair", "?")
            if "error" in pw:
                print(f"  {pair}: {pw['error']}")
                continue

            dl = pw.get("delong", {})
            mc = pw.get("mcnemar", {})

            dl_str = ""
            if "p_value" in dl:
                sig = "*" if dl.get("significant_at_05") else ""
                dl_str = f"DeLong p={dl['p_value']:.4f}{sig}"
            elif "skipped" in dl:
                dl_str = f"DeLong: {dl['skipped']}"

            mc_str = ""
            if "p_value" in mc:
                sig = "*" if mc.get("significant_at_05") else ""
                mc_str = f"McNemar p={mc['p_value']:.4f}{sig}"

            print(f"  {pair}")
            if dl_str:
                print(f"    {dl_str}")
            if mc_str:
                print(f"    {mc_str}")

            ba = pw.get("bootstrap_auroc", {})
            if "diff" in ba:
                ci_str = ""
                if ba.get("ci_lower") is not None:
                    ci_str = f" [{ba['ci_lower']:+.4f}, {ba['ci_upper']:+.4f}]"
                sig = "*" if ba.get("significant_at_05") else ""
                print(f"    AUROC diff: {ba['diff']:+.4f}{ci_str}{sig}")

    print(f"\n  Output: {output_dir}")
    print(f"{'=' * 60}\n")


def _json_default(o):
    """JSON fallback for numpy types."""
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


if __name__ == "__main__":
    main()
