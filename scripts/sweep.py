"""
Unified hyperparameter sweep orchestrator for OceanPath.

Supports both supervised (train.py) and SSL pretrain (pretrain.py) pipelines
via Hydra config composition. Uses Optuna for Bayesian optimization (TPE)
with subprocess-based trial isolation.

Each trial runs as an independent subprocess, ensuring:
  - No GPU memory leaks between trials
  - Full Hydra config resolution per trial
  - Crash isolation (a failed trial doesn't kill the sweep)

Supports single-objective and multi-objective (Pareto front) optimization.
For multi-objective, mark one metric as ``primary: true`` to sort the
Pareto front and generate re-run commands.

Usage:
    # Supervised sweep (default)
    python scripts/sweep.py sweep_space=supervised

    # SSL pretrain sweep
    python scripts/sweep.py sweep_space=pretrain

    # More trials, parallel on 2 GPUs
    python scripts/sweep.py sweep_space=supervised n_trials=60 n_jobs=2 gpu_ids=[0,1]

    # Resume an interrupted sweep
    python scripts/sweep.py study_name=supervised_sweep_20260315_120000

    # Disable screening for full k-fold validation
    python scripts/sweep.py sweep_space=supervised sweep_space.screening.enabled=false

Requires: pip install optuna
"""

import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from oceanpath.data.mmap_builder import validate_mmap_dir

try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    optuna = None
    TrialState = None

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Per-GPU semaphores, initialized in main() before study.optimize().
# Keys are GPU IDs (int), values are threading.Semaphore instances.
_gpu_semaphores: dict[int, threading.Semaphore] = {}


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _flatten_dict(d: dict, prefix: str = "") -> list[tuple[str, object]]:
    """Flatten a nested dict into (dotted_key, value) pairs.

    Example:
        {"training": {"max_epochs": 30, "lr": 1e-4}} →
        [("training.max_epochs", 30), ("training.lr", 1e-4)]
    """
    items = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, key))
        else:
            items.append((key, v))
    return items


def _format_value(v: object) -> str:
    """Format a Python value as a Hydra CLI override value."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _log_mmap_info(cfg: DictConfig) -> None:
    """Log mmap store metadata at sweep startup for sanity checking."""
    space_cfg = cfg.sweep_space

    # Extract mmap_dir from extra_overrides
    mmap_dir = None
    extra = OmegaConf.to_container(space_cfg.get("extra_overrides", []), resolve=True)
    for ov in extra:
        if ov.startswith("data.mmap_dir="):
            mmap_dir = ov.split("=", 1)[1]
            break

    if mmap_dir is None:
        logger.warning("No data.mmap_dir found in extra_overrides — using dataset default")
        return

    mmap_path = Path(mmap_dir)
    if not mmap_path.is_absolute():
        mmap_path = PROJECT_ROOT / mmap_path

    try:
        meta = validate_mmap_dir(str(mmap_path))
        logger.info(
            f"Mmap store: {mmap_dir}\n"
            f"  Slides:       {meta['n_slides']}\n"
            f"  Total patches: {meta['total_patches']:,}\n"
            f"  Feature dim:  {meta['feat_dim']}\n"
            f"  Feature dtype: {meta['feat_dtype']}\n"
            f"  Feat chunks:  {meta['num_feat_chunks']}\n"
            f"  Coord chunks: {meta['num_coord_chunks']}"
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Mmap validation FAILED for {mmap_dir}: {e}")
        sys.exit(1)


def _get_metrics_cfg(space_cfg) -> list[dict]:
    """Normalize objective config to always return a list of metric dicts.

    Supports both the old single-metric format::

        objective_metric: test / loss
        direction: minimize

    and the new multi-objective format::

        objective_metrics:
          - name: test/auc
            direction: maximize
            primary: true
          - name: test/f1
            direction: maximize
          - name: test/loss
            direction: minimize

    Returns a list of dicts, each with keys: name, direction, primary (bool).
    Exactly one metric will have ``primary: True``.
    """
    if "objective_metrics" in space_cfg:
        metrics = list(OmegaConf.to_container(space_cfg.objective_metrics, resolve=True))
        # Ensure primary flag exists on every entry
        has_primary = any(m.get("primary", False) for m in metrics)
        for m in metrics:
            m.setdefault("primary", False)
        # Default: first metric is primary if none is marked
        if not has_primary:
            metrics[0]["primary"] = True
        return metrics

    # Backward compat: single metric
    return [
        {
            "name": space_cfg.objective_metric,
            "direction": space_cfg.direction,
            "primary": True,
        }
    ]


def _primary_index(metrics_cfg: list[dict]) -> int:
    """Return the index of the primary metric."""
    for i, m in enumerate(metrics_cfg):
        if m.get("primary", False):
            return i
    return 0


# ═════════════════════════════════════════════════════════════════════════════
# Search space sampling
# ═════════════════════════════════════════════════════════════════════════════


def sample_overrides(trial: "optuna.Trial", search_space: list[dict]) -> list[str]:
    """Sample hyperparameters from search space and return Hydra CLI overrides.

    Parameters are processed in definition order. Conditional parameters
    (those with ``condition_param`` / ``condition_value``) are only sampled
    when the referenced parameter's sampled value matches.

    Returns a list of strings like ``["model=abmil", "training.lr=0.0003"]``.
    """
    overrides = []
    sampled: dict[str, object] = {}

    for spec in search_space:
        name = spec["name"]
        ptype = spec["type"]

        # ── Conditional gating ─────────────────────────────────────────────
        cond_param = spec.get("condition_param")
        if cond_param is not None:
            cond_value = spec["condition_value"]
            actual = sampled.get(cond_param)
            if actual is None or str(actual) != str(cond_value):
                continue

        # ── Sample ─────────────────────────────────────────────────────────
        if ptype == "categorical":
            value = trial.suggest_categorical(name, spec["choices"])

        elif ptype == "float":
            log = spec.get("log", False)
            step = spec.get("step")
            kwargs: dict = {"log": log}
            if step is not None and not log:
                kwargs["step"] = float(step)
            value = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), **kwargs)

        elif ptype == "int":
            log = spec.get("log", False)
            step = spec.get("step")
            kwargs = {"log": log}
            if step is not None and not log:
                kwargs["step"] = int(step)
            value = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), **kwargs)

        else:
            raise ValueError(f"Unknown param type '{ptype}' for '{name}'")

        sampled[name] = value
        overrides.append(f"{name}={_format_value(value)}")

    return overrides


# ═════════════════════════════════════════════════════════════════════════════
# Output parsing
# ═════════════════════════════════════════════════════════════════════════════


def resolve_output_dir(cfg: DictConfig, trial_exp_name: str) -> Path:
    """Predict where the training script writes its outputs."""
    pipeline = cfg.sweep_space.pipeline
    subdir = "pretrain" if pipeline == "pretrain" else "train"
    return Path(cfg.output_root) / subdir / trial_exp_name


def parse_objective(output_dir: Path, metric: str) -> float:
    """Extract a single objective metric from training output files.

    Search order:
      1. cv_summary.json    — supervised CV aggregated results
      2. fold_0/fold_metrics.json — supervised single-fold fallback
      3. metadata.json      — pretrain results

    For fold_metrics.json, also tries stripping ``_mean`` / ``_std`` suffixes
    so that ``val/auroc_mean`` can fall back to ``val/auroc`` in a single fold.
    """
    # 1. Supervised CV summary (always written by train.py, even for 1 fold)
    cv_path = output_dir / "cv_summary.json"
    if cv_path.is_file():
        data = json.loads(cv_path.read_text())
        if metric in data and data[metric] is not None:
            return float(data[metric])
        # Auto-try _mean suffix: "test/auroc" → "test/auroc_mean"
        mean_metric = f"{metric}_mean"
        if mean_metric in data and data[mean_metric] is not None:
            return float(data[mean_metric])

    # 2. Single-fold fallback
    fold_path = output_dir / "fold_0" / "fold_metrics.json"
    if fold_path.is_file():
        data = json.loads(fold_path.read_text())
        if metric in data and data[metric] is not None:
            return float(data[metric])
        # Try without _mean/_std suffix (e.g. val/auroc_mean → val/auroc)
        base_metric = metric.replace("_mean", "").replace("_std", "")
        if base_metric != metric and base_metric in data and data[base_metric] is not None:
            return float(data[base_metric])

    # 3. Pretrain metadata
    meta_path = output_dir / "metadata.json"
    if meta_path.is_file():
        data = json.loads(meta_path.read_text())
        if metric in data and data[metric] is not None:
            return float(data[metric])

    raise FileNotFoundError(
        f"Metric '{metric}' not found in {output_dir}.\n"
        f"Searched: cv_summary.json, fold_0/fold_metrics.json, metadata.json"
    )


def parse_all_objectives(output_dir: Path, metrics_cfg: list[dict]) -> list[float]:
    """Parse all objective metrics for a trial.

    Returns a list of float values in the same order as *metrics_cfg*.
    Raises on any missing or NaN metric (caller handles pruning).
    """
    values: list[float] = []
    for m in metrics_cfg:
        name = m["name"]
        v = parse_objective(output_dir, name)  # raises FileNotFoundError
        if math.isnan(v):
            raise ValueError(f"{name} is NaN")
        values.append(v)
    return values


# ═════════════════════════════════════════════════════════════════════════════
# Trial execution
# ═════════════════════════════════════════════════════════════════════════════


def run_trial(trial: "optuna.Trial", cfg: DictConfig):
    """Execute a single Optuna trial as a subprocess.

    Returns:
        float  — if single-objective
        tuple[float, ...] — if multi-objective
    """
    space_cfg = cfg.sweep_space
    search_space = OmegaConf.to_container(space_cfg.search_space, resolve=True)
    metrics_cfg = _get_metrics_cfg(space_cfg)
    is_multi = len(metrics_cfg) > 1

    # ── 1. Sample hyperparameters ──────────────────────────────────────────
    overrides = sample_overrides(trial, search_space)

    # ── 2. Fixed overrides ─────────────────────────────────────────────────
    fixed = OmegaConf.to_container(space_cfg.get("fixed_overrides", {}), resolve=True)
    for k, v in _flatten_dict(fixed):
        overrides.append(f"{k}={_format_value(v)}")

    # ── 2b. Extra raw overrides (passed as-is) ─────────────────────────────
    extra = OmegaConf.to_container(space_cfg.get("extra_overrides", []), resolve=True)
    overrides.extend(extra)

    # ── 3. Screening overrides ─────────────────────────────────────────────
    screening = OmegaConf.to_container(space_cfg.get("screening", {}), resolve=True)
    if screening.get("enabled", False):
        for k, v in _flatten_dict(screening.get("overrides", {})):
            overrides.append(f"{k}={_format_value(v)}")

    # ── 4. Trial experiment name ───────────────────────────────────────────
    trial_exp_name = f"sweep/{cfg.study_name}/trial_{trial.number:04d}"
    overrides.append(f"exp_name={trial_exp_name}")

    # ── 5. W&B control ────────────────────────────────────────────────────
    if cfg.get("disable_wandb", True):
        overrides.append("wandb.offline=true")

    # ── 6. Build command ───────────────────────────────────────────────────
    target_script = str(PROJECT_ROOT / "scripts" / space_cfg.target_script)
    cmd = [sys.executable, target_script, *overrides]

    # Log sampled params only (not fixed/screening boilerplate)
    param_names = {s["name"] for s in search_space}
    param_str = " ".join(o for o in overrides if "=" in o and o.split("=", 1)[0] in param_names)
    logger.info(f"Trial {trial.number:04d} | {param_str}")

    # ── 7. GPU assignment (round-robin) ────────────────────────────────────
    gpu_ids = list(cfg.get("gpu_ids", [0]))
    gpu_id = gpu_ids[trial.number % len(gpu_ids)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    # ── 8. Run subprocess (GPU semaphore limits concurrency per device) ──
    sem = _gpu_semaphores.get(gpu_id)
    t_start = time.time()
    if sem is not None:
        sem.acquire()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=cfg.get("trial_timeout", 7200),
        )
    except subprocess.TimeoutExpired as err:
        logger.error(f"Trial {trial.number:04d} timed out after {cfg.trial_timeout}s")
        raise optuna.TrialPruned("Timed out") from err
    finally:
        if sem is not None:
            sem.release()

    elapsed = time.time() - t_start

    # ── 9. Parse objectives ────────────────────────────────────────────────
    # Negative return codes (e.g. -6/SIGABRT, -11/SIGSEGV) often come from
    # PyTorch/CUDA cleanup during Python shutdown — the training itself may
    # have completed and written valid output files.  Try to parse outputs
    # before giving up.
    output_dir = resolve_output_dir(cfg, trial_exp_name)

    if result.returncode != 0:
        is_signal = result.returncode < 0
        if is_signal:
            logger.warning(
                f"Trial {trial.number:04d} exited with signal "
                f"{-result.returncode} (rc={result.returncode}); "
                f"checking for output files..."
            )
        else:
            stderr_tail = (result.stderr or "")[-3000:]
            logger.error(
                f"Trial {trial.number:04d} failed (rc={result.returncode}):\n{stderr_tail}"
            )

    try:
        values = parse_all_objectives(output_dir, metrics_cfg)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Trial {trial.number:04d} | {e}")
        if result.returncode != 0:
            raise optuna.TrialPruned(
                f"Failed (rc={result.returncode}) and no valid output: {e}"
            ) from e
        raise optuna.TrialPruned(str(e)) from e

    # Store all metric values as trial user attrs for easy inspection
    for m, v in zip(metrics_cfg, values):
        trial.set_user_attr(m["name"], v)

    parts = [f"{m['name']}={v:.6f}" for m, v in zip(metrics_cfg, values)]
    logger.info(f"Trial {trial.number:04d} | {' | '.join(parts)} | {elapsed / 60:.1f}min")

    if is_multi:
        return tuple(values)
    return values[0]


# ═════════════════════════════════════════════════════════════════════════════
# Reporting
# ═════════════════════════════════════════════════════════════════════════════


def _get_fixed_overrides_strs(cfg: DictConfig) -> list[str]:
    """Build Hydra CLI strings for fixed overrides."""
    fixed = OmegaConf.to_container(cfg.sweep_space.get("fixed_overrides", {}), resolve=True)
    strs = [f"{k}={_format_value(v)}" for k, v in _flatten_dict(fixed)]
    extra = OmegaConf.to_container(cfg.sweep_space.get("extra_overrides", []), resolve=True)
    strs.extend(extra)
    return strs


def _print_rerun_commands(
    trials,
    cfg: DictConfig,
    fixed_strs: list[str],
    metrics_cfg: list[dict],
    primary_idx: int,
    is_multi: bool,
) -> None:
    """Print re-run commands for a list of trials."""
    top_k = min(5, len(trials))
    print(f"\n  Re-run top {top_k} with full validation:")
    for t in trials[:top_k]:
        param_strs = [f"{k}={_format_value(v)}" for k, v in t.params.items()]
        all_overrides = fixed_strs + param_strs
        cmd = f"python scripts/{cfg.sweep_space.target_script} {' '.join(all_overrides)}"
        if is_multi:
            label_parts = [f"{m['name']}={t.values[i]:.6f}" for i, m in enumerate(metrics_cfg)]
            label = ", ".join(label_parts)
        else:
            label = f"{metrics_cfg[0]['name']}={t.value:.6f}"
        print(f"    # Trial #{t.number:04d} ({label})")
        print(f"    {cmd}\n")


def print_study_summary(study: "optuna.Study", cfg: DictConfig) -> None:
    """Print sweep summary with top trials and re-run commands.

    For multi-objective studies, displays the Pareto front sorted by the
    primary metric.  For single-objective, shows the top-K ranking.
    """
    metrics_cfg = _get_metrics_cfg(cfg.sweep_space)
    is_multi = len(metrics_cfg) > 1
    primary_idx = _primary_index(metrics_cfg)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]

    print(f"\n{'=' * 70}")
    print(f"  Sweep Complete: {cfg.study_name}")
    print(f"{'=' * 70}")
    print(f"  Completed: {len(completed)} / {len(study.trials)}")
    if pruned:
        print(f"  Pruned/Failed: {len(pruned)}")

    if not completed:
        print("  No completed trials.")
        print(f"{'=' * 70}")
        return

    fixed_strs = _get_fixed_overrides_strs(cfg)

    if is_multi:
        _print_multi_objective_summary(study, cfg, metrics_cfg, primary_idx, fixed_strs)
    else:
        _print_single_objective_summary(study, cfg, metrics_cfg, fixed_strs)

    print(f"  Storage: {cfg.storage}")
    print(f"{'=' * 70}")


def _print_multi_objective_summary(
    study: "optuna.Study",
    cfg: DictConfig,
    metrics_cfg: list[dict],
    primary_idx: int,
    fixed_strs: list[str],
) -> None:
    """Print Pareto front summary for multi-objective studies."""
    pareto = study.best_trials
    is_max = metrics_cfg[primary_idx]["direction"] == "maximize"
    pareto_sorted = sorted(pareto, key=lambda t: t.values[primary_idx], reverse=is_max)

    primary_name = metrics_cfg[primary_idx]["name"]
    print(f"\n  Pareto front: {len(pareto_sorted)} trials (sorted by {primary_name})")

    # ── Table header ───────────────────────────────────────────────────────
    col_width = max(14, *(len(m["name"]) for m in metrics_cfg))
    header = "    #     | " + " | ".join(f"{m['name']:>{col_width}}" for m in metrics_cfg)
    print(header)
    print(f"    {'─' * (len(header) - 4)}")

    for t in pareto_sorted:
        vals = " | ".join(f"{v:>{col_width}.6f}" for v in t.values)
        print(f"    #{t.number:04d}  | {vals}")

    # ── Best params (top Pareto trial by primary) ──────────────────────────
    best = pareto_sorted[0]
    print(f"\n  Best params (#{best.number:04d}, {primary_name}={best.values[primary_idx]:.6f}):")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    _print_rerun_commands(pareto_sorted, cfg, fixed_strs, metrics_cfg, primary_idx, is_multi=True)


def _print_single_objective_summary(
    study: "optuna.Study",
    cfg: DictConfig,
    metrics_cfg: list[dict],
    fixed_strs: list[str],
) -> None:
    """Print top-K summary for single-objective studies."""
    metric = metrics_cfg[0]["name"]
    is_max = metrics_cfg[0]["direction"] == "maximize"

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=is_max)
    top_k = min(5, len(sorted_trials))

    print(f"\n  Best {metric}: {study.best_value:.6f} (trial #{study.best_trial.number})")
    print(f"\n  Top {top_k} trials:")
    for i, t in enumerate(sorted_trials[:top_k], 1):
        print(f"    {i}. #{t.number:04d} | {metric}={t.value:.6f}")

    print("\n  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    _print_rerun_commands(sorted_trials, cfg, fixed_strs, metrics_cfg, 0, is_multi=False)


def save_results(study: "optuna.Study", cfg: DictConfig) -> Path:
    """Save sweep results to JSON alongside the SQLite database."""
    metrics_cfg = _get_metrics_cfg(cfg.sweep_space)
    is_multi = len(metrics_cfg) > 1
    primary_idx = _primary_index(metrics_cfg)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]

    results: dict = {
        "study_name": cfg.study_name,
        "pipeline": cfg.sweep_space.pipeline,
        "objective_metrics": [
            {
                "name": m["name"],
                "direction": m["direction"],
                "primary": m.get("primary", False),
            }
            for m in metrics_cfg
        ],
        "sampler": cfg.sampler,
        "seed": cfg.seed,
        "n_trials_total": len(study.trials),
        "n_trials_completed": len(completed),
        "screening_enabled": OmegaConf.to_container(
            cfg.sweep_space.get("screening", {}), resolve=True
        ).get("enabled", False),
    }

    # ── Best trial(s) ─────────────────────────────────────────────────────
    if is_multi and completed:
        pareto = study.best_trials
        is_max = metrics_cfg[primary_idx]["direction"] == "maximize"
        pareto_sorted = sorted(pareto, key=lambda t: t.values[primary_idx], reverse=is_max)
        results["pareto_front"] = [
            {
                "number": t.number,
                "values": {m["name"]: t.values[i] for i, m in enumerate(metrics_cfg)},
                "params": t.params,
            }
            for t in pareto_sorted
        ]
        # Convenience: best by primary metric
        best = pareto_sorted[0] if pareto_sorted else None
        results["best_by_primary"] = {
            "trial": best.number if best else None,
            "metric": metrics_cfg[primary_idx]["name"],
            "value": best.values[primary_idx] if best else None,
            "all_values": (
                {m["name"]: best.values[i] for i, m in enumerate(metrics_cfg)} if best else None
            ),
            "params": best.params if best else None,
        }
    elif completed:
        results["best_trial"] = study.best_trial.number
        results["best_value"] = study.best_value
        results["best_params"] = study.best_params
    else:
        results["best_trial"] = None
        results["best_value"] = None
        results["best_params"] = None

    # ── All trials ─────────────────────────────────────────────────────────
    results["trials"] = []
    for t in study.trials:
        trial_data: dict = {
            "number": t.number,
            "params": t.params,
            "state": t.state.name,
            "duration_s": (
                (t.datetime_complete - t.datetime_start).total_seconds()
                if t.datetime_complete and t.datetime_start
                else None
            ),
        }
        if t.state == TrialState.COMPLETE:
            if is_multi:
                trial_data["values"] = {m["name"]: t.values[i] for i, m in enumerate(metrics_cfg)}
            else:
                trial_data["value"] = t.value
        results["trials"].append(trial_data)

    results_path = Path(cfg.storage).with_suffix(".json")
    results_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Results saved to {results_path}")
    return results_path


# ═════════════════════════════════════════════════════════════════════════════
# Top-K revalidation + evaluation
# ═════════════════════════════════════════════════════════════════════════════


def _get_top_k_trials(
    study: "optuna.Study",
    metrics_cfg: list[dict],
    top_k: int,
) -> list:
    """Extract the top-K completed trials, ranked by primary metric."""
    is_multi = len(metrics_cfg) > 1
    primary_idx = _primary_index(metrics_cfg)
    is_max = metrics_cfg[primary_idx]["direction"] == "maximize"

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        return []

    if is_multi:
        pareto = study.best_trials
        sorted_trials = sorted(pareto, key=lambda t: t.values[primary_idx], reverse=is_max)
    else:
        sorted_trials = sorted(completed, key=lambda t: t.value, reverse=is_max)

    return sorted_trials[: min(top_k, len(sorted_trials))]


def _build_revalidation_overrides(
    trial,
    cfg: DictConfig,
) -> list[str]:
    """Build Hydra CLI overrides for a revalidation run (full CV + finalize + eval).

    Starts from the trial's sampled params, adds fixed overrides (without
    screening), and ensures finalization is enabled.
    """
    space_cfg = cfg.sweep_space
    revalid_cfg = OmegaConf.to_container(cfg.get("revalidation", {}), resolve=True)

    # ── 1. Trial params ──────────────────────────────────────────────────
    overrides = [f"{k}={_format_value(v)}" for k, v in trial.params.items()]

    # ── 2. Fixed overrides (same as sweep) ───────────────────────────────
    fixed = OmegaConf.to_container(space_cfg.get("fixed_overrides", {}), resolve=True)
    for k, v in _flatten_dict(fixed):
        overrides.append(f"{k}={_format_value(v)}")

    # ── 2b. Extra raw overrides ──────────────────────────────────────────
    extra = OmegaConf.to_container(space_cfg.get("extra_overrides", []), resolve=True)
    overrides.extend(extra)

    # ── 3. Revalidation-specific overrides (full CV, no screening) ───────
    revalid_overrides = revalid_cfg.get("overrides", {})
    for k, v in _flatten_dict(revalid_overrides):
        overrides.append(f"{k}={_format_value(v)}")

    # Ensure finalization runs
    overrides.append("training.skip_finalize=false")

    return overrides


def _run_revalidation_trial(
    trial,
    trial_rank: int,
    cfg: DictConfig,
    revalid_dir: Path,
    gpu_id: int,
) -> dict:
    """Run a single revalidation trial: train.py (full CV) then evaluate.py.

    Returns a dict with training and evaluation results, or an error key.
    """
    space_cfg = cfg.sweep_space
    revalid_cfg = OmegaConf.to_container(cfg.get("revalidation", {}), resolve=True)
    trial_timeout = revalid_cfg.get("trial_timeout", 14400)

    overrides = _build_revalidation_overrides(trial, cfg)
    exp_name = f"sweep/{cfg.study_name}/revalid_{trial_rank:02d}_trial_{trial.number:04d}"
    overrides.append(f"exp_name={exp_name}")

    # W&B control
    if cfg.get("disable_wandb", True):
        overrides.append("wandb.offline=true")

    param_str = " ".join(f"{k}={_format_value(v)}" for k, v in trial.params.items())
    logger.info(f"Revalid #{trial_rank} (trial {trial.number:04d}) | {param_str}")

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    result: dict = {
        "rank": trial_rank,
        "trial_number": trial.number,
        "params": dict(trial.params),
    }

    # ── Phase 1: Full CV training ────────────────────────────────────────
    train_script = str(PROJECT_ROOT / "scripts" / space_cfg.target_script)
    train_cmd = [sys.executable, train_script, *overrides]

    logger.info(f"  Phase 1: Full CV training → {exp_name}")
    t_start = time.time()

    sem = _gpu_semaphores.get(gpu_id)
    if sem is not None:
        sem.acquire()
    try:
        train_result = subprocess.run(
            train_cmd,
            env=env,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=trial_timeout,
        )
    except subprocess.TimeoutExpired:
        logger.error(f"  Revalid #{trial_rank} training timed out after {trial_timeout}s")
        result["error"] = "training_timeout"
        return result
    finally:
        if sem is not None:
            sem.release()

    train_elapsed = time.time() - t_start

    # Check training output
    output_dir = Path(cfg.output_root) / "train" / exp_name
    cv_summary_path = output_dir / "cv_summary.json"

    if not cv_summary_path.is_file():
        rc = train_result.returncode
        # Negative rc = killed by signal (e.g. CUDA cleanup); positive = real error
        if rc is not None and rc > 0:
            stderr_tail = (train_result.stderr or "")[-2000:]
            logger.error(f"  Revalid #{trial_rank} training failed (rc={rc}):\n{stderr_tail}")
        elif rc is not None and rc < 0:
            logger.warning(
                f"  Revalid #{trial_rank} exited with signal {-rc} and no cv_summary.json"
            )
        result["error"] = f"training_failed_rc={rc}"
        return result

    # Parse CV summary
    cv_summary = json.loads(cv_summary_path.read_text())
    result["cv_summary"] = cv_summary
    result["train_dir"] = str(output_dir)
    result["train_elapsed_s"] = round(train_elapsed, 1)
    logger.info(f"  Phase 1 complete: {train_elapsed / 60:.1f}min")

    # ── Phase 2: Evaluation ──────────────────────────────────────────────
    run_eval = revalid_cfg.get("run_eval", True)
    if not run_eval:
        logger.info("  Phase 2: Evaluation skipped (revalidation.run_eval=false)")
        result["eval_skipped"] = True
        return result

    eval_script = str(PROJECT_ROOT / "scripts" / "evaluate.py")

    # Build eval overrides: reuse the same config groups so paths resolve.
    # overrides already contains data/encoder/splits/model/training + exp_name.
    eval_overrides = list(overrides)
    eval_overrides.append(f"train_dir={output_dir}")

    # Eval-specific settings
    eval_n_bootstrap = revalid_cfg.get("n_bootstrap", 2000)
    eval_overrides.append(f"eval.n_bootstrap={eval_n_bootstrap}")
    # Reset class_names to null so evaluate.py uses numeric indices
    # (the hardcoded default in evaluate.yaml is GEJ-specific)
    eval_overrides.append("eval.class_names=null")

    # Enable W&B for eval if requested
    enable_eval_wandb = revalid_cfg.get("enable_eval_wandb", False)
    if not enable_eval_wandb:
        eval_overrides.append("wandb.offline=true")

    eval_cmd = [sys.executable, eval_script, *eval_overrides]

    logger.info(f"  Phase 2: Evaluation → {exp_name}")
    t_eval_start = time.time()

    if sem is not None:
        sem.acquire()
    try:
        eval_result = subprocess.run(
            eval_cmd,
            env=env,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=trial_timeout,
        )
    except subprocess.TimeoutExpired:
        logger.error(f"  Revalid #{trial_rank} evaluation timed out")
        result["eval_error"] = "eval_timeout"
        return result
    finally:
        if sem is not None:
            sem.release()

    eval_elapsed = time.time() - t_eval_start

    # Positive rc = real error; negative = signal (tolerate, check for output)
    if eval_result.returncode is not None and eval_result.returncode > 0:
        stderr_tail = (eval_result.stderr or "")[-2000:]
        logger.error(
            f"  Revalid #{trial_rank} evaluation failed "
            f"(rc={eval_result.returncode}):\n{stderr_tail}"
        )
        result["eval_error"] = f"eval_failed_rc={eval_result.returncode}"
        return result

    # Parse eval results
    eval_dir = output_dir / "eval"
    result["eval_dir"] = str(eval_dir)
    result["eval_elapsed_s"] = round(eval_elapsed, 1)

    # Read recommended model and its metrics
    rec_path = eval_dir / "recommended_model.json"
    if rec_path.is_file():
        rec = json.loads(rec_path.read_text())
        result["recommended_model"] = rec.get("recommended_model")
        result["eval_ranking"] = rec.get("ranking")
        result["eval_composite_scores"] = rec.get("composite_scores")

    # Read comparison for detailed metrics
    comp_path = eval_dir / "model_comparison.json"
    if comp_path.is_file():
        comp = json.loads(comp_path.read_text())
        result["eval_comparison"] = comp

    logger.info(
        f"  Phase 2 complete: {eval_elapsed / 60:.1f}min | "
        f"recommended={result.get('recommended_model', '?')}"
    )

    return result


def run_revalidation(
    study: "optuna.Study",
    cfg: DictConfig,
) -> list[dict]:
    """Run top-K revalidation: full CV training + evaluation for best trials.

    Returns a list of result dicts, sorted by eval composite score.
    """
    revalid_cfg = OmegaConf.to_container(cfg.get("revalidation", {}), resolve=True)

    if not revalid_cfg.get("enabled", False):
        logger.info("Revalidation disabled (revalidation.enabled=false)")
        return []

    top_k = revalid_cfg.get("top_k", 10)
    metrics_cfg = _get_metrics_cfg(cfg.sweep_space)
    top_trials = _get_top_k_trials(study, metrics_cfg, top_k)

    if not top_trials:
        logger.warning("No completed trials to revalidate")
        return []

    logger.info("=" * 70)
    logger.info(f"  REVALIDATION: Top {len(top_trials)} configs with full CV + evaluation")
    logger.info("=" * 70)

    gpu_ids = list(cfg.get("gpu_ids", [0]))
    revalid_dir = Path(cfg.output_root) / "sweeps" / cfg.study_name / "revalidation"
    revalid_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for rank, trial in enumerate(top_trials):
        gpu_id = gpu_ids[rank % len(gpu_ids)]
        try:
            r = _run_revalidation_trial(trial, rank, cfg, revalid_dir, gpu_id)
        except Exception as e:
            logger.error(f"Revalid #{rank} (trial {trial.number}) crashed: {e}")
            r = {
                "rank": rank,
                "trial_number": trial.number,
                "params": dict(trial.params),
                "error": str(e),
            }
        results.append(r)

    # ── Rank by eval composite score ─────────────────────────────────────
    def _sort_key(r):
        if "error" in r or "eval_error" in r:
            return -1.0
        scores = r.get("eval_composite_scores", {})
        rec = r.get("recommended_model", "")
        if rec and rec in scores:
            return scores[rec].get("composite", 0.0)
        # Fall back to best composite across all strategies
        best = 0.0
        for s in scores.values():
            if isinstance(s, dict):
                best = max(best, s.get("composite", 0.0))
        return best

    results.sort(key=_sort_key, reverse=True)

    # ── Save results ─────────────────────────────────────────────────────
    results_path = revalid_dir / "revalidation_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Revalidation results → {results_path}")

    # ── Print summary ────────────────────────────────────────────────────
    _print_revalidation_summary(results, metrics_cfg)

    return results


def _print_revalidation_summary(
    results: list[dict],
    metrics_cfg: list[dict],
) -> None:
    """Print a ranked summary of revalidation results."""
    primary_idx = _primary_index(metrics_cfg)
    primary_name = metrics_cfg[primary_idx]["name"]

    print(f"\n{'=' * 70}")
    print("  REVALIDATION RESULTS (ranked by eval composite score)")
    print(f"{'=' * 70}")

    for i, r in enumerate(results):
        trial_num = r.get("trial_number", "?")
        if "error" in r:
            print(f"  {i + 1:2d}. Trial #{trial_num:04d} — FAILED: {r['error']}")
            continue
        if "eval_error" in r:
            # Still has CV results
            cv = r.get("cv_summary", {})
            primary_mean = cv.get(f"{primary_name}_mean", "?")
            if isinstance(primary_mean, float):
                primary_mean = f"{primary_mean:.4f}"
            print(
                f"  {i + 1:2d}. Trial #{trial_num:04d} — "
                f"CV {primary_name}={primary_mean} (eval failed: {r['eval_error']})"
            )
            continue

        rec = r.get("recommended_model", "?")
        scores = r.get("eval_composite_scores", {})
        rec_scores = scores.get(rec, {})
        composite = rec_scores.get("composite", 0)

        # Get patient AUROC from eval comparison
        comp = r.get("eval_comparison", {})
        models = comp.get("models", {})
        rec_model = models.get(rec, {})
        patient_auroc = rec_model.get("patient_level", {}).get("auroc", {}).get("point")
        auroc_str = f"{patient_auroc:.4f}" if isinstance(patient_auroc, float) else "N/A"

        param_str = ", ".join(f"{k}={v}" for k, v in r.get("params", {}).items())

        print(
            f"  {i + 1:2d}. Trial #{trial_num:04d} | "
            f"composite={composite:.4f} | "
            f"patient_auroc={auroc_str} | "
            f"best={rec} | "
            f"{param_str}"
        )

    # Best config
    if results and "error" not in results[0] and "eval_error" not in results[0]:
        best = results[0]
        print(f"\n  {'─' * 66}")
        print(f"  BEST CONFIG (Trial #{best.get('trial_number', '?'):04d}):")
        for k, v in best.get("params", {}).items():
            print(f"    {k}: {v}")
        print(f"  Recommended strategy: {best.get('recommended_model', '?')}")
        print(f"  Train dir: {best.get('train_dir', '?')}")
        print(f"  Eval dir:  {best.get('eval_dir', '?')}")
    print(f"{'=' * 70}\n")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


@hydra.main(config_path="../configs", config_name="sweep", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if optuna is None:
        print("Optuna is required for hyperparameter sweeps.\nInstall it with: pip install optuna")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | sweep | %(message)s",
        force=True,
    )

    # Suppress Optuna's verbose internal logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    space_cfg = cfg.sweep_space
    metrics_cfg = _get_metrics_cfg(space_cfg)
    is_multi = len(metrics_cfg) > 1

    logger.info(f"Study: {cfg.study_name}")
    logger.info(f"Pipeline: {space_cfg.pipeline} ({space_cfg.target_script})")

    obj_str = ", ".join(f"{m['direction']} {m['name']}" for m in metrics_cfg)
    if is_multi:
        primary_name = metrics_cfg[_primary_index(metrics_cfg)]["name"]
        logger.info(f"Objectives: {obj_str} (primary: {primary_name})")
    else:
        logger.info(f"Objective: {obj_str}")

    logger.info(f"Trials: {cfg.n_trials} | Jobs: {cfg.n_jobs} | GPUs: {list(cfg.gpu_ids)}")
    logger.info(f"Screening: {'ON' if space_cfg.screening.get('enabled', False) else 'OFF'}")

    # ── Storage ────────────────────────────────────────────────────────────
    storage_path = Path(cfg.storage)
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Sampler ────────────────────────────────────────────────────────────
    chosen_sampler = cfg.sampler

    if is_multi:
        # Multi-objective samplers (CMA-ES doesn't support multi-objective)
        if chosen_sampler == "cmaes":
            logger.warning(
                "CMA-ES does not support multi-objective optimization. Falling back to TPE."
            )
            chosen_sampler = "tpe"
        samplers = {
            "tpe": lambda: optuna.samplers.TPESampler(seed=cfg.seed),
            "nsgaii": lambda: optuna.samplers.NSGAIISampler(seed=cfg.seed),
            "random": lambda: optuna.samplers.RandomSampler(seed=cfg.seed),
        }
    else:
        samplers = {
            "tpe": lambda: optuna.samplers.TPESampler(seed=cfg.seed),
            "random": lambda: optuna.samplers.RandomSampler(seed=cfg.seed),
            "cmaes": lambda: optuna.samplers.CmaEsSampler(seed=cfg.seed),
        }

    if chosen_sampler not in samplers:
        raise ValueError(f"Unknown sampler '{chosen_sampler}'. Choose from: {list(samplers)}")
    sampler = samplers[chosen_sampler]()
    logger.info(f"Sampler: {chosen_sampler} ({type(sampler).__name__})")

    # ── Create or resume study ─────────────────────────────────────────────
    directions = [m["direction"] for m in metrics_cfg]

    if is_multi:
        study = optuna.create_study(
            study_name=cfg.study_name,
            storage=f"sqlite:///{cfg.storage}",
            directions=directions,
            sampler=sampler,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            study_name=cfg.study_name,
            storage=f"sqlite:///{cfg.storage}",
            direction=directions[0],
            sampler=sampler,
            load_if_exists=True,
        )

    n_existing = len(study.trials)
    if n_existing > 0:
        n_complete = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        logger.info(f"Resuming study: {n_existing} existing trials ({n_complete} completed)")

    # ── GPU concurrency control ───────────────────────────────────────────
    gpu_ids = list(cfg.get("gpu_ids", [0]))
    max_per_gpu = int(cfg.get("max_concurrent_per_gpu", 1))

    _gpu_semaphores.clear()
    for gid in gpu_ids:
        _gpu_semaphores[gid] = threading.Semaphore(max_per_gpu)

    effective_parallel = len(gpu_ids) * max_per_gpu
    if cfg.n_jobs > effective_parallel:
        logger.warning(
            f"n_jobs ({cfg.n_jobs}) > GPUs ({len(gpu_ids)}) x "
            f"max_concurrent_per_gpu ({max_per_gpu}) = {effective_parallel}. "
            f"Excess threads will queue for a free GPU slot."
        )

    # ── Validate mmap store ────────────────────────────────────────────────
    _log_mmap_info(cfg)

    # ── Optimize ───────────────────────────────────────────────────────────
    study.optimize(
        lambda trial: run_trial(trial, cfg),
        n_trials=cfg.n_trials,
        n_jobs=cfg.n_jobs,
        show_progress_bar=True,
    )

    # ── Report ─────────────────────────────────────────────────────────────
    print_study_summary(study, cfg)
    save_results(study, cfg)

    # ── Top-K Revalidation + Evaluation ───────────────────────────────────
    run_revalidation(study, cfg)


if __name__ == "__main__":
    main()
