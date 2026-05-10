"""
Hyperparameter sweep orchestrator for supervised MIL training.

Uses Optuna TPE for Bayesian optimization with subprocess-based trial isolation.
Single-objective optimization on the primary metric (test/auroc by default),
with additional metrics tracked as trial user attributes for post-hoc analysis.

Each trial runs as an independent subprocess, ensuring:
  - No GPU memory leaks between trials
  - Full Hydra config resolution per trial
  - Crash isolation (a failed trial doesn't kill the sweep)

Two-stage flow:
  Stage 1 (screening): holdout split for fast trial ranking
  Stage 2 (revalidation): top-K configs get full k-fold CV + finalization +
    evaluate.py with bootstrap CIs and patient-level metrics

Usage:
    python scripts/sweep.py sweep_space=supervised
    python scripts/sweep.py sweep_space=supervised n_trials=60 n_jobs=2 gpu_ids=[0,1]
    python scripts/sweep.py study_name=supervised_sweep_20260315_120000  # resume
    python scripts/sweep.py sweep_space=supervised sweep_space.screening.enabled=false

Requires: pip install optuna
"""

import contextlib
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        # Hydra list syntax: [a,b,c] with no spaces
        return "[" + ",".join(_format_value(x) for x in v) + "]"
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


def _get_objective_cfg(space_cfg) -> dict:
    """Extract single-objective config from sweep space.

    Returns a dict with keys: metric, direction.
    """
    obj = OmegaConf.to_container(space_cfg.objective, resolve=True)
    return {"metric": obj["metric"], "direction": obj["direction"]}


def _get_tracked_metrics(space_cfg) -> list[str]:
    """Get list of additional metrics to track as user attributes."""
    return list(OmegaConf.to_container(space_cfg.get("tracked_metrics", []), resolve=True))


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
    return Path(cfg.output_root) / "train" / trial_exp_name


def parse_metric(output_dir: Path, metric: str) -> float:
    """Extract a single metric value from training output files.

    Search order:
      1. cv_summary.json    — supervised CV aggregated results
      2. fold_0/fold_metrics.json — supervised single-fold fallback

    For fold_metrics.json, also tries stripping ``_mean`` / ``_std`` suffixes
    so that ``val/auroc_mean`` can fall back to ``val/auroc`` in a single fold.
    """
    # 1. CV summary (always written by train.py, even for 1 fold)
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

    raise FileNotFoundError(
        f"Metric '{metric}' not found in {output_dir}.\n"
        f"Searched: cv_summary.json, fold_0/fold_metrics.json"
    )


def parse_tracked_metrics(output_dir: Path, tracked: list[str]) -> dict[str, float]:
    """Parse tracked metrics, returning only those that are available.

    Unlike the primary objective, missing tracked metrics are not fatal.
    """
    values = {}
    for metric in tracked:
        with contextlib.suppress(FileNotFoundError, ValueError):
            values[metric] = parse_metric(output_dir, metric)
    return values


# ═════════════════════════════════════════════════════════════════════════════
# Trial execution
# ═════════════════════════════════════════════════════════════════════════════


def run_trial(trial: "optuna.Trial", cfg: DictConfig) -> float:
    """Execute a single Optuna trial as a subprocess.

    Returns the primary objective metric value.
    """
    space_cfg = cfg.sweep_space
    search_space = OmegaConf.to_container(space_cfg.search_space, resolve=True)
    obj_cfg = _get_objective_cfg(space_cfg)
    tracked = _get_tracked_metrics(space_cfg)

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
        trial.set_user_attr("failure_reason", "timeout")
        timeout_val = cfg.get("trial_timeout", 7200)
        logger.error(f"Trial {trial.number:04d} timed out after {timeout_val}s")
        raise optuna.TrialPruned("Timed out") from err
    finally:
        if sem is not None:
            sem.release()

    elapsed = time.time() - t_start
    trial.set_user_attr("elapsed_s", round(elapsed, 1))

    # ── 9. Parse primary objective ─────────────────────────────────────────
    output_dir = resolve_output_dir(cfg, trial_exp_name)

    if result.returncode != 0:
        is_signal = result.returncode < 0
        if is_signal:
            # Negative return codes (e.g. -6/SIGABRT, -11/SIGSEGV) often come
            # from PyTorch/CUDA cleanup during Python shutdown — the training
            # itself may have completed and written valid output files.
            logger.warning(
                f"Trial {trial.number:04d} exited with signal "
                f"{-result.returncode} (rc={result.returncode}); "
                f"checking for output files..."
            )
        else:
            stderr_tail = (result.stderr or "")[-3000:]
            # Classify common failure reasons from stderr
            failure_reason = _classify_failure(result.returncode, stderr_tail)
            trial.set_user_attr("failure_reason", failure_reason)
            logger.error(
                f"Trial {trial.number:04d} failed ({failure_reason}, "
                f"rc={result.returncode}):\n{stderr_tail}"
            )

    try:
        primary_value = parse_metric(output_dir, obj_cfg["metric"])
    except (FileNotFoundError, ValueError) as e:
        trial.set_user_attr("failure_reason", trial.user_attrs.get("failure_reason", "no_output"))
        logger.error(f"Trial {trial.number:04d} | {e}")
        raise optuna.TrialPruned(str(e)) from e

    if math.isnan(primary_value):
        trial.set_user_attr("failure_reason", "nan_metric")
        raise optuna.TrialPruned(f"{obj_cfg['metric']} is NaN")

    # ── 10. Parse and store tracked metrics as user attrs ──────────────────
    tracked_values = parse_tracked_metrics(output_dir, tracked)
    for name, v in tracked_values.items():
        trial.set_user_attr(name, v)

    # Also store the primary metric for uniform access via user attrs
    trial.set_user_attr(obj_cfg["metric"], primary_value)

    metric_parts = [f"{obj_cfg['metric']}={primary_value:.6f}"]
    metric_parts.extend(f"{k}={v:.6f}" for k, v in tracked_values.items())
    logger.info(f"Trial {trial.number:04d} | {' | '.join(metric_parts)} | {elapsed / 60:.1f}min")

    return primary_value


def _classify_failure(returncode: int, stderr: str) -> str:
    """Classify a trial failure reason from the return code and stderr."""
    stderr_lower = stderr.lower()
    if "cuda out of memory" in stderr_lower or "cudnn error" in stderr_lower:
        return "oom"
    if "nan" in stderr_lower and ("loss" in stderr_lower or "gradient" in stderr_lower):
        return "nan_divergence"
    if "killed" in stderr_lower or returncode == -9:
        return "killed"
    if "modulenotfounderror" in stderr_lower or "importerror" in stderr_lower:
        return "import_error"
    if "filenotfounderror" in stderr_lower:
        return "file_not_found"
    if "runtimeerror" in stderr_lower:
        return "runtime_error"
    return f"unknown_rc={returncode}"


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


def print_study_summary(study: "optuna.Study", cfg: DictConfig) -> None:
    """Print sweep summary with top trials and re-run commands."""
    obj_cfg = _get_objective_cfg(cfg.sweep_space)
    metric = obj_cfg["metric"]
    is_max = obj_cfg["direction"] == "maximize"

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]

    print(f"\n{'=' * 70}")
    print(f"  Sweep Complete: {cfg.study_name}")
    print(f"{'=' * 70}")
    print(f"  Completed: {len(completed)} / {len(study.trials)}")
    if pruned:
        # Breakdown failure reasons
        reasons: dict[str, int] = {}
        for t in pruned:
            reason = t.user_attrs.get("failure_reason", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        reason_str = ", ".join(f"{r}: {c}" for r, c in sorted(reasons.items()))
        print(f"  Pruned/Failed: {len(pruned)} ({reason_str})")

    if not completed:
        print("  No completed trials.")
        print(f"{'=' * 70}")
        return

    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=is_max)
    top_k = min(5, len(sorted_trials))

    print(f"\n  Best {metric}: {study.best_value:.6f} (trial #{study.best_trial.number})")
    print(f"\n  Top {top_k} trials:")

    # Show tracked metrics alongside primary
    tracked = _get_tracked_metrics(cfg.sweep_space)
    for i, t in enumerate(sorted_trials[:top_k], 1):
        parts = [f"{metric}={t.value:.6f}"]
        for tm in tracked:
            tv = t.user_attrs.get(tm)
            if tv is not None:
                parts.append(f"{tm}={tv:.4f}")
        elapsed = t.user_attrs.get("elapsed_s")
        time_str = f" | {elapsed / 60:.1f}min" if elapsed is not None else ""
        print(f"    {i}. #{t.number:04d} | {' | '.join(parts)}{time_str}")

    print("\n  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Re-run commands
    fixed_strs = _get_fixed_overrides_strs(cfg)
    print(f"\n  Re-run top {top_k} with full validation:")
    for t in sorted_trials[:top_k]:
        param_strs = [f"{k}={_format_value(v)}" for k, v in t.params.items()]
        all_overrides = fixed_strs + param_strs
        cmd = f"python scripts/{cfg.sweep_space.target_script} {' '.join(all_overrides)}"
        print(f"    # Trial #{t.number:04d} ({metric}={t.value:.6f})")
        print(f"    {cmd}\n")

    print(f"  Storage: {cfg.storage}")
    print(f"{'=' * 70}")


def save_results(study: "optuna.Study", cfg: DictConfig) -> Path:
    """Save sweep results to JSON alongside the SQLite database."""
    obj_cfg = _get_objective_cfg(cfg.sweep_space)
    tracked = _get_tracked_metrics(cfg.sweep_space)
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]

    results: dict = {
        "study_name": cfg.study_name,
        "pipeline": cfg.sweep_space.pipeline,
        "objective": obj_cfg,
        "tracked_metrics": tracked,
        "sampler": cfg.sampler,
        "seed": cfg.seed,
        "n_trials_total": len(study.trials),
        "n_trials_completed": len(completed),
        "screening_enabled": OmegaConf.to_container(
            cfg.sweep_space.get("screening", {}), resolve=True
        ).get("enabled", False),
    }

    # ── Best trial ────────────────────────────────────────────────────────
    if completed:
        results["best_trial"] = study.best_trial.number
        results["best_value"] = study.best_value
        results["best_params"] = study.best_params
        # Include tracked metrics for best trial
        results["best_tracked"] = {m: study.best_trial.user_attrs.get(m) for m in tracked}
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
            trial_data["value"] = t.value
            # Include all tracked metrics
            trial_data["tracked"] = {m: t.user_attrs.get(m) for m in tracked}
        else:
            trial_data["failure_reason"] = t.user_attrs.get("failure_reason", "unknown")
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
    obj_cfg: dict,
    top_k: int,
) -> list:
    """Extract the top-K completed trials, ranked by primary metric."""
    is_max = obj_cfg["direction"] == "maximize"
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        return []
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

    # ── 2. Fixed overrides (same as sweep — uses full kfold, not screening) ─
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

    # Ensure finalization runs (redundant if already in revalidation.overrides,
    # but safe — Hydra last-wins semantics guarantee correctness)
    if not any(o.startswith("training.skip_finalize=") for o in overrides):
        overrides.append("training.skip_finalize=false")

    return overrides


def _run_revalidation_trial(
    trial,
    trial_rank: int,
    cfg: DictConfig,
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
    logger.info(f"Revalid #{trial_rank} (trial {trial.number:04d}) | GPU {gpu_id} | {param_str}")

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    result: dict = {
        "rank": trial_rank,
        "trial_number": trial.number,
        "params": dict(trial.params),
        "screening_value": trial.value,
    }

    # ── Phase 1: Full CV training ────────────────────────────────────────
    train_script = str(PROJECT_ROOT / "scripts" / space_cfg.target_script)
    train_cmd = [sys.executable, train_script, *overrides]

    logger.info(f"  Revalid #{trial_rank}: Phase 1 — Full CV training → {exp_name}")
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
        if rc is not None and rc > 0:
            stderr_tail = (train_result.stderr or "")[-2000:]
            failure = _classify_failure(rc, stderr_tail)
            logger.error(f"  Revalid #{trial_rank} training failed ({failure}):\n{stderr_tail}")
            result["error"] = f"training_failed_{failure}"
        elif rc is not None and rc < 0:
            logger.warning(
                f"  Revalid #{trial_rank} exited with signal {-rc} and no cv_summary.json"
            )
            result["error"] = f"training_signal_{-rc}"
        else:
            result["error"] = "no_cv_summary"
        return result

    # Parse CV summary
    cv_summary = json.loads(cv_summary_path.read_text())
    result["cv_summary"] = cv_summary
    result["train_dir"] = str(output_dir)
    result["train_elapsed_s"] = round(train_elapsed, 1)
    logger.info(f"  Revalid #{trial_rank}: Phase 1 complete ({train_elapsed / 60:.1f}min)")

    # ── Phase 2: Evaluation ──────────────────────────────────────────────
    run_eval = revalid_cfg.get("run_eval", True)
    if not run_eval:
        logger.info(f"  Revalid #{trial_rank}: Phase 2 skipped (run_eval=false)")
        result["eval_skipped"] = True
        return result

    eval_script = str(PROJECT_ROOT / "scripts" / "evaluate.py")

    # Build eval overrides: reuse the same config groups so paths resolve.
    eval_overrides = list(overrides)
    eval_overrides.append(f"train_dir={output_dir}")

    # Eval-specific settings
    eval_n_bootstrap = revalid_cfg.get("n_bootstrap", 2000)
    eval_overrides.append(f"eval.n_bootstrap={eval_n_bootstrap}")
    # Reset class_names to null so evaluate.py uses numeric indices
    eval_overrides.append("eval.class_names=null")

    enable_eval_wandb = revalid_cfg.get("enable_eval_wandb", False)
    if not enable_eval_wandb:
        eval_overrides.append("wandb.offline=true")

    eval_cmd = [sys.executable, eval_script, *eval_overrides]

    logger.info(f"  Revalid #{trial_rank}: Phase 2 — Evaluation → {exp_name}")
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

    if eval_result.returncode is not None and eval_result.returncode > 0:
        stderr_tail = (eval_result.stderr or "")[-2000:]
        logger.error(
            f"  Revalid #{trial_rank} evaluation failed "
            f"(rc={eval_result.returncode}):\n{stderr_tail}"
        )
        result["eval_error"] = f"eval_failed_rc={eval_result.returncode}"
        return result

    if eval_result.returncode is not None and eval_result.returncode < 0:
        logger.warning(
            f"  Revalid #{trial_rank} evaluation exited with signal "
            f"{-eval_result.returncode} (rc={eval_result.returncode}); "
            f"checking for output files..."
        )

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
        f"  Revalid #{trial_rank}: Phase 2 complete ({eval_elapsed / 60:.1f}min) | "
        f"recommended={result.get('recommended_model', '?')}"
    )

    return result


def run_revalidation(
    study: "optuna.Study",
    cfg: DictConfig,
) -> list[dict]:
    """Run top-K revalidation: full CV training + evaluation for best trials.

    Runs trials in parallel across available GPUs using ThreadPoolExecutor,
    with per-GPU semaphores controlling concurrency.

    Returns a list of result dicts, sorted by eval composite score.
    """
    revalid_cfg = OmegaConf.to_container(cfg.get("revalidation", {}), resolve=True)

    if not revalid_cfg.get("enabled", False):
        logger.info("Revalidation disabled (revalidation.enabled=false)")
        return []

    top_k = revalid_cfg.get("top_k", 10)
    obj_cfg = _get_objective_cfg(cfg.sweep_space)
    top_trials = _get_top_k_trials(study, obj_cfg, top_k)

    if not top_trials:
        logger.warning("No completed trials to revalidate")
        return []

    logger.info("=" * 70)
    logger.info(f"  REVALIDATION: Top {len(top_trials)} configs with full CV + evaluation")
    logger.info("=" * 70)

    gpu_ids = list(cfg.get("gpu_ids", [0]))

    # ── Parallel revalidation ────────────────────────────────────────────
    # Use ThreadPoolExecutor with the same concurrency model as screening:
    # multiple workers round-robin across GPUs, with per-GPU semaphores
    # preventing oversubscription. Defaults to the sweep's n_jobs.
    max_workers = revalid_cfg.get("n_jobs", cfg.get("n_jobs", 1))
    logger.info(f"  Parallel workers: {max_workers} | GPUs: {gpu_ids}")
    results = [None] * len(top_trials)

    def _run_one(rank_trial):
        rank, trial = rank_trial
        gpu_id = gpu_ids[rank % len(gpu_ids)]
        try:
            return rank, _run_revalidation_trial(trial, rank, cfg, gpu_id)
        except Exception as e:
            logger.error(f"Revalid #{rank} (trial {trial.number}) crashed: {e}")
            return rank, {
                "rank": rank,
                "trial_number": trial.number,
                "params": dict(trial.params),
                "screening_value": trial.value,
                "error": str(e),
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_run_one, (rank, trial)) for rank, trial in enumerate(top_trials)
        ]
        for future in as_completed(futures):
            rank, r = future.result()
            results[rank] = r

    # ── Rank by eval composite score ─────────────────────────────────────
    def _sort_key(r):
        if r is None or "error" in r or "eval_error" in r:
            return -1.0
        # Eval was skipped — fall back to CV summary primary metric
        if r.get("eval_skipped"):
            cv = r.get("cv_summary", {})
            obj_metric = obj_cfg["metric"]
            for key in [obj_metric, f"{obj_metric}_mean"]:
                if key in cv and cv[key] is not None:
                    return float(cv[key])
            return 0.0
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

    results = [r for r in results if r is not None]
    results.sort(key=_sort_key, reverse=True)

    # ── Save results ─────────────────────────────────────────────────────
    revalid_dir = Path(cfg.output_root) / "sweeps" / cfg.study_name / "revalidation"
    revalid_dir.mkdir(parents=True, exist_ok=True)
    results_path = revalid_dir / "revalidation_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Revalidation results → {results_path}")

    # ── Print summary ────────────────────────────────────────────────────
    _print_revalidation_summary(results, obj_cfg)

    return results


def _print_revalidation_summary(
    results: list[dict],
    obj_cfg: dict,
) -> None:
    """Print a ranked summary of revalidation results."""
    metric = obj_cfg["metric"]

    print(f"\n{'=' * 70}")
    print("  REVALIDATION RESULTS (ranked by eval composite score)")
    print(f"{'=' * 70}")

    for i, r in enumerate(results):
        trial_num = r.get("trial_number", "?")
        screening_val = r.get("screening_value")
        screen_str = (
            f" (screening {metric}={screening_val:.4f})" if screening_val is not None else ""
        )

        if "error" in r:
            print(f"  {i + 1:2d}. Trial #{trial_num:04d}{screen_str} — FAILED: {r['error']}")
            continue
        if "eval_error" in r:
            cv = r.get("cv_summary", {})
            primary_mean = cv.get(f"{metric}_mean", "?")
            if isinstance(primary_mean, float):
                primary_mean = f"{primary_mean:.4f}"
            print(
                f"  {i + 1:2d}. Trial #{trial_num:04d}{screen_str} — "
                f"CV {metric}={primary_mean} (eval failed: {r['eval_error']})"
            )
            continue
        if r.get("eval_skipped"):
            cv = r.get("cv_summary", {})
            primary_mean = cv.get(f"{metric}_mean", cv.get(metric, "?"))
            if isinstance(primary_mean, float):
                primary_mean = f"{primary_mean:.4f}"
            param_str = ", ".join(f"{k}={v}" for k, v in r.get("params", {}).items())
            print(
                f"  {i + 1:2d}. Trial #{trial_num:04d}{screen_str} — "
                f"CV {metric}={primary_mean} (eval skipped) | {param_str}"
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
        if best.get("eval_skipped"):
            cv = best.get("cv_summary", {})
            obj_metric = obj_cfg["metric"]
            cv_val = cv.get(f"{obj_metric}_mean", cv.get(obj_metric, "?"))
            if isinstance(cv_val, float):
                cv_val = f"{cv_val:.4f}"
            print(f"  CV {obj_metric}: {cv_val} (eval was skipped)")
        else:
            print(f"  Recommended strategy: {best.get('recommended_model', '?')}")
        print(f"  Train dir: {best.get('train_dir', '?')}")
        if not best.get("eval_skipped"):
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
    obj_cfg = _get_objective_cfg(space_cfg)
    tracked = _get_tracked_metrics(space_cfg)

    logger.info(f"Study: {cfg.study_name}")
    logger.info(f"Pipeline: {space_cfg.pipeline} ({space_cfg.target_script})")
    logger.info(f"Objective: {obj_cfg['direction']} {obj_cfg['metric']}")
    if tracked:
        logger.info(f"Tracked metrics: {tracked}")
    logger.info(f"Trials: {cfg.n_trials} | Jobs: {cfg.n_jobs} | GPUs: {list(cfg.gpu_ids)}")
    logger.info(f"Screening: {'ON' if space_cfg.screening.get('enabled', False) else 'OFF'}")

    # ── Storage ────────────────────────────────────────────────────────────
    storage_path = Path(cfg.storage)
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Sampler ────────────────────────────────────────────────────────────
    chosen_sampler = cfg.sampler
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
    study = optuna.create_study(
        study_name=cfg.study_name,
        storage=f"sqlite:///{cfg.storage}",
        direction=obj_cfg["direction"],
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
