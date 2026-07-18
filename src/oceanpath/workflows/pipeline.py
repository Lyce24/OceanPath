"""
Workflow orchestration for the essential OceanPath supervised pipeline.

Runs the six-stage extract/mmap/split/train/evaluate/export pipeline.

Usage:
    # Supervised pipeline
    python scripts/pipeline.py pipeline_profile=supervised \
        platform=local data=blca encoder=univ1 model=abmil

    # Validate DAG and freshness only
    python scripts/pipeline.py ... pipeline.dry_run=true
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from oceanpath.config import FoundationPaths
from oceanpath.runtime import setup_logging
from oceanpath.workflows.supervised_pipeline import build_supervised_pipeline

logger = logging.getLogger(__name__)


def _runtime_choices() -> dict[str, str]:
    runtime = HydraConfig.get().runtime
    choices = {}
    for key, val in runtime.choices.items():
        if isinstance(val, str):
            choices[key] = val
    return choices


def _runtime_task_overrides() -> list[str]:
    """Return explicit task overrides from the parent Hydra invocation."""
    try:
        return [str(value) for value in HydraConfig.get().overrides.task]
    except (AttributeError, ValueError):
        return []


def _override_root(override: str) -> str:
    """Extract the top-level config key from a Hydra override."""
    key = override.lstrip("+~").split("=", maxsplit=1)[0]
    return key.split(".", maxsplit=1)[0].split("@", maxsplit=1)[0]


def _forward_task_overrides(
    allowed_roots: set[str], *, group_roots: set[str] | None = None
) -> list[str]:
    """Forward explicit value overrides relevant to a child workflow.

    Config-group selections are reconstructed from Hydra's runtime choices.
    Leading ``+`` is removed because a key absent from the pipeline root may
    already exist in the child root (for example ``+eval.seed=7``).
    """
    forwarded = []
    for override in _runtime_task_overrides():
        root = _override_root(override)
        if root not in allowed_roots:
            continue
        key = override.lstrip("+~").split("=", maxsplit=1)[0].split("@", maxsplit=1)[0]
        if root in (group_roots or set()) and "." not in key:
            continue
        normalized = override.lstrip("+")
        forwarded.append(normalized)
    return forwarded


def _append_group_overrides(
    overrides: list[str],
    choices: dict[str, str],
    required_groups: list[str],
) -> None:
    for group in required_groups:
        selected = choices.get(group)
        if selected and selected not in ("null", "None"):
            overrides.append(f"{group}={selected}")


def _run_command(cmd: list[str], timeout_sec: int | None, print_command: bool) -> None:
    if print_command:
        logger.info("$ %s", " ".join(shlex.quote(part) for part in cmd))

    completed = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[3]),
        check=False,
        timeout=timeout_sec,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}")


def _run_stage_script(
    cfg: DictConfig,
    choices: dict[str, str],
    *,
    stage_name: str,
    script_name: str,
    required_groups: list[str],
    extra_overrides: list[str] | None = None,
    forwarded_roots: list[str] | None = None,
    enforced_overrides: list[str] | None = None,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / script_name
    logger.debug("Launching child workflow for stage %s", stage_name)

    overrides: list[str] = []
    _append_group_overrides(overrides, choices, required_groups)
    allowed_roots = set(required_groups) | set(forwarded_roots or [])
    overrides.extend(_forward_task_overrides(allowed_roots, group_roots=set(required_groups)))
    overrides.append(f"verbose={str(bool(cfg.get('verbose', False))).lower()}")
    overrides.extend(extra_overrides or [])
    # The DAG invokes a child only after deciding its output is stale/forced.
    # These final overrides prevent the child from immediately reusing it.
    overrides.extend(enforced_overrides or [])

    cmd = [str(cfg.pipeline.python_exe), str(script_path), *overrides]
    timeout = OmegaConf.select(cfg, "pipeline.stage_timeout_sec", default=None)
    _run_command(
        cmd,
        timeout_sec=int(timeout) if timeout not in (None, "null") else None,
        print_command=bool(cfg.pipeline.print_stage_commands),
    )


def _run_export(cfg: DictConfig, choices: dict[str, str]) -> None:
    paths = FoundationPaths.from_config(cfg)

    _run_stage_script(
        cfg,
        choices,
        stage_name="export_model",
        script_name="export_model.py",
        required_groups=[
            "platform",
            "runtime",
            "data",
            "encoder",
            "extraction",
            "splits",
            "model",
            "training",
            "experiment",
            "eval",
            "export",
        ],
        extra_overrides=[
            f"exp_name={paths.experiment_name}",
            f"train_dir={paths.train_dir}",
            f"eval_dir={paths.eval_dir}",
            f"export.artifact_dir={paths.artifact_dir}",
        ],
        forwarded_roots=["export", "exp_name", "train_dir", "eval_dir", "num_classes"],
    )


def _build_supervised_stage_runs(cfg: DictConfig, choices: dict[str, str]) -> dict:
    paths = FoundationPaths.from_config(cfg)
    return {
        "extract_features": lambda _cfg: _run_stage_script(
            cfg,
            choices,
            stage_name="extract_features",
            script_name="extract_features.py",
            required_groups=["platform", "runtime", "data", "encoder", "extraction"],
            enforced_overrides=["force=true"],
        ),
        "build_mmap": lambda _cfg: _run_stage_script(
            cfg,
            choices,
            stage_name="build_mmap",
            script_name="build_mmap.py",
            required_groups=[
                "platform",
                "runtime",
                "data",
                "encoder",
                "extraction",
                "storage",
            ],
            enforced_overrides=["force=true"],
        ),
        "split_data": lambda _cfg: _run_stage_script(
            cfg,
            choices,
            stage_name="split_data",
            script_name="make_splits.py",
            required_groups=[
                "platform",
                "runtime",
                "data",
                "encoder",
                "extraction",
                "splits",
            ],
            enforced_overrides=["force=true"],
        ),
        "train_model": lambda _cfg: _run_stage_script(
            cfg,
            choices,
            stage_name="train_model",
            script_name="train.py",
            required_groups=[
                "platform",
                "runtime",
                "data",
                "encoder",
                "extraction",
                "splits",
                "model",
                "training",
                "experiment",
            ],
            extra_overrides=[f"exp_name={paths.experiment_name}"],
            forwarded_roots=["wandb", "exp_name"],
            enforced_overrides=["training.force_rerun=true"],
        ),
        "evaluate": lambda _cfg: _run_stage_script(
            cfg,
            choices,
            stage_name="evaluate",
            script_name="evaluate.py",
            required_groups=[
                "platform",
                "runtime",
                "data",
                "encoder",
                "extraction",
                "splits",
                "model",
                "training",
                "experiment",
                "eval",
            ],
            extra_overrides=[
                f"exp_name={paths.experiment_name}",
                f"train_dir={paths.train_dir}",
                f"eval_dir={paths.eval_dir}",
            ],
            forwarded_roots=["eval", "wandb", "exp_name", "train_dir", "eval_dir"],
        ),
        "export_model": lambda _cfg: _run_export(cfg, choices),
    }


def run_pipeline_workflow(cfg: DictConfig) -> dict:
    """Run or inspect the essential foundation DAG."""
    setup_logging(cfg, stage="pipeline")
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    mode = str(cfg.pipeline.mode).lower()
    target = str(cfg.pipeline.target)
    choices = _runtime_choices()

    if mode == "supervised":
        stage_runs = _build_supervised_stage_runs(cfg, choices)
        runner = build_supervised_pipeline(cfg=cfg, stage_runs=stage_runs)
    else:
        raise ValueError(f"Unsupported pipeline.mode '{cfg.pipeline.mode}'")

    if target not in runner.stages():
        raise ValueError(f"Unknown pipeline.target '{target}'. Valid stages: {runner.stages()}")

    dag_text = runner.render_dag(
        target=target,
        cfg=cfg,
        include_fingerprint=bool(cfg.pipeline.include_fingerprint),
    )
    if bool(cfg.pipeline.show_dag):
        print(dag_text)

    dag_output = OmegaConf.select(cfg, "pipeline.dag_output_path", default=None)
    if dag_output not in (None, "null"):
        dag_path = Path(str(dag_output))
        dag_path.parent.mkdir(parents=True, exist_ok=True)
        dag_path.write_text(dag_text)
        logger.info("Mermaid DAG written: %s", dag_path)

    result = runner.execute(
        target=target,
        cfg=cfg,
        dry_run=bool(cfg.pipeline.dry_run),
        force=bool(cfg.pipeline.force),
    )

    logger.info(
        "Pipeline result: executed=%s skipped=%s errors=%s",
        result["executed"],
        result["skipped"],
        list(result["errors"].keys()),
    )
    logger.info("Pipeline fingerprint: %s", result["pipeline_fingerprint"])

    if result["errors"] and not bool(cfg.pipeline.dry_run):
        raise RuntimeError(f"Pipeline failed: {result['errors']}")
    return result
