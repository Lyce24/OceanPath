"""
Unified DAG runner for the OceanPath supervised pipeline.

Runs the 7-stage train/eval/analyze/export pipeline.

Usage:
    # Supervised pipeline
    python scripts/pipeline.py pipeline_profile=supervised \
        platform=local data=gej encoder=univ1 model=abmil

    # Validate DAG and freshness only
    python scripts/pipeline.py ... pipeline.dry_run=true
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from oceanpath.pipeline import build_supervised_pipeline
from oceanpath.utils.repro import config_fingerprint

logger = logging.getLogger(__name__)


def _setup_logging(cfg: DictConfig) -> None:
    level = logging.DEBUG if cfg.get("verbose", False) else logging.INFO
    exp_name = OmegaConf.select(cfg, "exp_name", default="pipeline")
    mode = OmegaConf.select(cfg, "pipeline.mode", default="unknown")
    fmt = f"%(asctime)s | %(levelname)-7s | mode={mode} | exp={exp_name} | %(message)s"
    logging.basicConfig(level=level, format=fmt, force=True)


def _runtime_choices() -> dict[str, str]:
    runtime = HydraConfig.get().runtime
    choices = {}
    for key, val in runtime.choices.items():
        if isinstance(val, str):
            choices[key] = val
    return choices


def _resolve_artifact_dir(cfg: DictConfig) -> Path:
    explicit = OmegaConf.select(cfg, "export.artifact_dir", default=None)
    if explicit not in (None, "null"):
        return Path(str(explicit))

    artifact_root = OmegaConf.select(cfg, "export.artifact_root", default=None)
    if artifact_root is None:
        artifact_root = Path(cfg.platform.output_root) / "artifacts"
    else:
        artifact_root = Path(str(artifact_root))

    return artifact_root / f"{cfg.exp_name}_{config_fingerprint(cfg)}"


def _append_group_overrides(
    overrides: list[str],
    choices: dict[str, str],
    required_groups: list[str],
) -> None:
    for group in required_groups:
        selected = choices.get(group)
        if selected and selected not in ("null", "None"):
            overrides.append(f"{group}={selected}")


def _stage_override_list(cfg: DictConfig, stage_name: str) -> list[str]:
    extra = OmegaConf.select(cfg, f"stage_overrides.{stage_name}", default=[])
    if extra is None:
        return []
    return [str(x) for x in extra]


def _run_command(cmd: list[str], timeout_sec: int | None, print_command: bool) -> None:
    if print_command:
        logger.info("$ %s", " ".join(shlex.quote(part) for part in cmd))

    completed = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[1]),
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
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / script_name

    overrides: list[str] = []
    _append_group_overrides(overrides, choices, required_groups)
    overrides.append(f"verbose={str(bool(cfg.get('verbose', False))).lower()}")
    overrides.extend(extra_overrides or [])
    overrides.extend(_stage_override_list(cfg, stage_name))

    cmd = [str(cfg.pipeline.python_exe), str(script_path), *overrides]
    timeout = OmegaConf.select(cfg, "pipeline.stage_timeout_sec", default=None)
    _run_command(
        cmd,
        timeout_sec=int(timeout) if timeout not in (None, "null") else None,
        print_command=bool(cfg.pipeline.print_stage_commands),
    )


def _run_supervised_export_and_serve(cfg: DictConfig, choices: dict[str, str]) -> None:
    artifact_dir = _resolve_artifact_dir(cfg)

    _run_stage_script(
        cfg,
        choices,
        stage_name="export_and_serve",
        script_name="export_model.py",
        required_groups=[
            "platform",
            "data",
            "encoder",
            "extraction",
            "splits",
            "model",
            "training",
        ],
        extra_overrides=[
            f"exp_name={cfg.exp_name}",
            f"train_dir={cfg.train_dir}",
            f"eval_dir={cfg.eval_dir}",
            f"export.artifact_dir={artifact_dir}",
        ],
    )

    if bool(cfg.pipeline.run_serve):
        _run_stage_script(
            cfg,
            choices,
            stage_name="export_and_serve",
            script_name="serve.py",
            required_groups=[
                "platform",
                "data",
                "encoder",
                "extraction",
                "splits",
                "model",
                "training",
            ],
            extra_overrides=[
                f"exp_name={cfg.exp_name}",
                f"export.artifact_dir={artifact_dir}",
            ],
        )


def _build_supervised_stage_runs(cfg: DictConfig, choices: dict[str, str]) -> dict:
    return {
        "extract_features": lambda _cfg: _run_stage_script(
            cfg,
            choices,
            stage_name="extract_features",
            script_name="extract_features.py",
            required_groups=["platform", "data", "encoder", "extraction"],
        ),
        "build_mmap": lambda _cfg: _run_stage_script(
            cfg,
            choices,
            stage_name="build_mmap",
            script_name="build_mmap.py",
            required_groups=["platform", "data", "encoder", "extraction", "storage"],
        ),
        "split_data": lambda _cfg: _run_stage_script(
            cfg,
            choices,
            stage_name="split_data",
            script_name="make_splits.py",
            required_groups=["platform", "data", "encoder", "extraction", "splits"],
        ),
        "train_model": lambda _cfg: _run_stage_script(
            cfg,
            choices,
            stage_name="train_model",
            script_name="train.py",
            required_groups=[
                "platform",
                "data",
                "encoder",
                "extraction",
                "splits",
                "model",
                "training",
            ],
            extra_overrides=[f"exp_name={cfg.exp_name}"],
        ),
        "evaluate": lambda _cfg: _run_stage_script(
            cfg,
            choices,
            stage_name="evaluate",
            script_name="evaluate.py",
            required_groups=[
                "platform",
                "data",
                "encoder",
                "extraction",
                "splits",
                "model",
                "training",
            ],
            extra_overrides=[
                f"exp_name={cfg.exp_name}",
                f"train_dir={cfg.train_dir}",
                f"eval_dir={cfg.eval_dir}",
            ],
        ),
        "analyze": lambda _cfg: _run_stage_script(
            cfg,
            choices,
            stage_name="analyze",
            script_name="analyze.py",
            required_groups=[
                "platform",
                "data",
                "encoder",
                "extraction",
                "splits",
                "model",
                "training",
            ],
            extra_overrides=[
                f"exp_name={cfg.exp_name}",
                f"train_dir={cfg.train_dir}",
                f"eval_dir={cfg.eval_dir}",
                f"output_dir={cfg.output_dir}",
            ],
        ),
        "export_and_serve": lambda _cfg: _run_supervised_export_and_serve(cfg, choices),
    }


@hydra.main(config_path="../configs", config_name="pipeline", version_base="1.3")
def main(cfg: DictConfig) -> None:
    _setup_logging(cfg)
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    mode = str(cfg.pipeline.mode).lower()
    target = str(cfg.pipeline.target)
    choices = _runtime_choices()

    if mode == "supervised":
        stage_runs = _build_supervised_stage_runs(cfg, choices)
        runner = build_supervised_pipeline(cfg=cfg, stage_runs=stage_runs)
    else:
        raise ValueError(f"Unsupported pipeline.mode '{cfg.pipeline.mode}'")

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

    if result["errors"]:
        raise RuntimeError(f"Pipeline failed: {result['errors']}")


if __name__ == "__main__":
    main()
