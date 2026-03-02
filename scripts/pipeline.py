"""
Unified DAG runner for OceanPath pipelines.

Supports two modes:
  1. supervised: 7-stage train/eval/analyze/export pipeline
  2. pretrain:   4-stage SSL pretraining-only pipeline

Usage:
    # Supervised pipeline
    python scripts/pipeline.py pipeline_profile=supervised \
        platform=local data=gej encoder=univ1 model=abmil

    # SSL pretraining-only pipeline
    python scripts/pipeline.py pipeline_profile=pretrain_only \
        platform=local data=uni2h_pretrain encoder=uni2h model=abmil \
        pretrain_training=vicreg

    # Validate DAG and freshness only
    python scripts/pipeline.py ... pipeline.dry_run=true
"""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from oceanpath.pipeline import build_pretraining_pipeline, build_supervised_pipeline
from oceanpath.pipeline.transactions import atomic_output
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


def _write_pretrain_split_manifest(cfg: DictConfig) -> None:
    mmap_dir = Path(str(cfg.data.mmap_dir))
    index_path = mmap_dir / "index_arrays.npz"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing mmap index: {index_path}")

    idx = np.load(str(index_path), allow_pickle=True)
    slide_ids = sorted(str(x) for x in idx["slide_ids"].tolist())

    csv_path = OmegaConf.select(cfg, "pretrain_split.csv_path", default=None)
    filename_column = OmegaConf.select(cfg, "pretrain_split.filename_column", default="slide_id")

    if csv_path not in (None, "null"):
        csv_path_obj = Path(str(csv_path))
        if csv_path_obj.is_file():
            df = pd.read_csv(csv_path_obj)
            if filename_column in df.columns:
                csv_ids = {Path(str(v)).stem for v in df[filename_column].astype(str).tolist()}
            elif "slide_id" in df.columns:
                csv_ids = {Path(str(v)).stem for v in df["slide_id"].astype(str).tolist()}
            else:
                raise ValueError(
                    f"CSV {csv_path_obj} has no '{filename_column}' or 'slide_id' column"
                )
            pre_filter_n = len(slide_ids)
            slide_ids = sorted(set(slide_ids) & csv_ids)
            logger.info(
                "Pretrain split CSV filter: %d -> %d slides (%s)",
                pre_filter_n,
                len(slide_ids),
                csv_path_obj,
            )
        else:
            logger.warning("pretrain_split.csv_path does not exist: %s", csv_path_obj)

    if len(slide_ids) < 2:
        raise ValueError(
            f"Need at least 2 slides for train/val split, got {len(slide_ids)} after filtering"
        )

    val_frac = float(cfg.pretrain_split.val_frac)
    seed = int(cfg.pretrain_split.seed)
    n = len(slide_ids)
    n_val = min(max(1, int(n * val_frac)), n - 1)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    val_ids = [slide_ids[i] for i in val_idx]
    train_ids = [slide_ids[i] for i in train_idx]

    manifest_path = Path(str(cfg.pretrain_split.output_path))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "val_frac": val_frac,
        "seed": seed,
        "n_total": n,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "mmap_dir": str(mmap_dir),
        "csv_path": None if csv_path in (None, "null") else str(csv_path),
        "filename_column": filename_column,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info(
        "Pretrain split manifest written: %s (%d train / %d val)",
        manifest_path,
        len(train_ids),
        len(val_ids),
    )


def _run_pretrain_export(cfg: DictConfig) -> None:
    pretrain_dir = Path(str(cfg.output_dir))
    if not pretrain_dir.is_dir():
        raise FileNotFoundError(f"Pretrain output directory not found: {pretrain_dir}")

    artifact_dir = _resolve_artifact_dir(cfg)

    metadata_path = pretrain_dir / "metadata.json"
    config_path = pretrain_dir / "config.yaml"
    agg_path = pretrain_dir / "aggregator_weights.pt"
    split_manifest_path = Path(str(cfg.pretrain_split.output_path))

    best_ckpt: Path | None = None
    if metadata_path.is_file():
        try:
            meta = json.loads(metadata_path.read_text())
            candidate = meta.get("best_checkpoint")
            if candidate:
                p = Path(str(candidate))
                if p.is_file():
                    best_ckpt = p
        except Exception:
            pass

    if best_ckpt is None:
        ckpt_dir = pretrain_dir / "checkpoints"
        if ckpt_dir.is_dir():
            best_candidates = sorted(ckpt_dir.glob("best-*.ckpt"))
            if best_candidates:
                best_ckpt = best_candidates[-1]
            else:
                any_ckpts = sorted(ckpt_dir.glob("*.ckpt"))
                if any_ckpts:
                    best_ckpt = any_ckpts[-1]

    with atomic_output(artifact_dir) as tmp_dir:
        if config_path.is_file():
            shutil.copy2(config_path, tmp_dir / "pretrain_config.yaml")
        if metadata_path.is_file():
            shutil.copy2(metadata_path, tmp_dir / "metadata.json")
        if split_manifest_path.is_file():
            shutil.copy2(split_manifest_path, tmp_dir / "split_manifest.json")
        if agg_path.is_file():
            shutil.copy2(agg_path, tmp_dir / "aggregator_weights.pt")
        if best_ckpt is not None:
            shutil.copy2(best_ckpt, tmp_dir / "best.ckpt")

        export_info = {
            "type": "ssl_pretrain_artifact",
            "exp_name": str(cfg.exp_name),
            "pretrain_dir": str(pretrain_dir),
            "best_checkpoint": str(best_ckpt) if best_ckpt is not None else None,
            "aggregator_weights": str(agg_path) if agg_path.is_file() else None,
            "recommended_finetune_override": (
                "training.aggregator_weights_path=aggregator_weights.pt"
                if agg_path.is_file()
                else None
            ),
        }
        (tmp_dir / "artifact_manifest.json").write_text(
            json.dumps(export_info, indent=2, sort_keys=True)
        )

    logger.info("Pretrain artifact exported: %s", artifact_dir)

    if bool(cfg.pipeline.run_serve):
        logger.warning(
            "pipeline.run_serve=true was requested, but SSL pretrain artifacts are "
            "packaged weights (not deployable classification endpoints). Skipping serve stage."
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


def _build_pretrain_stage_runs(cfg: DictConfig, choices: dict[str, str]) -> dict:
    return {
        "build_mmap": lambda _cfg: _run_stage_script(
            cfg,
            choices,
            stage_name="build_mmap",
            script_name="build_mmap.py",
            required_groups=["platform", "data", "encoder", "extraction", "storage"],
        ),
        "split_data": lambda _cfg: _write_pretrain_split_manifest(cfg),
        "pretrain_model": lambda _cfg: _run_stage_script(
            cfg,
            choices,
            stage_name="pretrain_model",
            script_name="pretrain.py",
            required_groups=["platform", "data", "encoder", "model", "pretrain_training"],
            extra_overrides=[
                f"exp_name={cfg.exp_name}",
                f"output_dir={cfg.output_dir}",
                f"pretrain_split_manifest={cfg.pretrain_split.output_path}",
            ],
        ),
        "export_and_serve": lambda _cfg: _run_pretrain_export(cfg),
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
    elif mode == "pretrain":
        stage_runs = _build_pretrain_stage_runs(cfg, choices)
        runner = build_pretraining_pipeline(cfg=cfg, stage_runs=stage_runs)
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
