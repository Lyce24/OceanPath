"""
DAG-based pipeline orchestration with stage fingerprints.

Features:
- Explicit + implicit stage dependencies
- Make-like freshness checks (mtime-based)
- Fingerprint-aware freshness (config-sensitive skip/run decisions)
- Mermaid DAG rendering
- Factory builders for supervised and SSL-pretraining pipelines
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from oceanpath.pipeline.transactions import transaction_matches, write_stage_transaction
from oceanpath.utils.repro import config_fingerprint

logger = logging.getLogger(__name__)


def _cfg_select(cfg, key: str, default=None):
    """Safe dotted-key config selection for DictConfig/dict inputs."""
    try:
        from omegaconf import OmegaConf

        return OmegaConf.select(cfg, key, default=default)
    except Exception:
        if not isinstance(cfg, dict):
            return default
        cur = cfg
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def _cfg_to_plain(value):
    """Convert config values to plain Python for stable hashing."""
    try:
        from omegaconf import OmegaConf

        if hasattr(value, "_metadata"):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value


def _as_path(value, default: str | Path | None = None) -> Path:
    if value is None:
        if default is None:
            raise ValueError("Path value is None and no default provided")
        return Path(default)
    return Path(str(value))


def _artifact_dir_from_cfg(cfg, exp_name: str) -> Path:
    explicit = _cfg_select(cfg, "export.artifact_dir", default=None)
    if explicit not in (None, "null"):
        return Path(str(explicit))

    artifact_root = _cfg_select(cfg, "export.artifact_root", default=None)
    if artifact_root is None:
        output_root = _cfg_select(cfg, "platform.output_root", default="outputs")
        artifact_root = Path(output_root) / "artifacts"
    else:
        artifact_root = Path(str(artifact_root))

    return artifact_root / f"{exp_name}_{config_fingerprint(cfg)}"


@dataclass
class Stage:
    """
    A pipeline stage with I/O contracts and optional run function.
    """

    name: str
    inputs: list[Path] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    config_keys: list[str] = field(default_factory=list)
    run: Callable | None = None
    depends_on: list[str] = field(default_factory=list)
    validator: Callable | None = None
    description: str = ""


class PipelineRunner:
    """
    DAG-based runner with freshness checks and fingerprint-aware transactions.
    """

    def __init__(self):
        self._stages: dict[str, Stage] = {}

    def register(self, stage: Stage) -> PipelineRunner:
        if stage.name in self._stages:
            raise ValueError(f"Stage '{stage.name}' already registered")
        self._stages[stage.name] = stage
        return self

    def stages(self) -> list[str]:
        return list(self._stages.keys())

    # ── Fingerprints ─────────────────────────────────────────────────────

    def stage_fingerprint(self, stage_name: str, cfg=None) -> str:
        if stage_name not in self._stages:
            raise ValueError(f"Unknown stage: '{stage_name}'")
        return self._compute_stage_fingerprint(self._stages[stage_name], cfg=cfg)

    def _compute_stage_fingerprint(self, stage: Stage, cfg=None) -> str:
        cfg_payload = {}
        if cfg is not None:
            for key in stage.config_keys:
                cfg_payload[key] = _cfg_to_plain(_cfg_select(cfg, key, default="__MISSING__"))

        payload = {
            "stage": stage.name,
            "depends_on": sorted(stage.depends_on),
            "inputs": sorted(str(Path(p)) for p in stage.inputs),
            "outputs": sorted(str(Path(p)) for p in stage.outputs),
            "config_keys": sorted(stage.config_keys),
            "config": cfg_payload,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def pipeline_fingerprint(self, target: str, cfg=None) -> str:
        order = self._topological_sort(target)
        stage_fps = [self._compute_stage_fingerprint(self._stages[name], cfg=cfg) for name in order]
        canonical = json.dumps(
            {"target": target, "order": order, "stage_fingerprints": stage_fps},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    # ── Dependency resolution ────────────────────────────────────────────

    def _build_dependency_graph(self) -> dict[str, list[str]]:
        graph = {name: list(stage.depends_on) for name, stage in self._stages.items()}

        output_owners: dict[str, str] = {}
        for name, stage in self._stages.items():
            for out_path in stage.outputs:
                output_owners[str(Path(out_path))] = name

        for name, stage in self._stages.items():
            for in_path in stage.inputs:
                owner = output_owners.get(str(Path(in_path)))
                if owner and owner != name and owner not in graph[name]:
                    graph[name].append(owner)

        return graph

    def _topological_sort(self, target: str) -> list[str]:
        graph = self._build_dependency_graph()

        needed = set()
        stack = [target]
        while stack:
            node = stack.pop()
            if node not in self._stages:
                raise ValueError(f"Unknown stage: '{node}'")
            if node in needed:
                continue
            needed.add(node)
            for dep in graph.get(node, []):
                stack.append(dep)

        sub_graph = {n: [d for d in graph[n] if d in needed] for n in needed}
        in_degree = {n: len(sub_graph[n]) for n in needed}

        queue = sorted([n for n in needed if in_degree[n] == 0])
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for n in needed:
                if node in sub_graph[n]:
                    in_degree[n] -= 1
                    if in_degree[n] == 0:
                        queue.append(n)
            queue.sort()

        if len(order) != len(needed):
            remaining = needed - set(order)
            raise RuntimeError(f"Cycle detected in pipeline DAG involving: {remaining}")

        return order

    # ── Freshness checks ────────────────────────────────────────────────

    def _outputs_are_fresh(self, stage: Stage, cfg=None) -> bool:
        """
        Outputs are fresh iff:
        - all outputs exist
        - output transaction fingerprints match (when cfg is provided)
        - outputs are newer than all existing inputs
        """
        if not stage.outputs:
            return False

        output_paths = [Path(p) for p in stage.outputs]
        if not all(p.exists() for p in output_paths):
            return False

        if cfg is not None:
            stage_fp = self._compute_stage_fingerprint(stage, cfg=cfg)
            for out_path in output_paths:
                if not transaction_matches(
                    out_path, stage_name=stage.name, stage_fingerprint=stage_fp
                ):
                    return False

        if not stage.inputs:
            return True

        input_mtimes = []
        for in_path in stage.inputs:
            p = Path(in_path)
            if p.exists():
                input_mtimes.append(p.stat().st_mtime)

        if not input_mtimes:
            return True

        newest_input = max(input_mtimes)
        oldest_output = min(p.stat().st_mtime for p in output_paths)
        return oldest_output >= newest_input

    # ── Execution ───────────────────────────────────────────────────────

    def execute(
        self,
        target: str,
        cfg=None,
        dry_run: bool = False,
        force: bool = False,
    ) -> dict:
        order = self._topological_sort(target)

        result = {
            "target": target,
            "order": order,
            "executed": [],
            "skipped": [],
            "timings": {},
            "errors": {},
            "dry_run": dry_run,
            "stage_fingerprints": {},
            "pipeline_fingerprint": self.pipeline_fingerprint(target, cfg=cfg),
        }

        logger.info(f"Pipeline: {' -> '.join(order)} (target={target})")

        if dry_run:
            return self._dry_run(order, cfg, result)

        for stage_name in order:
            stage = self._stages[stage_name]
            stage_fp = self._compute_stage_fingerprint(stage, cfg=cfg)
            result["stage_fingerprints"][stage_name] = stage_fp

            if not force and self._outputs_are_fresh(stage, cfg=cfg):
                logger.info(f"  [{stage_name}] SKIP (outputs fresh)")
                result["skipped"].append(stage_name)
                continue

            missing = [str(p) for p in stage.inputs if not Path(p).exists()]
            if missing:
                msg = f"Missing inputs for '{stage_name}': {missing}"
                logger.error(msg)
                result["errors"][stage_name] = msg
                break

            if stage.run is None:
                logger.warning(f"  [{stage_name}] No run function registered")
                result["skipped"].append(stage_name)
                continue

            logger.info(f"  [{stage_name}] RUNNING...")
            t0 = time.monotonic()

            try:
                stage.run(cfg)
                elapsed = time.monotonic() - t0
                result["executed"].append(stage_name)
                result["timings"][stage_name] = round(elapsed, 1)
                logger.info(f"  [{stage_name}] DONE ({elapsed:.1f}s)")

                if stage.validator:
                    for out_path in stage.outputs:
                        stage.validator(Path(out_path))

                for out_path in stage.outputs:
                    write_stage_transaction(
                        out_path,
                        stage_name=stage.name,
                        stage_fingerprint=stage_fp,
                        inputs=[str(Path(p)) for p in stage.inputs],
                        config_keys=list(stage.config_keys),
                        extra={"target": target},
                    )

            except Exception as e:
                elapsed = time.monotonic() - t0
                result["errors"][stage_name] = str(e)
                result["timings"][stage_name] = round(elapsed, 1)
                logger.error(f"  [{stage_name}] FAILED ({elapsed:.1f}s): {e}")
                break

        return result

    def _dry_run(self, order: list[str], cfg, result: dict) -> dict:
        logger.info("DRY RUN — validating pipeline")

        for stage_name in order:
            stage = self._stages[stage_name]
            issues = []
            stage_fp = self._compute_stage_fingerprint(stage, cfg=cfg)
            result["stage_fingerprints"][stage_name] = stage_fp

            for in_path in stage.inputs:
                if not Path(in_path).exists():
                    issues.append(f"missing input: {in_path}")

            all_outputs_exist = all(Path(p).exists() for p in stage.outputs)
            fresh = self._outputs_are_fresh(stage, cfg=cfg)

            if issues:
                result["errors"][stage_name] = issues
                status = "ISSUES"
            elif fresh:
                status = "FRESH"
            elif all_outputs_exist:
                status = "STALE (fingerprint/mtime)"
            else:
                status = "STALE (missing outputs)"

            logger.info(f"  [{stage_name}] {status} fp={stage_fp}")

        return result

    # ── DAG rendering ───────────────────────────────────────────────────

    def render_dag(
        self,
        target: str | None = None,
        cfg=None,
        include_fingerprint: bool = True,
    ) -> str:
        """
        Return a Mermaid DAG definition.

        Example:
            graph TD
                extract["extract\\nfp:abc123"]
                build_mmap["build_mmap\\nfp:def456"]
                extract --> build_mmap
        """
        graph = self._build_dependency_graph()
        stage_names = self._topological_sort(target) if target else self.stages()
        stage_set = set(stage_names)

        def _node_id(name: str) -> str:
            return "stage_" + re.sub(r"[^0-9A-Za-z_]", "_", name)

        lines = ["graph TD"]
        for name in stage_names:
            stage = self._stages[name]
            label = stage.name
            if stage.description:
                label += f"\\n{stage.description}"
            if include_fingerprint:
                fp = self._compute_stage_fingerprint(stage, cfg=cfg)
                label += f"\\nfp:{fp}"
            label = label.replace('"', "'")
            lines.append(f'    {_node_id(name)}["{label}"]')

        for name in stage_names:
            for dep in sorted(graph.get(name, [])):
                if dep in stage_set:
                    lines.append(f"    {_node_id(dep)} --> {_node_id(name)}")

        return "\n".join(lines)

    def print_dag(self, target: str | None = None, cfg=None) -> None:
        graph = self._build_dependency_graph()
        stage_names = self._topological_sort(target) if target else self.stages()
        print("\nPipeline DAG:")
        print("-" * 72)
        for name in stage_names:
            stage = self._stages[name]
            deps = graph.get(name, [])
            dep_str = f" <- {', '.join(sorted(deps))}" if deps else ""
            fresh = self._outputs_are_fresh(stage, cfg=cfg)
            status = "✓" if fresh else "○"
            fp = self._compute_stage_fingerprint(stage, cfg=cfg)
            print(f"  {status} {name}{dep_str} [fp:{fp}]")
        print()


def _stage_run(stage_name: str, stage_runs: dict[str, Callable] | None) -> Callable | None:
    if not stage_runs:
        return None
    return stage_runs.get(stage_name)


def _resolve_supervised_paths(cfg) -> dict[str, Path]:
    exp_name = _cfg_select(cfg, "exp_name", default="default")
    data_name = _cfg_select(cfg, "data.name", default="data")
    encoder_name = _cfg_select(cfg, "encoder.name", default="encoder")
    splits_name = _cfg_select(cfg, "splits.name", default="splits")
    platform = _cfg_select(cfg, "platform", default={}) or {}

    slide_dir = _as_path(
        _cfg_select(
            cfg, "data.slide_dir", default=_cfg_select(cfg, "platform.slide_root", default="slides")
        )
    )
    feature_h5_dir = _cfg_select(cfg, "data.feature_h5_dir", default=None)
    if feature_h5_dir is None:
        feature_root = _cfg_select(
            cfg,
            "platform.feature_root",
            default=_cfg_select(cfg, "platform.features_root", default="features"),
        )
        features_subdir = _cfg_select(
            cfg, "encoder.features_subdir", default=f"features_{encoder_name}"
        )
        feature_h5_dir = Path(feature_root) / data_name / str(features_subdir)
    feature_h5_dir = _as_path(feature_h5_dir)

    mmap_dir = _as_path(
        _cfg_select(
            cfg,
            "data.mmap_dir",
            default=Path(_cfg_select(cfg, "platform.mmap_root", default="mmap"))
            / data_name
            / encoder_name,
        )
    )

    splits_dir = _cfg_select(cfg, "splits.output_dir", default=None)
    if splits_dir is None:
        splits_root = _cfg_select(cfg, "platform.splits_root", default="splits")
        splits_dir = Path(splits_root) / data_name / splits_name
    splits_dir = _as_path(splits_dir)

    output_root = _cfg_select(cfg, "platform.output_root", default="outputs")
    train_dir = _as_path(
        _cfg_select(cfg, "train_dir", default=Path(output_root) / "train" / exp_name)
    )
    eval_dir = _as_path(_cfg_select(cfg, "eval_dir", default=train_dir / "eval"))
    analysis_dir = _as_path(
        _cfg_select(cfg, "output_dir", default=Path(output_root) / "analyze" / exp_name)
    )
    artifact_dir = _artifact_dir_from_cfg(cfg, exp_name)

    return {
        "slide_dir": slide_dir,
        "feature_h5_dir": feature_h5_dir,
        "mmap_dir": mmap_dir,
        "splits_dir": splits_dir,
        "train_dir": train_dir,
        "eval_dir": eval_dir,
        "analysis_dir": analysis_dir,
        "artifact_dir": artifact_dir,
        "csv_path": _as_path(
            _cfg_select(
                cfg, "data.csv_path", default=Path(platform.get("project_root", ".")) / "manifests"
            )
        ),
    }


def build_supervised_pipeline(
    cfg,
    stage_runs: dict[str, Callable] | None = None,
) -> PipelineRunner:
    """
    Build the 7-stage supervised pipeline DAG.

    Stages:
      1. feature extraction
      2. build mmap
      3. split data
      4. train model
      5. evaluate
      6. analysis
      7. export + serve
    """
    p = _resolve_supervised_paths(cfg)
    runner = PipelineRunner()

    runner.register(
        Stage(
            name="extract_features",
            description="Stage 1: feature extraction",
            inputs=[p["slide_dir"]],
            outputs=[p["feature_h5_dir"]],
            config_keys=["platform", "data", "encoder", "extraction"],
            run=_stage_run("extract_features", stage_runs),
        )
    )
    runner.register(
        Stage(
            name="build_mmap",
            description="Stage 2: build mmap",
            inputs=[p["feature_h5_dir"]],
            outputs=[p["mmap_dir"]],
            config_keys=["platform", "data", "encoder", "storage"],
            depends_on=["extract_features"],
            run=_stage_run("build_mmap", stage_runs),
        )
    )
    runner.register(
        Stage(
            name="split_data",
            description="Stage 3: split data",
            inputs=[p["csv_path"]],
            outputs=[p["splits_dir"]],
            config_keys=["platform", "data", "splits"],
            depends_on=["build_mmap"],
            run=_stage_run("split_data", stage_runs),
        )
    )
    runner.register(
        Stage(
            name="train_model",
            description="Stage 4: train model (model/datamodule/optional preload weights)",
            inputs=[p["mmap_dir"], p["splits_dir"]],
            outputs=[p["train_dir"]],
            config_keys=["platform", "data", "splits", "model", "training"],
            depends_on=["split_data"],
            run=_stage_run("train_model", stage_runs),
        )
    )
    runner.register(
        Stage(
            name="evaluate",
            description="Stage 5: evaluate",
            inputs=[p["train_dir"]],
            outputs=[p["eval_dir"]],
            config_keys=["platform", "eval"],
            depends_on=["train_model"],
            run=_stage_run("evaluate", stage_runs),
        )
    )
    runner.register(
        Stage(
            name="analyze",
            description="Stage 6: analysis",
            inputs=[p["train_dir"], p["eval_dir"]],
            outputs=[p["analysis_dir"]],
            config_keys=["platform", "analysis"],
            depends_on=["evaluate"],
            run=_stage_run("analyze", stage_runs),
        )
    )
    runner.register(
        Stage(
            name="export_and_serve",
            description="Stage 7: export and serve",
            inputs=[p["train_dir"], p["eval_dir"], p["analysis_dir"]],
            outputs=[p["artifact_dir"]],
            config_keys=["platform", "export", "serve"],
            depends_on=["analyze"],
            run=_stage_run("export_and_serve", stage_runs),
        )
    )

    return runner


def build_pretraining_pipeline(
    cfg,
    stage_runs: dict[str, Callable] | None = None,
) -> PipelineRunner:
    """
    Build the 4-stage SSL pretraining-only pipeline DAG.

    Stages:
      1. build mmap (includes data download/prep before mmap)
      2. split data (train/val)
      3. pretrain model (SSL structures/loss/callbacks/datamodule)
      4. export + serve
    """
    exp_name = _cfg_select(cfg, "exp_name", default="pretrain_default")
    output_root = _cfg_select(cfg, "platform.output_root", default="outputs")

    mmap_dir = _as_path(
        _cfg_select(
            cfg,
            "data.mmap_dir",
            default=Path(_cfg_select(cfg, "platform.mmap_root", default="mmap"))
            / _cfg_select(cfg, "data.name", default="data")
            / _cfg_select(cfg, "encoder.name", default="encoder"),
        )
    )
    pretrain_dir = _as_path(
        _cfg_select(cfg, "output_dir", default=Path(output_root) / "pretrain" / exp_name)
    )
    split_manifest = pretrain_dir / "split_manifest.json"
    artifact_dir = _artifact_dir_from_cfg(cfg, exp_name)

    runner = PipelineRunner()

    runner.register(
        Stage(
            name="build_mmap",
            description="Stage 1: build mmap (data download/prep included)",
            inputs=[],
            outputs=[mmap_dir],
            config_keys=["platform", "data", "encoder", "storage"],
            run=_stage_run("build_mmap", stage_runs),
        )
    )
    runner.register(
        Stage(
            name="split_data",
            description="Stage 2: split train/val for pretraining",
            inputs=[mmap_dir],
            outputs=[split_manifest],
            config_keys=["pretrain_training", "data"],
            depends_on=["build_mmap"],
            run=_stage_run("split_data", stage_runs),
        )
    )
    runner.register(
        Stage(
            name="pretrain_model",
            description="Stage 3: SSL pretrain (model/loss/callbacks/datamodule)",
            inputs=[mmap_dir, split_manifest],
            outputs=[pretrain_dir],
            config_keys=["platform", "data", "encoder", "model", "pretrain", "pretrain_training"],
            depends_on=["split_data"],
            run=_stage_run("pretrain_model", stage_runs),
        )
    )
    runner.register(
        Stage(
            name="export_and_serve",
            description="Stage 4: export and serve",
            inputs=[pretrain_dir],
            outputs=[artifact_dir],
            config_keys=["platform", "export", "serve"],
            depends_on=["pretrain_model"],
            run=_stage_run("export_and_serve", stage_runs),
        )
    )

    return runner


def build_oceanpath_pipeline(
    cfg,
    stage_runs: dict[str, Callable] | None = None,
) -> PipelineRunner:
    """Backward-compatible alias for the standard supervised pipeline."""
    return build_supervised_pipeline(cfg=cfg, stage_runs=stage_runs)
