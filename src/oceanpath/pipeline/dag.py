"""
DAG-based pipeline orchestration with stage fingerprints.

Features:
- Explicit + implicit stage dependencies
- Make-like freshness checks (mtime-based)
- Fingerprint-aware freshness (config-sensitive skip/run decisions)
- Mermaid DAG rendering
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from oceanpath.pipeline.transactions import (
    stage_transaction_mtime,
    transaction_matches,
    write_stage_transaction,
)

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


def _path_digest(path: Path) -> str:
    """Hash a source file or a directory tree in stable relative-path order."""
    if not path.exists():
        return "__MISSING__"

    digest = hashlib.sha256()
    if path.is_file():
        files = [path]
    else:
        files = sorted(
            item
            for item in path.rglob("*")
            if item.is_file()
            and "__pycache__" not in item.parts
            and not any(part.startswith(".") for part in item.relative_to(path).parts)
        )
    for file_path in files:
        relative = file_path.name if path.is_file() else file_path.relative_to(path).as_posix()
        digest.update(relative.encode())
        with file_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    """Return whether ``path`` is ``parent`` or one of its descendants."""
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def _latest_mtime(path: Path, *, exclude: list[Path] | None = None) -> float:
    """Return the newest mtime in a file or directory tree.

    Directory mtimes do not change when an existing nested file is rewritten,
    so freshness checks must inspect descendants. Nested stage outputs are
    excluded to avoid treating a stage's own output as a changed input.
    """
    excluded = exclude or []

    def _excluded(candidate: Path) -> bool:
        return any(_is_within(candidate, root) for root in excluded)

    if _excluded(path):
        raise ValueError(f"Cannot inspect excluded path: {path}")
    if path.is_file():
        return path.stat().st_mtime

    mtimes = [path.stat().st_mtime]
    for child in path.rglob("*"):
        if _excluded(child):
            continue
        try:
            mtimes.append(child.stat().st_mtime)
        except FileNotFoundError:
            # Concurrent writers can replace temporary paths while inspecting.
            continue
    return max(mtimes)


@dataclass
class Stage:
    """
    A pipeline stage with I/O contracts and optional run function.
    """

    name: str
    inputs: list[Path] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    config_keys: list[str] = field(default_factory=list)
    code_paths: list[Path] = field(default_factory=list)
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
            "code": {
                str(path): _path_digest(Path(path)) for path in sorted(stage.code_paths, key=str)
            },
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
        - output validators accept the current artifacts
        - output transaction fingerprints match (when cfg is provided)
        - the stage commit is newer than every declared input tree
        """
        if not stage.outputs:
            return False

        output_paths = [Path(p) for p in stage.outputs]
        if not all(p.exists() for p in output_paths):
            return False

        if stage.validator is not None:
            try:
                for out_path in output_paths:
                    stage.validator(out_path)
            except Exception as exc:
                logger.warning("Output validation failed for stage '%s': %s", stage.name, exc)
                return False

        transaction_mtimes: list[float] = []

        if cfg is not None:
            stage_fp = self._compute_stage_fingerprint(stage, cfg=cfg)
            for out_path in output_paths:
                if not transaction_matches(
                    out_path, stage_name=stage.name, stage_fingerprint=stage_fp
                ):
                    return False
                committed_at = stage_transaction_mtime(out_path)
                if committed_at is None:
                    return False
                transaction_mtimes.append(committed_at)

        if not stage.inputs:
            return True

        if any(not Path(in_path).exists() for in_path in stage.inputs):
            return False

        input_mtimes = []
        for in_path in stage.inputs:
            p = Path(in_path)
            input_mtimes.append(_latest_mtime(p, exclude=output_paths))

        newest_input = max(input_mtimes)
        oldest_output = (
            min(transaction_mtimes)
            if transaction_mtimes
            else min(_latest_mtime(p) for p in output_paths)
        )
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

        result: dict[str, Any] = {
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

                missing_outputs = [str(path) for path in stage.outputs if not Path(path).exists()]
                if missing_outputs:
                    raise FileNotFoundError(
                        f"Stage '{stage_name}' did not produce declared outputs: {missing_outputs}"
                    )

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

                elapsed = time.monotonic() - t0
                result["executed"].append(stage_name)
                result["timings"][stage_name] = round(elapsed, 1)
                logger.info(f"  [{stage_name}] DONE ({elapsed:.1f}s)")

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
            for issue in issues:
                logger.info("    - %s", issue)

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
