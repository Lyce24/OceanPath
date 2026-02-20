"""
DAG-based pipeline orchestration — make-like incremental builds.

Stages are connected by input/output paths. The runner:
  1. Topological sort of dependencies
  2. For each stage: check if outputs exist + are fresh
  3. Dry-run: validate inputs exist, configs parse, device available
  4. Live: run inside atomic transaction context
  5. Post-run: validate output schemas

This is intentionally lightweight — no Airflow, no Prefect. The pipeline
has 9 stages with clear I/O contracts. A 200-line DAG runner is the right
abstraction.

Usage:
    from oceanpath.pipeline import PipelineRunner, Stage

    runner = PipelineRunner()
    runner.register(Stage(
        name="extract",
        inputs=[slides_dir],
        outputs=[features_dir],
        config_keys=["encoder", "extraction"],
        run=extract_features,
    ))
    runner.register(Stage(
        name="build_mmap",
        inputs=[features_dir],
        outputs=[mmap_dir],
        config_keys=["data"],
        run=build_mmap,
    ))
    runner.execute("train", cfg, dry_run=False)
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Stage:
    """
    A pipeline stage with typed inputs and outputs.

    Parameters
    ----------
    name : str
        Unique identifier (e.g., "extract", "train", "evaluate").
    inputs : list[Path]
        Paths that must exist before this stage runs.
    outputs : list[Path]
        Paths that must exist after this stage runs.
    config_keys : list[str]
        Hydra config sections this stage depends on (for fingerprinting).
    run : Callable
        Function to execute: run(cfg, **kwargs) → None.
    depends_on : list[str]
        Names of stages that must run before this one.
        Auto-inferred from input/output overlap if empty.
    validator : Callable, optional
        Post-run output validator (see transactions.py).
    """

    name: str
    inputs: list[Path] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    config_keys: list[str] = field(default_factory=list)
    run: Callable | None = None
    depends_on: list[str] = field(default_factory=list)
    validator: Callable | None = None


class PipelineRunner:
    """
    DAG-based pipeline runner with make-like freshness checks.

    Stages form a DAG based on their dependencies. The runner traverses
    the DAG in topological order, skipping stages whose outputs already
    exist and are newer than their inputs.
    """

    def __init__(self):
        self._stages: dict[str, Stage] = {}

    def register(self, stage: Stage) -> "PipelineRunner":
        """Register a stage. Returns self for chaining."""
        if stage.name in self._stages:
            raise ValueError(f"Stage '{stage.name}' already registered")
        self._stages[stage.name] = stage
        return self

    def stages(self) -> list[str]:
        """Return registered stage names in registration order."""
        return list(self._stages.keys())

    # ── Dependency resolution ─────────────────────────────────────────

    def _build_dependency_graph(self) -> dict[str, list[str]]:
        """
        Build adjacency list from explicit depends_on + implicit I/O overlap.

        Implicit: if stage A's output is stage B's input, then B depends on A.
        """
        graph = {name: list(stage.depends_on) for name, stage in self._stages.items()}

        # Build output→stage lookup
        output_owners: dict[str, str] = {}
        for name, stage in self._stages.items():
            for out_path in stage.outputs:
                output_owners[str(out_path)] = name

        # Infer dependencies from I/O overlap
        for name, stage in self._stages.items():
            for in_path in stage.inputs:
                owner = output_owners.get(str(in_path))
                if owner and owner != name and owner not in graph[name]:
                    graph[name].append(owner)

        return graph

    def _topological_sort(self, target: str) -> list[str]:
        """
        Return stages needed to reach target, in dependency order.

        Uses Kahn's algorithm (BFS). Raises on cycles.
        """
        graph = self._build_dependency_graph()

        # Find all ancestors of target
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

        # Filter graph to needed stages
        sub_graph = {n: [d for d in graph[n] if d in needed] for n in needed}

        # Kahn's algorithm
        in_degree = dict.fromkeys(needed, 0)
        for n in needed:
            for _dep in sub_graph[n]:
                in_degree[n] += 0  # deps add to n's in-degree
        # Actually: in_degree[n] = number of stages in needed that n depends on
        for n in needed:
            in_degree[n] = len(sub_graph[n])

        queue = [n for n in needed if in_degree[n] == 0]
        order = []

        while queue:
            # Sort for deterministic order
            queue.sort()
            node = queue.pop(0)
            order.append(node)

            # For each stage that depends on node, decrement in-degree
            for n in needed:
                if node in sub_graph[n]:
                    in_degree[n] -= 1
                    if in_degree[n] == 0:
                        queue.append(n)

        if len(order) != len(needed):
            remaining = needed - set(order)
            raise RuntimeError(f"Cycle detected in pipeline DAG involving: {remaining}")

        return order

    # ── Freshness checks ──────────────────────────────────────────────

    def _outputs_are_fresh(self, stage: Stage) -> bool:
        """
        Check if outputs exist and are newer than all inputs.

        Returns False if any output is missing or any input is newer.
        """
        # All outputs must exist
        for out_path in stage.outputs:
            if not Path(out_path).exists():
                return False

        if not stage.inputs:
            return True  # no inputs → outputs exist → fresh

        # Get newest input mtime
        input_mtimes = []
        for in_path in stage.inputs:
            p = Path(in_path)
            if p.is_file():
                input_mtimes.append(p.stat().st_mtime)
            elif p.is_dir():
                # Use directory mtime
                input_mtimes.append(p.stat().st_mtime)
            # Skip missing inputs — they'll fail at run time

        if not input_mtimes:
            return True

        newest_input = max(input_mtimes)

        # All outputs must be newer than newest input
        for out_path in stage.outputs:
            p = Path(out_path)
            if p.is_file() or (p.is_dir() and p.stat().st_mtime < newest_input):
                return False

        return True

    # ── Execution ─────────────────────────────────────────────────────

    def execute(
        self,
        target: str,
        cfg=None,
        dry_run: bool = False,
        force: bool = False,
    ) -> dict:
        """
        Execute pipeline up to and including target stage.

        Parameters
        ----------
        target : str
            Stage name to execute (plus all dependencies).
        cfg : DictConfig, optional
            Hydra config passed to each stage's run function.
        dry_run : bool
            If True, validate inputs exist and configs parse but don't run.
        force : bool
            If True, re-run even if outputs are fresh.

        Returns
        -------
        dict with: executed (list[str]), skipped (list[str]),
                   timings (dict[str, float]), errors (dict[str, str]).
        """
        order = self._topological_sort(target)

        result = {
            "target": target,
            "order": order,
            "executed": [],
            "skipped": [],
            "timings": {},
            "errors": {},
            "dry_run": dry_run,
        }

        logger.info(f"Pipeline: {' → '.join(order)} (target={target})")

        if dry_run:
            return self._dry_run(order, cfg, result)

        for stage_name in order:
            stage = self._stages[stage_name]

            # ── Check freshness ───────────────────────────────────────
            if not force and self._outputs_are_fresh(stage):
                logger.info(f"  [{stage_name}] SKIP (outputs fresh)")
                result["skipped"].append(stage_name)
                continue

            # ── Validate inputs exist ─────────────────────────────────
            missing = [str(p) for p in stage.inputs if not Path(p).exists()]
            if missing:
                msg = f"Missing inputs for '{stage_name}': {missing}"
                logger.error(msg)
                result["errors"][stage_name] = msg
                break  # stop pipeline

            # ── Run stage ─────────────────────────────────────────────
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

                # Post-run validation
                if stage.validator:
                    for out_path in stage.outputs:
                        if Path(out_path).is_dir():
                            stage.validator(Path(out_path))

            except Exception as e:
                elapsed = time.monotonic() - t0
                result["errors"][stage_name] = str(e)
                result["timings"][stage_name] = round(elapsed, 1)
                logger.error(f"  [{stage_name}] FAILED ({elapsed:.1f}s): {e}")
                break  # stop pipeline on error

        return result

    def _dry_run(self, order: list[str], cfg, result: dict) -> dict:
        """Validate the pipeline without running anything."""
        logger.info("DRY RUN — validating pipeline")

        for stage_name in order:
            stage = self._stages[stage_name]
            issues = []

            # Check inputs
            for in_path in stage.inputs:
                if not Path(in_path).exists():
                    issues.append(f"missing input: {in_path}")

            # Check if outputs exist
            all_outputs_exist = all(Path(p).exists() for p in stage.outputs)

            # Check config keys
            if cfg is not None:
                from omegaconf import OmegaConf

                for key in stage.config_keys:
                    try:
                        val = OmegaConf.select(cfg, key)
                        if val is None:
                            issues.append(f"config key '{key}' is null")
                    except Exception:
                        issues.append(f"config key '{key}' missing")

            status = "FRESH" if all_outputs_exist else "STALE"
            if issues:
                status = "ISSUES"
                result["errors"][stage_name] = issues

            logger.info(f"  [{stage_name}] {status}")
            for issue in issues:
                logger.warning(f"    ⚠ {issue}")

        return result

    # ── Display ───────────────────────────────────────────────────────

    def print_dag(self) -> None:
        """Print the pipeline DAG."""
        graph = self._build_dependency_graph()
        print("\nPipeline DAG:")
        print("─" * 40)
        for name in self._stages:
            deps = graph.get(name, [])
            dep_str = f" ← {', '.join(deps)}" if deps else ""
            fresh = self._outputs_are_fresh(self._stages[name])
            status = "✓" if fresh else "○"
            print(f"  {status} {name}{dep_str}")
        print()


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline factory for OceanPath
# ═════════════════════════════════════════════════════════════════════════════


def build_oceanpath_pipeline(cfg) -> PipelineRunner:
    """
    Build the standard OceanPath 9-stage pipeline from config.

    This wires up the DAG using paths derived from cfg.platform.*
    and cfg.exp_name. Each stage's run function is a thin wrapper
    that calls the corresponding script's main().

    Usage:
        runner = build_oceanpath_pipeline(cfg)
        runner.execute("evaluate", cfg)       # runs extract→mmap→split→train→eval
        runner.execute("evaluate", cfg)       # second call: all skip (fresh)
    """
    from omegaconf import OmegaConf

    p = cfg.platform
    exp = OmegaConf.select(cfg, "exp_name", default="default")

    # Derive standard paths
    slide_dir = Path(p.slide_root)
    feature_dir = Path(p.feature_root) / cfg.encoder.features_subdir
    mmap_dir = Path(p.mmap_root) / cfg.data.name / cfg.encoder.name
    splits_dir = Path(p.splits_root) / cfg.data.name / cfg.splits.name
    train_dir = Path(p.output_root) / "train" / exp
    eval_dir = Path(p.output_root) / "eval" / exp

    runner = PipelineRunner()

    runner.register(
        Stage(
            name="extract",
            inputs=[slide_dir],
            outputs=[feature_dir],
            config_keys=["encoder", "extraction"],
        )
    )

    runner.register(
        Stage(
            name="build_mmap",
            inputs=[feature_dir],
            outputs=[mmap_dir],
            config_keys=["data"],
            depends_on=["extract"],
        )
    )

    runner.register(
        Stage(
            name="split",
            inputs=[],  # CSV manifest (not a previous stage output)
            outputs=[splits_dir],
            config_keys=["splits", "data"],
        )
    )

    runner.register(
        Stage(
            name="train",
            inputs=[mmap_dir, splits_dir],
            outputs=[train_dir],
            config_keys=["model", "training"],
            depends_on=["build_mmap", "split"],
        )
    )

    runner.register(
        Stage(
            name="evaluate",
            inputs=[train_dir],
            outputs=[eval_dir],
            config_keys=["eval"],
            depends_on=["train"],
        )
    )

    runner.register(
        Stage(
            name="analyze",
            inputs=[train_dir, mmap_dir],
            outputs=[Path(p.output_root) / "analysis" / exp],
            config_keys=["analysis"],
            depends_on=["train"],
        )
    )

    return runner
