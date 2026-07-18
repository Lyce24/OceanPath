"""
Tests for oceanpath.pipeline — DAG orchestration and atomic transactions.

Covers:
  - Topological sort with explicit and implicit dependencies
  - Cycle detection
  - Make-like freshness checks (skip when outputs are newer than inputs)
  - Dry-run mode (validate without executing)
  - Atomic output: commit on success, rollback on failure
  - Validators: file existence, parquet non-empty, NaN detection
"""

import time
from unittest.mock import MagicMock

import pytest

# ═════════════════════════════════════════════════════════════════════════════
# Stage and PipelineRunner
# ═════════════════════════════════════════════════════════════════════════════


class TestStageRegistration:
    def test_register_stage(self):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        runner = PipelineRunner()
        runner.register(Stage(name="extract"))
        assert "extract" in runner.stages()

    def test_duplicate_name_raises(self):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        runner = PipelineRunner()
        runner.register(Stage(name="extract"))
        with pytest.raises(ValueError, match="already registered"):
            runner.register(Stage(name="extract"))

    def test_chaining(self):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        runner = PipelineRunner().register(Stage(name="a")).register(Stage(name="b"))
        assert runner.stages() == ["a", "b"]


class TestTopologicalSort:
    def test_linear_chain(self):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        runner = PipelineRunner()
        runner.register(Stage(name="a"))
        runner.register(Stage(name="b", depends_on=["a"]))
        runner.register(Stage(name="c", depends_on=["b"]))

        order = runner._topological_sort("c")
        assert order == ["a", "b", "c"]

    def test_single_stage_no_deps(self):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        runner = PipelineRunner()
        runner.register(Stage(name="solo"))

        order = runner._topological_sort("solo")
        assert order == ["solo"]

    def test_diamond_dependency(self):
        """A → B, A → C, B → D, C → D."""
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        runner = PipelineRunner()
        runner.register(Stage(name="a"))
        runner.register(Stage(name="b", depends_on=["a"]))
        runner.register(Stage(name="c", depends_on=["a"]))
        runner.register(Stage(name="d", depends_on=["b", "c"]))

        order = runner._topological_sort("d")
        assert order[0] == "a"  # a must come first
        assert order[-1] == "d"  # d must come last
        assert set(order) == {"a", "b", "c", "d"}

    def test_only_needed_stages(self):
        """Requesting 'b' should not include 'c'."""
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        runner = PipelineRunner()
        runner.register(Stage(name="a"))
        runner.register(Stage(name="b", depends_on=["a"]))
        runner.register(Stage(name="c", depends_on=["a"]))

        order = runner._topological_sort("b")
        assert "c" not in order
        assert order == ["a", "b"]

    def test_unknown_target_raises(self):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        runner = PipelineRunner()
        runner.register(Stage(name="a"))

        with pytest.raises(ValueError, match="Unknown stage"):
            runner._topological_sort("nonexistent")

    def test_cycle_detection(self):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        runner = PipelineRunner()
        runner.register(Stage(name="a", depends_on=["b"]))
        runner.register(Stage(name="b", depends_on=["a"]))

        with pytest.raises(RuntimeError, match="Cycle"):
            runner._topological_sort("a")

    def test_implicit_io_dependency(self, tmp_path):
        """Stage B's input is Stage A's output → inferred dependency."""
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        shared = tmp_path / "features"
        runner = PipelineRunner()
        runner.register(Stage(name="a", outputs=[shared]))
        runner.register(Stage(name="b", inputs=[shared]))

        order = runner._topological_sort("b")
        assert order == ["a", "b"]


class TestFreshnessChecks:
    def test_missing_output_is_stale(self, tmp_path):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        runner = PipelineRunner()
        runner.register(Stage(name="a", outputs=[tmp_path / "out.txt"]))

        assert not runner._outputs_are_fresh(runner._stages["a"])

    def test_existing_output_no_inputs_is_fresh(self, tmp_path):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        out = tmp_path / "out.txt"
        out.write_text("done")

        runner = PipelineRunner()
        runner.register(Stage(name="a", outputs=[out]))

        assert runner._outputs_are_fresh(runner._stages["a"])

    def test_output_newer_than_input_is_fresh(self, tmp_path):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        inp = tmp_path / "input.csv"
        out = tmp_path / "output.parquet"

        inp.write_text("data")
        time.sleep(0.05)
        out.write_text("result")

        runner = PipelineRunner()
        runner.register(Stage(name="a", inputs=[inp], outputs=[out]))

        assert runner._outputs_are_fresh(runner._stages["a"])

    def test_output_older_than_input_is_stale(self, tmp_path):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        out = tmp_path / "output.parquet"
        inp = tmp_path / "input.csv"

        out.write_text("old result")
        time.sleep(0.05)
        inp.write_text("new data")

        runner = PipelineRunner()
        runner.register(Stage(name="a", inputs=[inp], outputs=[out]))

        assert not runner._outputs_are_fresh(runner._stages["a"])

    def test_nested_input_rewrite_is_stale_with_transaction(self, tmp_path):
        from omegaconf import OmegaConf

        from oceanpath.pipeline.dag import PipelineRunner, Stage
        from oceanpath.pipeline.transactions import write_stage_transaction

        input_dir = tmp_path / "features"
        input_dir.mkdir()
        nested_input = input_dir / "slide.h5"
        nested_input.write_text("old features")
        output_dir = tmp_path / "mmap"
        output_dir.mkdir()

        cfg = OmegaConf.create({"storage": {"precision": 16}})
        runner = PipelineRunner().register(
            Stage(
                name="build_mmap",
                inputs=[input_dir],
                outputs=[output_dir],
                config_keys=["storage"],
            )
        )
        stage = runner._stages["build_mmap"]
        write_stage_transaction(
            output_dir,
            stage_name=stage.name,
            stage_fingerprint=runner.stage_fingerprint(stage.name, cfg=cfg),
        )
        assert runner._outputs_are_fresh(stage, cfg=cfg)

        time.sleep(0.05)
        nested_input.write_text("new features")

        assert not runner._outputs_are_fresh(stage, cfg=cfg)

    def test_invalid_existing_output_is_stale(self, tmp_path):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        output = tmp_path / "artifact"
        output.mkdir()

        def _validate(path):
            if not (path / "required.json").exists():
                raise FileNotFoundError("required.json")

        runner = PipelineRunner().register(
            Stage(name="export", outputs=[output], validator=_validate)
        )

        assert not runner._outputs_are_fresh(runner._stages["export"])

    def test_missing_declared_input_is_never_fresh(self, tmp_path):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        output = tmp_path / "result.txt"
        output.write_text("stale")
        runner = PipelineRunner().register(
            Stage(name="train", inputs=[tmp_path / "missing"], outputs=[output])
        )

        assert not runner._outputs_are_fresh(runner._stages["train"])


class TestPipelineChildOverrides:
    def test_material_parent_overrides_are_forwarded(self, monkeypatch):
        from oceanpath.workflows import pipeline as workflow

        monkeypatch.setattr(
            workflow,
            "_runtime_task_overrides",
            lambda: [
                "training.lr=0.123",
                "platform.project_root=/scratch/project",
                "pipeline.force=true",
                "+eval.seed=7",
            ],
        )

        assert workflow._forward_task_overrides({"training", "platform"}) == [
            "training.lr=0.123",
            "platform.project_root=/scratch/project",
        ]
        assert workflow._forward_task_overrides({"eval"}) == ["eval.seed=7"]

    def test_dag_run_enforces_child_recomputation(self, monkeypatch):
        from omegaconf import OmegaConf

        from oceanpath.workflows import pipeline as workflow

        captured = {}
        monkeypatch.setattr(workflow, "_runtime_task_overrides", lambda: ["training.lr=0.123"])
        monkeypatch.setattr(
            workflow,
            "_run_command",
            lambda cmd, timeout_sec, print_command: captured.update(cmd=cmd),
        )
        cfg = OmegaConf.create(
            {
                "verbose": False,
                "pipeline": {
                    "python_exe": "python",
                    "stage_timeout_sec": None,
                    "print_stage_commands": False,
                },
            }
        )

        workflow._run_stage_script(
            cfg,
            {"training": "default"},
            stage_name="train_model",
            script_name="train.py",
            required_groups=["training"],
            enforced_overrides=["training.force_rerun=true"],
        )

        assert "training.lr=0.123" in captured["cmd"]
        assert captured["cmd"][-1] == "training.force_rerun=true"

    def test_export_child_composes_the_selected_eval_profile(self, monkeypatch):
        from types import SimpleNamespace

        from omegaconf import OmegaConf

        from oceanpath.workflows import pipeline as workflow

        captured = {}
        monkeypatch.setattr(
            workflow.FoundationPaths,
            "from_config",
            lambda _cfg: SimpleNamespace(
                experiment_name="example",
                train_dir="/outputs/train",
                eval_dir="/outputs/train/eval",
                artifact_dir="/outputs/artifacts/example",
            ),
        )
        monkeypatch.setattr(
            workflow,
            "_run_stage_script",
            lambda _cfg, _choices, **kwargs: captured.update(kwargs),
        )

        workflow._run_export(OmegaConf.create({}), {"eval": "default"})

        assert "eval" in captured["required_groups"]


class TestExecution:
    def test_runs_stage(self, tmp_path):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        out = tmp_path / "result.txt"
        run_fn = MagicMock(side_effect=lambda cfg: out.write_text("done"))

        runner = PipelineRunner()
        runner.register(Stage(name="a", outputs=[out], run=run_fn))

        result = runner.execute("a", cfg=None)
        assert "a" in result["executed"]
        run_fn.assert_called_once()

    def test_skips_fresh_stage(self, tmp_path):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        out = tmp_path / "result.txt"
        out.write_text("already done")

        run_fn = MagicMock()
        runner = PipelineRunner()
        runner.register(Stage(name="a", outputs=[out], run=run_fn))

        result = runner.execute("a", cfg=None)
        assert "a" in result["skipped"]
        run_fn.assert_not_called()

    def test_force_reruns_fresh_stage(self, tmp_path):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        out = tmp_path / "result.txt"
        out.write_text("already done")

        run_fn = MagicMock()
        runner = PipelineRunner()
        runner.register(Stage(name="a", outputs=[out], run=run_fn))

        result = runner.execute("a", cfg=None, force=True)
        assert "a" in result["executed"]
        run_fn.assert_called_once()

    def test_stops_on_error(self, tmp_path):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        run_a = MagicMock(side_effect=RuntimeError("boom"))
        run_b = MagicMock()

        runner = PipelineRunner()
        runner.register(Stage(name="a", run=run_a))
        runner.register(Stage(name="b", depends_on=["a"], run=run_b))

        result = runner.execute("b", cfg=None, force=True)
        assert "a" in result["errors"]
        assert "b" not in result["executed"]
        run_b.assert_not_called()

    def test_stops_on_missing_input(self, tmp_path):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        run_fn = MagicMock()
        runner = PipelineRunner()
        runner.register(
            Stage(
                name="a",
                inputs=[tmp_path / "nonexistent"],
                run=run_fn,
            )
        )

        result = runner.execute("a", cfg=None, force=True)
        assert "a" in result["errors"]
        run_fn.assert_not_called()

    def test_missing_declared_output_fails_without_creating_it(self, tmp_path):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        output = tmp_path / "missing-output"
        run_fn = MagicMock()
        runner = PipelineRunner()
        runner.register(Stage(name="a", outputs=[output], run=run_fn))

        result = runner.execute("a", cfg=None)

        assert "a" in result["errors"]
        assert "a" not in result["executed"]
        assert "did not produce declared outputs" in result["errors"]["a"]
        assert not output.exists()

    def test_dry_run_explains_missing_inputs(self, tmp_path, caplog):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        missing = tmp_path / "missing.csv"
        run_fn = MagicMock()
        runner = PipelineRunner()
        runner.register(Stage(name="a", inputs=[missing], run=run_fn))

        with caplog.at_level("INFO"):
            result = runner.execute("a", cfg=None, dry_run=True)

        assert result["errors"]["a"] == [f"missing input: {missing}"]
        assert f"missing input: {missing}" in caplog.text
        run_fn.assert_not_called()


# ═════════════════════════════════════════════════════════════════════════════
# Atomic transactions
# ═════════════════════════════════════════════════════════════════════════════


class TestAtomicOutput:
    def test_commit_on_success(self, tmp_path):
        from oceanpath.pipeline.transactions import atomic_output

        final = tmp_path / "output"

        with atomic_output(final) as tmp:
            (tmp / "model.pt").write_text("weights")
            (tmp / "metrics.json").write_text("{}")

        assert final.is_dir()
        assert (final / "model.pt").is_file()
        assert (final / "metrics.json").is_file()
        assert not final.with_suffix(".tmp").exists()

    def test_rollback_on_exception(self, tmp_path):
        from oceanpath.pipeline.transactions import atomic_output

        final = tmp_path / "output"

        def simulate_crash():
            with atomic_output(final) as tmp:
                (tmp / "partial.txt").write_text("incomplete")
                raise RuntimeError("training crashed")

        with pytest.raises(RuntimeError, match="training crashed"):
            simulate_crash()

        assert not final.exists()
        assert not final.with_suffix(".tmp").exists()

    def test_validator_prevents_commit(self, tmp_path):
        from oceanpath.pipeline.transactions import atomic_output

        final = tmp_path / "output"

        def bad_validator(d):
            raise ValueError("missing required file")

        with (
            pytest.raises(ValueError, match=r"missing required file"),
            atomic_output(final, validator=bad_validator) as tmp,
        ):
            (tmp / "incomplete.txt").write_text("data")

        assert not final.exists()

    def test_replaces_existing_output(self, tmp_path):
        from oceanpath.pipeline.transactions import atomic_output

        final = tmp_path / "output"
        final.mkdir()
        (final / "old.txt").write_text("old data")

        with atomic_output(final) as tmp:
            (tmp / "new.txt").write_text("new data")

        assert (final / "new.txt").is_file()
        assert not (final / "old.txt").exists()

    def test_cleans_stale_tmp(self, tmp_path):
        from oceanpath.pipeline.transactions import atomic_output

        final = tmp_path / "output"
        stale = final.with_suffix(".tmp")
        stale.mkdir(parents=True)
        (stale / "crashed.txt").write_text("leftover")

        with atomic_output(final) as tmp:
            (tmp / "fresh.txt").write_text("new")

        assert final.is_dir()
        assert not stale.exists()


class TestValidators:
    def test_validate_glob_not_empty(self, tmp_path):
        from oceanpath.pipeline.transactions import validate_glob_not_empty

        validator = validate_glob_not_empty("*.h5")
        with pytest.raises(FileNotFoundError, match=r"\*\.h5"):
            validator(tmp_path)

        (tmp_path / "slide.h5").touch()
        validator(tmp_path)

    def test_validate_files_exist_passes(self, tmp_path):
        from oceanpath.pipeline.transactions import validate_files_exist

        (tmp_path / "model.pt").write_text("weights")
        (tmp_path / "metrics.json").write_text("{}")

        validator = validate_files_exist("model.pt", "metrics.json")
        validator(tmp_path)  # should not raise

    def test_validate_files_exist_fails(self, tmp_path):
        from oceanpath.pipeline.transactions import validate_files_exist

        (tmp_path / "model.pt").write_text("weights")

        validator = validate_files_exist("model.pt", "metrics.json")
        with pytest.raises(FileNotFoundError, match=r"metrics\.json"):
            validator(tmp_path)

    def test_validate_parquet_not_empty_passes(self, tmp_path):
        import pandas as pd

        from oceanpath.pipeline.transactions import validate_parquet_not_empty

        df = pd.DataFrame({"a": [1, 2, 3]})
        df.to_parquet(tmp_path / "data.parquet", index=False)

        validator = validate_parquet_not_empty("data.parquet")
        validator(tmp_path)

    def test_validate_parquet_not_empty_fails(self, tmp_path):
        import pandas as pd

        from oceanpath.pipeline.transactions import validate_parquet_not_empty

        df = pd.DataFrame({"a": []})
        df.to_parquet(tmp_path / "data.parquet", index=False)

        validator = validate_parquet_not_empty("data.parquet")
        with pytest.raises(ValueError, match=r"Empty parquet file"):
            validator(tmp_path)

    def test_validate_no_nans(self, tmp_path):
        import numpy as np
        import pandas as pd

        from oceanpath.pipeline.transactions import validate_no_nans

        df = pd.DataFrame({"prob_0": [0.5, np.nan, 0.3]})
        df.to_parquet(tmp_path / "preds.parquet", index=False)

        validator = validate_no_nans("preds.parquet")
        with pytest.raises(ValueError, match=r"NaN detected in column prob_0"):
            validator(tmp_path)

    def test_compose_validators(self, tmp_path):
        from oceanpath.pipeline.transactions import (
            compose_validators,
            validate_files_exist,
        )

        v1 = validate_files_exist("a.txt")
        v2 = validate_files_exist("b.txt")
        composed = compose_validators(v1, v2)

        (tmp_path / "a.txt").write_text("ok")
        with pytest.raises(FileNotFoundError, match=r"b\.txt"):
            composed(tmp_path)


# ═════════════════════════════════════════════════════════════════════════════
# Fingerprints, DAG rendering, and pipeline factories
# ═════════════════════════════════════════════════════════════════════════════


class TestFingerprints:
    def test_stage_fingerprint_deterministic(self):
        from omegaconf import OmegaConf

        from oceanpath.pipeline.dag import PipelineRunner, Stage

        cfg = OmegaConf.create({"model": {"name": "abmil"}, "training": {"lr": 1e-3}})
        runner = PipelineRunner().register(Stage(name="train", config_keys=["model", "training"]))

        fp1 = runner.stage_fingerprint("train", cfg=cfg)
        fp2 = runner.stage_fingerprint("train", cfg=cfg)
        assert fp1 == fp2

    def test_stage_fingerprint_changes_with_config(self):
        from omegaconf import OmegaConf

        from oceanpath.pipeline.dag import PipelineRunner, Stage

        cfg1 = OmegaConf.create({"training": {"lr": 1e-3}})
        cfg2 = OmegaConf.create({"training": {"lr": 1e-4}})
        runner = PipelineRunner().register(Stage(name="train", config_keys=["training"]))

        assert runner.stage_fingerprint("train", cfg=cfg1) != runner.stage_fingerprint(
            "train", cfg=cfg2
        )

    def test_pipeline_fingerprint_changes_with_config(self):
        from omegaconf import OmegaConf

        from oceanpath.pipeline.dag import PipelineRunner, Stage

        cfg1 = OmegaConf.create({"a": {"x": 1}, "b": {"y": 2}})
        cfg2 = OmegaConf.create({"a": {"x": 2}, "b": {"y": 2}})

        runner = PipelineRunner()
        runner.register(Stage(name="s1", config_keys=["a"]))
        runner.register(Stage(name="s2", config_keys=["b"], depends_on=["s1"]))

        fp1 = runner.pipeline_fingerprint("s2", cfg=cfg1)
        fp2 = runner.pipeline_fingerprint("s2", cfg=cfg2)
        assert fp1 != fp2

    def test_stage_fingerprint_changes_with_domain_code(self, tmp_path):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        source_dir = tmp_path / "domain"
        source_dir.mkdir()
        implementation = source_dir / "service.py"
        implementation.write_text("VERSION = 1\n")
        runner = PipelineRunner().register(Stage(name="train", code_paths=[source_dir]))

        before = runner.stage_fingerprint("train")
        implementation.write_text("VERSION = 2\n")

        assert runner.stage_fingerprint("train") != before

    def test_stage_fingerprint_ignores_python_cache_files(self, tmp_path):
        from oceanpath.pipeline.dag import PipelineRunner, Stage

        source_dir = tmp_path / "domain"
        source_dir.mkdir()
        (source_dir / "service.py").write_text("VERSION = 1\n")
        runner = PipelineRunner().register(Stage(name="train", code_paths=[source_dir]))
        before = runner.stage_fingerprint("train")

        cache = source_dir / "__pycache__"
        cache.mkdir()
        (cache / "service.cpython-310.pyc").write_bytes(b"environment-specific")

        assert runner.stage_fingerprint("train") == before


class TestDagRendering:
    def test_render_dag_mermaid_contains_edges_and_fingerprints(self):
        from omegaconf import OmegaConf

        from oceanpath.pipeline.dag import PipelineRunner, Stage

        cfg = OmegaConf.create({"x": 1})
        runner = PipelineRunner()
        runner.register(Stage(name="a", config_keys=["x"]))
        runner.register(Stage(name="b", depends_on=["a"], config_keys=["x"]))

        mermaid = runner.render_dag(target="b", cfg=cfg, include_fingerprint=True)
        assert mermaid.startswith("graph TD")
        assert "fp:" in mermaid
        assert "-->" in mermaid


class TestTransactionMetadata:
    def test_stage_transaction_roundtrip(self, tmp_path):
        from oceanpath.pipeline.transactions import (
            read_stage_transaction,
            transaction_matches,
            write_stage_transaction,
        )

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        write_stage_transaction(
            out_dir,
            stage_name="train_model",
            stage_fingerprint="abc123",
            inputs=["/tmp/in.parquet"],
            config_keys=["model", "training"],
        )

        meta = read_stage_transaction(out_dir)
        assert meta is not None
        assert meta["stage_name"] == "train_model"
        assert meta["stage_fingerprint"] == "abc123"
        assert transaction_matches(out_dir, stage_name="train_model", stage_fingerprint="abc123")
        assert not transaction_matches(
            out_dir, stage_name="train_model", stage_fingerprint="xyz999"
        )


class TestPipelineFactories:
    def _supervised_cfg(self):
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "exp_name": "exp_supervised",
                "platform": {
                    "project_root": "/proj",
                    "slide_root": "/slides",
                    "feature_root": "/features",
                    "mmap_root": "/mmap",
                    "splits_root": "/splits",
                    "output_root": "/outputs",
                },
                "data": {
                    "name": "blca",
                    "slide_dir": "/slides/blca",
                    "feature_h5_dir": "/features/blca/h5",
                    "mmap_dir": "/mmap/blca/uni",
                    "csv_path": "/proj/manifests/blca.csv",
                },
                "encoder": {"name": "uni", "features_subdir": "features_uni"},
                "splits": {"name": "kfold5"},
                "model": {"name": "abmil"},
                "training": {"lr": 1e-3},
                "eval": {},
                "analysis": {},
                "export": {"artifact_root": "/artifacts"},
                "serve": {"backend": "auto"},
            }
        )

    def test_supervised_factory_stages(self):
        from oceanpath.workflows.supervised_pipeline import build_supervised_pipeline

        runner = build_supervised_pipeline(self._supervised_cfg())
        assert runner.stages() == [
            "extract_features",
            "build_mmap",
            "split_data",
            "train_model",
            "evaluate",
            "export_model",
        ]

    def test_split_branch_is_independent_until_training(self):
        from oceanpath.workflows.supervised_pipeline import build_supervised_pipeline

        runner = build_supervised_pipeline(self._supervised_cfg())
        graph = runner._build_dependency_graph()

        assert graph["split_data"] == []
        assert set(graph["train_model"]) == {"build_mmap", "split_data"}

    def test_factory_code_participates_in_every_stage_fingerprint(self):
        from oceanpath.workflows.supervised_pipeline import build_supervised_pipeline

        runner = build_supervised_pipeline(self._supervised_cfg())

        for stage in runner._stages.values():
            assert any(path.name == "supervised_pipeline.py" for path in stage.code_paths)


def test_generic_pipeline_package_has_no_foundation_or_workflow_dependencies():
    from pathlib import Path

    pipeline_dir = Path(__file__).resolve().parents[1] / "src" / "oceanpath" / "pipeline"
    source = "\n".join(path.read_text() for path in pipeline_dir.glob("*.py"))

    forbidden = (
        "FoundationPaths",
        "oceanpath.config",
        "oceanpath.extraction",
        "oceanpath.splitting",
        "oceanpath.storage",
        "oceanpath.workflows",
    )
    for dependency in forbidden:
        assert dependency not in source
    assert "build_supervised_pipeline" not in source
