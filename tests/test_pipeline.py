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
