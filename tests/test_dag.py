"""
Tests for the pipeline DAG runner (``oceanpath.pipeline.dag``), beyond the
ordering, mtime freshness and factory stage lists in test_pipeline.py.

Covers:
  * fingerprint-aware freshness: each run records a transaction per output; the
    same config skips, a change to a key the stage reads re-runs it, a change to
    other keys does not; a failed validator leaves the stage stale
  * stages without outputs or without a run function; outputs whose inputs are
    gone; dry runs (issues and fingerprints, nothing executed)
  * Mermaid rendering and print_dag status marks
  * the supervised and pretraining factories: paths from platform roots, plain
    dict configs, artifact directories, wired run functions executing in order
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from oceanpath.pipeline.dag import (
    PipelineRunner,
    Stage,
    build_oceanpath_pipeline,
    build_pretraining_pipeline,
    build_supervised_pipeline,
)
from oceanpath.pipeline.transactions import read_stage_transaction

# ═════════════════════════════════════════════════════════════════════════════
# Execution and freshness
# ═════════════════════════════════════════════════════════════════════════════


class TestFingerprintFreshness:
    @staticmethod
    def _train_runner(out: Path, calls: list) -> PipelineRunner:
        def run(cfg):
            calls.append(cfg.training.lr)
            out.write_text(str(cfg.training.lr))

        stage = Stage(name="train", outputs=[out], config_keys=["training"], run=run)
        return PipelineRunner().register(stage)

    def test_config_changes_decide_whether_a_stage_reruns(self, tmp_path):
        out, calls = tmp_path / "model.txt", []
        runner = self._train_runner(out, calls)
        cfg = OmegaConf.create({"training": {"lr": 1e-3}, "eval": {"bootstrap": 100}})

        first = runner.execute("train", cfg=cfg)
        assert first["executed"] == ["train"]
        transaction = read_stage_transaction(out)
        assert transaction["stage_name"] == "train"
        assert transaction["stage_fingerprint"] == first["stage_fingerprints"]["train"]

        assert runner.execute("train", cfg=cfg)["skipped"] == ["train"]
        other_keys = OmegaConf.merge(cfg, {"eval": {"bootstrap": 500}})
        assert runner.execute("train", cfg=other_keys)["skipped"] == ["train"]
        new_lr = OmegaConf.merge(cfg, {"training": {"lr": 1e-4}})
        assert runner.execute("train", cfg=new_lr)["executed"] == ["train"]
        assert calls == [1e-3, 1e-4]

    def test_failed_validation_leaves_the_stage_stale(self, tmp_path):
        out = tmp_path / "preds.txt"
        checked = []

        def validator(path):
            checked.append(path)
            if len(checked) == 1:
                raise ValueError("NaN in predictions")

        stage = Stage(
            name="evaluate",
            outputs=[out],
            config_keys=["eval"],
            run=lambda cfg: out.write_text("preds"),
            validator=validator,
        )
        runner = PipelineRunner().register(stage)
        cfg = OmegaConf.create({"eval": {}})

        failed = runner.execute("evaluate", cfg=cfg)
        assert failed["errors"] == {"evaluate": "NaN in predictions"}
        assert failed["executed"] == ["evaluate"]  # it ran; its output was rejected
        assert read_stage_transaction(out) is None
        # The output file exists but carries no transaction, so it is not trusted.
        assert runner.execute("evaluate", cfg=cfg)["errors"] == {}
        assert checked == [out, out]
        assert runner.execute("evaluate", cfg=cfg)["skipped"] == ["evaluate"]

    def test_stage_without_outputs_always_runs(self):
        calls = []
        runner = PipelineRunner().register(Stage(name="notify", run=calls.append))
        runner.execute("notify")
        runner.execute("notify")
        assert calls == [None, None]

    def test_stage_without_a_run_function_is_skipped_and_the_pipeline_continues(self, tmp_path):
        out = tmp_path / "b.txt"
        runner = PipelineRunner()
        runner.register(Stage(name="a"))
        runner.register(
            Stage(name="b", depends_on=["a"], outputs=[out], run=lambda cfg: out.touch())
        )
        result = runner.execute("b")
        assert (result["skipped"], result["executed"]) == (["a"], ["b"])
        assert out.is_file()

    def test_outputs_stay_fresh_when_their_inputs_are_gone(self, tmp_path):
        """E.g. raw slides archived after feature extraction: nothing to re-run."""
        out, ran = tmp_path / "features", []
        out.mkdir()
        stage = Stage(name="extract", inputs=[tmp_path / "slides"], outputs=[out], run=ran.append)
        result = PipelineRunner().register(stage).execute("extract")
        assert result["skipped"] == ["extract"] and ran == []

    def test_unknown_stage_fingerprint(self):
        with pytest.raises(ValueError, match="Unknown stage"):
            PipelineRunner().stage_fingerprint("missing")


class TestDryRun:
    def test_reports_issues_and_fingerprints_without_running(self, tmp_path):
        ran = []
        csv = tmp_path / "manifest.csv"
        csv.write_text("x")
        runner = PipelineRunner()
        runner.register(
            Stage(name="split", inputs=[csv], outputs=[tmp_path / "splits"], run=ran.append)
        )
        runner.register(
            Stage(
                name="train",
                inputs=[tmp_path / "splits", tmp_path / "mmap"],
                outputs=[tmp_path / "train"],
                run=ran.append,
            )
        )
        result = runner.execute("train", dry_run=True)
        assert result["dry_run"] is True
        assert result["order"] == ["split", "train"]
        assert (result["executed"], result["skipped"], ran) == ([], [], [])
        assert result["errors"] == {
            "train": [
                f"missing input: {tmp_path / 'splits'}",
                f"missing input: {tmp_path / 'mmap'}",
            ]
        }
        assert set(result["stage_fingerprints"]) == {"split", "train"}
        assert not (tmp_path / "splits").exists()


# ═════════════════════════════════════════════════════════════════════════════
# Rendering
# ═════════════════════════════════════════════════════════════════════════════


class TestRendering:
    @staticmethod
    def _runner(tmp_path) -> PipelineRunner:
        runner = PipelineRunner()
        runner.register(Stage(name="build-mmap", description='Stage "2"', outputs=[tmp_path / "m"]))
        runner.register(Stage(name="train", inputs=[tmp_path / "m"]))  # implicit dependency
        runner.register(Stage(name="unrelated"))
        return runner

    def test_mermaid_nodes_labels_and_edges(self, tmp_path):
        runner = self._runner(tmp_path)
        lines = runner.render_dag(target="train", include_fingerprint=False).splitlines()
        assert lines == [
            "graph TD",
            "    stage_build_mmap[\"build-mmap\\nStage '2'\"]",
            '    stage_train["train"]',
            "    stage_build_mmap --> stage_train",
        ]
        everything = runner.render_dag()
        assert "stage_unrelated" in everything
        assert everything.count("fp:") == 3

    def test_print_dag_marks_fresh_and_stale_stages(self, tmp_path, capsys):
        runner = self._runner(tmp_path)
        (tmp_path / "m").mkdir()
        runner.print_dag(target="train")
        printed = capsys.readouterr().out
        fp = runner.stage_fingerprint("build-mmap")
        assert f"✓ build-mmap [fp:{fp}]" in printed
        assert "○ train <- build-mmap [fp:" in printed
        assert "unrelated" not in printed


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline factories
# ═════════════════════════════════════════════════════════════════════════════

SUPERVISED_CFG = {
    "exp_name": "exp1",
    "platform": {
        "project_root": "/proj",
        "slide_root": "/slides",
        "feature_root": "/features",
        "mmap_root": "/mmap",
        "splits_root": "/splits",
        "output_root": "/out",
    },
    "data": {"name": "blca"},
    "encoder": {"name": "uni", "features_subdir": "feats_uni"},
    "splits": {"name": "kfold5"},
}


def _io(runner: PipelineRunner) -> dict[str, tuple[list[Path], list[Path]]]:
    return {name: (stage.inputs, stage.outputs) for name, stage in runner._stages.items()}


class TestPipelineFactories:
    @pytest.mark.parametrize("as_dictconfig", [False, True])
    def test_supervised_paths_derive_from_platform_roots(self, as_dictconfig):
        cfg = OmegaConf.create(SUPERVISED_CFG) if as_dictconfig else SUPERVISED_CFG
        io = _io(build_supervised_pipeline(cfg))
        train, evaluation = Path("/out/train/exp1"), Path("/out/train/exp1/eval")
        assert io["extract_features"] == ([Path("/slides")], [Path("/features/blca/feats_uni")])
        assert io["build_mmap"] == ([Path("/features/blca/feats_uni")], [Path("/mmap/blca/uni")])
        assert io["split_data"] == ([Path("/proj/manifests")], [Path("/splits/blca/kfold5")])
        assert io["train_model"] == ([Path("/mmap/blca/uni"), Path("/splits/blca/kfold5")], [train])
        assert io["evaluate"] == ([train], [evaluation])
        assert io["analyze"] == ([train, evaluation], [Path("/out/analyze/exp1")])
        (artifact,) = io["export_and_serve"][1]
        assert artifact.parent == Path("/out/artifacts")
        assert artifact.name.startswith("exp1_")

    def test_explicit_paths_win(self):
        cfg = {
            "exp_name": "exp1",
            "data": {"name": "blca", "feature_h5_dir": "/h5", "mmap_dir": "/store"},
            "splits": {"output_dir": "/my_splits"},
            "export": {"artifact_dir": "/artifacts/exp1"},
        }
        io = _io(build_supervised_pipeline(cfg))
        assert io["build_mmap"] == ([Path("/h5")], [Path("/store")])
        assert io["split_data"][1] == [Path("/my_splits")]
        assert io["export_and_serve"][1] == [Path("/artifacts/exp1")]
        # "null" from the command line means unset: fall back to the artifact root.
        cfg["export"] = {"artifact_dir": "null", "artifact_root": "/models"}
        (artifact,) = _io(build_supervised_pipeline(cfg))["export_and_serve"][1]
        assert artifact.parent == Path("/models")

    def test_a_null_required_path_is_rejected(self):
        with pytest.raises(ValueError, match="None"):
            build_supervised_pipeline(OmegaConf.create({"data": {"mmap_dir": None}}))

    def test_wired_stages_run_in_order_and_then_stay_fresh(self, tmp_path):
        root = tmp_path.as_posix()
        cfg = OmegaConf.create(
            {
                "exp_name": "e",
                "platform": {"output_root": f"{root}/out"},
                "data": {
                    "name": "d",
                    "slide_dir": f"{root}/slides",
                    "feature_h5_dir": f"{root}/h5",
                    "mmap_dir": f"{root}/mmap",
                },
            }
        )
        (tmp_path / "slides").mkdir()
        ran = []

        def make_dir(name):
            def run(cfg):
                ran.append(name)
                (tmp_path / name).mkdir()

            return run

        runner = build_supervised_pipeline(
            cfg, stage_runs={"extract_features": make_dir("h5"), "build_mmap": make_dir("mmap")}
        )
        assert runner._stages["train_model"].run is None
        result = runner.execute("build_mmap", cfg=cfg)
        assert result["executed"] == ["extract_features", "build_mmap"] and ran == ["h5", "mmap"]
        assert runner.execute("build_mmap", cfg=cfg)["skipped"] == [
            "extract_features",
            "build_mmap",
        ]

    def test_pretraining_paths_and_order(self):
        cfg = {
            "exp_name": "ssl1",
            "platform": {"mmap_root": "/mmap", "output_root": "/out"},
            "data": {"name": "pre"},
            "encoder": {"name": "uni2"},
        }
        runner = build_pretraining_pipeline(cfg)
        io = _io(runner)
        manifest = Path("/out/pretrain/ssl1/split_manifest.json")
        assert io["build_mmap"] == ([], [Path("/mmap/pre/uni2")])
        assert io["split_data"] == ([Path("/mmap/pre/uni2")], [manifest])
        assert io["pretrain_model"] == (
            [Path("/mmap/pre/uni2"), manifest],
            [Path("/out/pretrain/ssl1")],
        )
        assert runner._topological_sort("export_and_serve") == [
            "build_mmap",
            "split_data",
            "pretrain_model",
            "export_and_serve",
        ]

    def test_oceanpath_pipeline_is_the_supervised_pipeline(self):
        a = build_oceanpath_pipeline(SUPERVISED_CFG)
        b = build_supervised_pipeline(SUPERVISED_CFG)
        assert _io(a) == _io(b)
        assert a.pipeline_fingerprint("export_and_serve", SUPERVISED_CFG) == b.pipeline_fingerprint(
            "export_and_serve", SUPERVISED_CFG
        )
