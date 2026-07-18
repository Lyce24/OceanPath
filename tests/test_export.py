"""Contract tests for portable model metadata export."""

import json

import pytest
import torch
import torch.nn as nn

from oceanpath.export import ModelCard
from oceanpath.export.exporter import _all_validations_passed


def test_explicit_ensemble_export_is_rejected():
    from oceanpath.workflows.export import _resolve_model_strategy

    with pytest.raises(ValueError, match="multiple checkpoints"):
        _resolve_model_strategy("ensemble")


def test_auto_export_is_rejected_to_prevent_test_set_model_selection():
    from oceanpath.workflows.export import _resolve_model_strategy

    with pytest.raises(ValueError, match="Unsupported"):
        _resolve_model_strategy("auto")


def test_explicit_portable_export_strategy_is_accepted():
    from oceanpath.workflows.export import _resolve_model_strategy

    assert _resolve_model_strategy("refit") == "refit"


def _reference_config() -> dict:
    return {
        "exp_name": "blca_reference",
        "num_classes": 2,
        "data": {"name": "blca", "label_columns": ["Binary WHO 2022"]},
        "encoder": {"name": "uni_v1", "feature_dim": 1024},
        "extraction": {"patch_size": 256, "mag": 20, "overlap": 0},
        "splits": {"name": "kfold5_seed42", "n_folds": 5},
        "model": {"name": "abmil", "arch": "abmil", "embed_dim": 512},
        "training": {"lr": 1e-4, "weight_decay": 5e-3, "max_epochs": 60, "seed": 42},
    }


def test_model_card_uses_canonical_fingerprint():
    card = ModelCard(_reference_config()).build()

    assert card["experiment_name"] == "blca_reference"
    assert len(card["config_fingerprint"]) == 12
    assert card["model"]["num_classes"] == 2
    assert card["data"]["dataset"] == "blca"


def test_model_card_write_round_trip(tmp_path):
    path = tmp_path / "artifact" / "model_card.json"
    written = ModelCard(_reference_config()).add_metrics({"auroc": 0.9}).write(str(path))

    assert written == path
    payload = json.loads(path.read_text())
    assert payload["metrics"]["auroc"] == 0.9


def test_export_report_fails_when_a_format_fails():
    report = {
        "formats": ["onnx"],
        "exports": {
            "onnx": {"status": "failed", "error": "unsupported operator"},
            "checkpoint": {"path": "model.ckpt"},
        },
        "validation": {},
    }

    assert not _all_validations_passed(report)


def test_export_report_fails_when_numerical_validation_fails():
    report = {
        "formats": ["onnx"],
        "exports": {"onnx": {"status": "success"}},
        "validation": {"onnx_numerical": {"status": "fail"}},
    }

    assert not _all_validations_passed(report)


def test_export_report_accepts_successful_formats_and_skipped_gpu_check():
    report = {
        "formats": ["onnx", "torchscript"],
        "exports": {
            "onnx": {"status": "success"},
            "torchscript": {"status": "success"},
            "checkpoint": {"path": "model.ckpt"},
        },
        "validation": {
            "portability": {"status": "skipped"},
            "onnx_numerical": {"status": "pass"},
            "torchscript_numerical": {"status": "pass"},
        },
    }

    assert _all_validations_passed(report)


def test_failed_export_skips_missing_file_validation(tmp_path, monkeypatch):
    from oceanpath.export import Exporter

    source = tmp_path / "source.ckpt"
    source.write_bytes(b"checkpoint")
    exporter = Exporter(
        checkpoint_path=str(source),
        output_dir=str(tmp_path / "artifact"),
        formats=["onnx"],
    )
    monkeypatch.setattr(exporter, "_load_model", lambda: nn.Linear(1, 1))
    monkeypatch.setattr(
        exporter,
        "_validate_portability",
        lambda _model: {"status": "pass", "tests": []},
    )
    monkeypatch.setattr(
        exporter,
        "_export_onnx",
        lambda _model, _path: {"status": "failed", "error": "export error"},
    )
    monkeypatch.setattr(
        exporter,
        "_validate_onnx",
        lambda _model, _path: (_ for _ in ()).throw(
            AssertionError("validation must not run without an exported file")
        ),
    )

    report = exporter.run(skip_validation=False)

    assert report["success"] is False
    assert report["validation"]["onnx_numerical"]["status"] == "skipped"


def test_portability_exception_is_reported_fail_closed(tmp_path, monkeypatch):
    from oceanpath.export import Exporter

    source = tmp_path / "source.ckpt"
    source.write_bytes(b"checkpoint")
    exporter = Exporter(
        checkpoint_path=str(source),
        output_dir=str(tmp_path / "artifact"),
        in_dim=1,
        formats=["onnx"],
    )
    monkeypatch.setattr(exporter, "_load_model", lambda: nn.Linear(1, 1))
    monkeypatch.setattr(
        exporter,
        "_validate_portability",
        lambda _model: (_ for _ in ()).throw(RuntimeError("device validation failed")),
    )
    monkeypatch.setattr(
        exporter,
        "_export_onnx",
        lambda _model, _path: {"status": "failed", "error": "export error"},
    )

    report = exporter.run(skip_validation=False)

    assert report["success"] is False
    assert report["validation"]["portability"] == {
        "status": "fail",
        "error": "device validation failed",
    }


def test_reexport_refreshes_fallback_checkpoint(tmp_path, monkeypatch):
    from oceanpath.export import Exporter

    source = tmp_path / "source.ckpt"
    output_dir = tmp_path / "artifact"

    def _export_once(payload: bytes):
        source.write_bytes(payload)
        exporter = Exporter(
            checkpoint_path=str(source),
            output_dir=str(output_dir),
            formats=["torchscript"],
        )
        monkeypatch.setattr(exporter, "_load_model", lambda: nn.Linear(1, 1))
        monkeypatch.setattr(
            exporter,
            "_export_torchscript",
            lambda _model, _path: {"status": "success"},
        )
        exporter.run(skip_validation=True)

    _export_once(b"old checkpoint")
    _export_once(b"new checkpoint")

    assert (output_dir / "model.ckpt").read_bytes() == b"new checkpoint"
    assert not (output_dir / "model.ckpt.tmp").exists()


def test_tiny_checkpoint_exports_real_onnx_and_torchscript(tmp_path):
    import lightning as lightning_package

    from oceanpath.export import Exporter
    from oceanpath.training.lightning import MILTrainModule

    module = MILTrainModule(
        arch="abmil",
        in_dim=8,
        num_classes=2,
        model_cfg={"embed_dim": 16, "attn_dim": 8, "dropout": 0.0},
        canary_interval=0,
        collect_embeddings=False,
    )
    checkpoint = tmp_path / "tiny.ckpt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": lightning_package.__version__,
        },
        checkpoint,
    )

    report = Exporter(
        checkpoint_path=str(checkpoint),
        output_dir=str(tmp_path / "artifact"),
        in_dim=8,
        num_classes=2,
        formats=["onnx", "torchscript"],
        validation_n_patches=[3],
        device="cpu",
    ).run(skip_validation=False)

    assert report["success"] is True
    assert (tmp_path / "artifact" / "model.onnx").is_file()
    assert (tmp_path / "artifact" / "model.pt").is_file()

    from oceanpath.serving.server import InferenceBackend

    prediction = InferenceBackend(
        str(tmp_path / "artifact"),
        backend="onnx",
    ).predict(torch.randn(3, 8).numpy())
    assert prediction["pred_class"] in (0, 1)
    assert abs(sum(prediction["probabilities"]) - 1.0) < 1.0e-5


def _atomic_export_config(tmp_path):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "exp_name": "reference",
            "platform": {
                "project_root": str(tmp_path),
                "output_root": str(tmp_path / "outputs"),
            },
            "data": {
                "name": "blca",
                "slide_dir": str(tmp_path / "slides"),
                "csv_path": str(tmp_path / "manifest.csv"),
                "feature_job_dir": str(tmp_path / "features"),
                "feature_h5_dir": str(tmp_path / "features" / "h5"),
                "mmap_dir": str(tmp_path / "mmap"),
            },
            "encoder": {"name": "uni", "feature_dim": 4},
            "splits": {"name": "kfold"},
            "train_dir": str(tmp_path / "train"),
            "eval_dir": str(tmp_path / "eval"),
            "export": {
                "artifact_dir": str(tmp_path / "artifact"),
                "formats": ["onnx"],
            },
            "runtime": {"verify_environment": False},
            "dry_run": False,
        }
    )


def test_export_workflow_rolls_back_partial_artifact(tmp_path, monkeypatch):
    import oceanpath.workflows.export as workflow

    cfg = _atomic_export_config(tmp_path)
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "previous.txt").write_text("known-good")

    def _fail(_cfg, paths, **_kwargs):
        (paths.artifact_dir / "model.onnx").write_text("partial")
        raise RuntimeError("export failed")

    monkeypatch.setattr(workflow, "_run_export", _fail)

    with pytest.raises(RuntimeError, match="export failed"):
        workflow.run_export(cfg)

    assert (artifact_dir / "previous.txt").read_text() == "known-good"
    assert not artifact_dir.with_suffix(".tmp").exists()


def test_export_workflow_atomically_commits_complete_artifact(tmp_path, monkeypatch):
    import oceanpath.workflows.export as workflow

    cfg = _atomic_export_config(tmp_path)
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "previous.txt").write_text("old")

    def _succeed(_cfg, paths, **_kwargs):
        for name in (
            "export_report.json",
            "model_card.json",
            "model.ckpt",
            "export_config.yaml",
            "model.onnx",
        ):
            (paths.artifact_dir / name).write_text("new")

    monkeypatch.setattr(workflow, "_run_export", _succeed)

    result = workflow.run_export(cfg)

    assert result.output_dir == artifact_dir
    assert (artifact_dir / "model.onnx").read_text() == "new"
    assert (artifact_dir / "_run" / "status.json").is_file()
    assert not (artifact_dir / "previous.txt").exists()
