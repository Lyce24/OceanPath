"""
Tests for the serving path (``oceanpath.serving``): export and the FastAPI app.

Train-free: each test builds a tiny ``MILTrainModule`` (a ``WSIClassifier``
inside) and writes it with ``Trainer.save_checkpoint``, which is the Lightning
checkpoint format ``Exporter`` loads. The exporter then writes ONNX and
TorchScript artifacts and checks their numerical parity with PyTorch across
bag sizes; ``create_app`` serves the artifact directory.

Covers:
  * Exporter: report and artifact layout, parity across bag sizes for every
    aggregator that exports on CPU, and the failure paths: a format that fails
    to export, a parity check that cannot run (no onnxruntime), and parity
    checks that fail on a tampered artifact or on a trace specialised to one
    bag size
  * ModelCard: serving contract and provenance fields
  * InferenceBackend / create_app: GET /health and /model-card, POST /predict
    and /predict/batch for the onnx, torchscript and pytorch backends, input
    validation, and agreement between the backends

The exporter's CUDA portability branch is not exercised (CPU-only suite):
``torch.cuda.is_available`` is patched to False so every run takes the CPU
path.
"""

import hashlib
import io
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import numpy as np
import pytest
import torch
import torch.nn as nn

pytest.importorskip("onnxruntime", reason="serving tests need the 'serve' extra")
pytest.importorskip("fastapi", reason="serving tests need the 'serve' extra")

import onnx
import onnxruntime as ort
from fastapi.testclient import TestClient
from onnx import numpy_helper

from oceanpath import models as model_registry
from oceanpath.models.base import BaseMIL
from oceanpath.modules.train_module import MILTrainModule
from oceanpath.serving import Exporter, InferenceBackend, ModelCard, create_app
from oceanpath.serving.exporter import config_fingerprint
from oceanpath.serving.model_card import (
    load_oof_metrics,
    load_threshold_info,
)

IN_DIM = 16
NUM_CLASSES = 3
BAG_SIZES = [1, 17, 256]
BACKENDS = ["onnx", "torchscript", "pytorch"]
ABMIL_CFG = {"embed_dim": 8, "attn_dim": 8, "dropout": 0.0}
CARD_CFG = {
    "exp_name": "tiny_abmil",
    "num_classes": NUM_CLASSES,
    "model": {"arch": "abmil", "name": "abmil", "embed_dim": 8},
    "encoder": {"name": "synthetic", "feature_dim": IN_DIM},
    "training": {"lr": 1e-3, "seed": 0},
}


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures and helpers
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _write_checkpoint(
    path: Path, arch: str = "abmil", model_cfg=None, seed: int = 0, num_classes: int = NUM_CLASSES
):
    """Save an untrained MILTrainModule the way training does (Trainer.save_checkpoint)."""
    torch.manual_seed(seed)
    module = MILTrainModule(
        arch=arch,
        in_dim=IN_DIM,
        num_classes=num_classes,
        model_cfg=dict(ABMIL_CFG if model_cfg is None else model_cfg),
        loss_type="bce" if num_classes == 1 else "ce",
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.strategy.connect(module)
    trainer.save_checkpoint(path)
    return module.model.eval()


def _export(
    tmp_path: Path, arch: str = "abmil", model_cfg=None, skip_validation=False, **exporter_kwargs
):
    ckpt = tmp_path / "model.ckpt"
    model = _write_checkpoint(ckpt, arch, model_cfg)
    exporter_kwargs.setdefault("validation_n_patches", BAG_SIZES)
    exporter = Exporter(
        checkpoint_path=str(ckpt),
        output_dir=str(tmp_path / "artifact"),
        in_dim=IN_DIM,
        num_classes=NUM_CLASSES,
        **exporter_kwargs,
    )
    return exporter.run(skip_validation=skip_validation), model, tmp_path / "artifact"


@pytest.fixture(scope="module")
def abmil_artifact(tmp_path_factory):
    """ABMIL exported to ONNX + TorchScript, with a model card (as export_model.py writes it)."""
    root = tmp_path_factory.mktemp("abmil_export")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.cuda, "is_available", lambda: False)
        report, model, artifact = _export(root)
    card = ModelCard(CARD_CFG, report).add_metrics({"auroc": 0.91}).add_custom("note", "test")
    card.write(str(artifact / "model_card.json"))
    return SimpleNamespace(dir=artifact, report=report, model=model, ckpt=root / "model.ckpt")


def _reference(model, bag: np.ndarray):
    """PyTorch logits / probabilities / embedding for one [N, D] bag."""
    with torch.no_grad():
        out = model(torch.from_numpy(bag).unsqueeze(0))
    logits = out.logits[0]
    return logits.numpy(), torch.softmax(logits, dim=-1).numpy(), out.slide_embedding[0].numpy()


def _bag(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, IN_DIM)).astype(np.float32)


def _reuse_onnx_export(monkeypatch, abmil_artifact, tamper=None) -> None:
    """Make Exporter._export_onnx copy the module artifact's ONNX graph (optionally
    tampered with) instead of exporting again.

    The first dynamo ONNX export in a process costs ~15 s of warm-up and later
    ones ~1-2 s, so the default run exports once (``abmil_artifact``). The copy
    matches the checkpoint ``_export`` writes: same architecture and seed.
    """

    def copy_export(self, model, path):
        proto = onnx.load(str(abmil_artifact.dir / "model.onnx"))  # external data inlined
        if tamper is not None:
            tamper(proto)
        onnx.save(proto, str(path))
        return {"path": str(path), "status": "success"}

    monkeypatch.setattr(Exporter, "_export_onnx", copy_export)


def _register(monkeypatch, name: str, cls) -> None:
    """Make a test-only aggregator loadable by MILTrainModule (and the exporter)."""
    monkeypatch.setitem(model_registry._AGGREGATOR_REGISTRY, name, cls)


class _BagSizeBranchMIL(BaseMIL):
    """Mean-pools large bags and max-pools small ones, deciding in Python.

    Tracing evaluates the branch once, for the exporter's example bag of 100
    patches, so the TorchScript artifact always mean-pools and disagrees with
    PyTorch on bags of 50 patches or fewer.
    """

    def __init__(self, in_dim: int = IN_DIM, embed_dim: int = 8, **kwargs):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim)
        self.proj = nn.Linear(in_dim, embed_dim)

    def forward_features(self, h, mask=None, coords=None, return_attention=False):
        z = self.proj(h)
        pooled = z.mean(dim=1) if h.shape[1] > 50 else z.amax(dim=1)
        return pooled, {}


class _TracedBagReshapeMIL(BaseMIL):
    """Reshapes with the bag size read as a Python int, so the traced graph only
    accepts the example bag size and raises on any other."""

    def __init__(self, in_dim: int = IN_DIM, embed_dim: int = 8, **kwargs):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim)
        self.proj = nn.Linear(in_dim, embed_dim)

    def forward_features(self, h, mask=None, coords=None, return_attention=False):
        n = int(h.shape[1])
        z = self.proj(h).reshape(-1, n, 2, self.embed_dim // 2)
        return z.flatten(2).mean(dim=1), {}


# ═════════════════════════════════════════════════════════════════════════════
# Exporter
# ═════════════════════════════════════════════════════════════════════════════


class TestExportReport:
    def test_every_check_passes(self, abmil_artifact):
        report = abmil_artifact.report
        assert report["success"] is True
        assert report["exports"]["onnx"]["status"] == "success"
        assert report["exports"]["torchscript"]["status"] == "success"
        assert report["validation"]["portability"]["status"] == "pass"
        for name in ("onnx_numerical", "torchscript_numerical"):
            result = report["validation"][name]
            assert result["status"] == "pass"
            assert [t["n_patches"] for t in result["tests"]] == BAG_SIZES
            assert all(t["status"] == "pass" for t in result["tests"])
            assert all(t["max_logit_diff"] <= 1e-5 for t in result["tests"])

    def test_provenance_fields(self, abmil_artifact):
        report = abmil_artifact.report
        digest = hashlib.sha256(abmil_artifact.ckpt.read_bytes()).hexdigest()
        assert report["checkpoint_hash"] == digest
        assert report["model_params"] == sum(p.numel() for p in abmil_artifact.model.parameters())
        assert (report["in_dim"], report["num_classes"]) == (IN_DIM, NUM_CLASSES)

    def test_artifact_layout(self, abmil_artifact):
        names = {p.name for p in abmil_artifact.dir.iterdir()}
        expected = {"model.onnx", "model.pt", "model.ckpt", "export_report.json", "model_card.json"}
        assert expected <= names
        saved = json.loads((abmil_artifact.dir / "export_report.json").read_text())
        assert saved["success"] is True
        assert saved["validation"].keys() == abmil_artifact.report["validation"].keys()
        # The fallback checkpoint is a byte-for-byte copy.
        assert (abmil_artifact.dir / "model.ckpt").read_bytes() == abmil_artifact.ckpt.read_bytes()

    def test_onnx_graph_has_dynamic_batch_and_bag_axes(self, abmil_artifact):
        session = ort.InferenceSession(
            str(abmil_artifact.dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        (features,) = session.get_inputs()
        assert features.name == "features"
        assert isinstance(features.shape[0], str)  # symbolic batch axis
        assert isinstance(features.shape[1], str)  # symbolic n_patches axis
        assert features.shape[2] == IN_DIM
        assert [o.name for o in session.get_outputs()] == ["logits", "probabilities", "embedding"]

    @pytest.mark.parametrize("n_patches", [2, 63, 1000])
    def test_artifacts_match_pytorch_on_unseen_bag_sizes(self, abmil_artifact, n_patches):
        bag = _bag(n_patches, seed=n_patches)
        logits, probs, embedding = _reference(abmil_artifact.model, bag)
        session = ort.InferenceSession(
            str(abmil_artifact.dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        onnx_out = session.run(None, {"features": bag[None]})
        ts = torch.jit.load(str(abmil_artifact.dir / "model.pt"), map_location="cpu")
        with torch.no_grad():
            ts_out = [t.numpy() for t in ts(torch.from_numpy(bag[None]))]
        for out in (onnx_out, ts_out):
            np.testing.assert_allclose(out[0][0], logits, atol=1e-5, rtol=1e-4)
            np.testing.assert_allclose(out[1][0], probs, atol=1e-6, rtol=1e-4)
            np.testing.assert_allclose(out[2][0], embedding, atol=1e-5, rtol=1e-4)

    def test_onnx_handles_a_batch_of_bags(self, abmil_artifact):
        bags = np.stack([_bag(40, seed=s) for s in range(3)])
        session = ort.InferenceSession(
            str(abmil_artifact.dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        logits = session.run(None, {"features": bags})[0]
        assert logits.shape == (3, NUM_CLASSES)
        for i in range(3):
            np.testing.assert_allclose(
                logits[i], _reference(abmil_artifact.model, bags[i])[0], atol=1e-5, rtol=1e-4
            )

    @pytest.mark.slow  # one dynamo ONNX export each (2-12 s on CPU)
    @pytest.mark.parametrize(
        "arch, model_cfg",
        [
            ("static", {"embed_dim": 8, "dropout": 0.0}),
            ("mhabmil", {"embed_dim": 8, "num_heads": 2, "attn_dim": 8, "dropout": 0.0}),
            ("vit", {"embed_dim": 8, "dropout": 0.0}),
        ],
    )
    def test_other_cpu_aggregators_export_with_parity(self, tmp_path, arch, model_cfg):
        report, _, _ = _export(tmp_path, arch, model_cfg)
        assert report["success"] is True, report["validation"]
        assert report["validation"]["onnx_numerical"]["status"] == "pass"
        assert report["validation"]["torchscript_numerical"]["status"] == "pass"

    def test_skip_validation_exports_without_checks(self, tmp_path):
        report, _, artifact = _export(tmp_path, formats=["torchscript"], skip_validation=True)
        assert report["exports"]["torchscript"]["status"] == "success"
        assert "onnx" not in report["exports"]
        assert report["validation"] == {}
        assert report["success"] is True  # nothing failed; nothing was checked either
        assert (artifact / "model.pt").is_file()
        assert not (artifact / "model.onnx").exists()


class TestExportFailures:
    """The exporter must not mark an artifact successful unless every requested
    format exported and its parity check ran and passed."""

    def test_untampered_onnx_copy_passes(self, tmp_path, monkeypatch, abmil_artifact):
        """Control for the tampering test: the reused graph matches the checkpoint."""
        _reuse_onnx_export(monkeypatch, abmil_artifact)
        report, _, _ = _export(tmp_path, formats=["onnx"])
        assert report["validation"]["onnx_numerical"]["status"] == "pass"
        assert report["success"] is True

    def test_tampered_onnx_fails_the_parity_check(self, tmp_path, monkeypatch, abmil_artifact):
        def shift_classifier_bias(proto):
            for init in proto.graph.initializer:
                if init.name.endswith("head.bias"):
                    tampered = numpy_helper.to_array(init) + np.float32(0.5)
                    init.CopyFrom(numpy_helper.from_array(tampered, init.name))
                    return
            pytest.fail("classifier bias not found among the ONNX initializers")

        _reuse_onnx_export(monkeypatch, abmil_artifact, tamper=shift_classifier_bias)
        report, _, _ = _export(tmp_path, formats=["onnx"])
        assert report["exports"]["onnx"]["status"] == "success"
        result = report["validation"]["onnx_numerical"]
        assert result["status"] == "fail"
        assert all(t["status"] == "fail" for t in result["tests"])
        assert all(t["max_logit_diff"] == pytest.approx(0.5, abs=1e-4) for t in result["tests"])
        assert report["success"] is False

    def test_tampered_torchscript_fails_the_parity_check(self, tmp_path, monkeypatch):
        export_ts = Exporter._export_torchscript

        def export_then_tamper(self, model, path):
            result = export_ts(self, model, path)
            scripted = torch.jit.load(str(path))
            with torch.no_grad():
                scripted.model.head.bias.add_(0.5)
            scripted.save(str(path))
            return result

        monkeypatch.setattr(Exporter, "_export_torchscript", export_then_tamper)
        report, _, _ = _export(tmp_path, formats=["torchscript"])
        result = report["validation"]["torchscript_numerical"]
        assert result["status"] == "fail"
        assert all(t["max_logit_diff"] == pytest.approx(0.5, abs=1e-4) for t in result["tests"])
        assert report["success"] is False

    def test_trace_specialised_to_one_bag_size_fails_the_parity_check(self, tmp_path, monkeypatch):
        _register(monkeypatch, "_bag_size_branch", _BagSizeBranchMIL)
        report, _, _ = _export(
            tmp_path,
            "_bag_size_branch",
            {"embed_dim": 8},
            formats=["torchscript"],
            validation_n_patches=[10, 100, 500],
        )
        result = report["validation"]["torchscript_numerical"]
        assert [t["status"] for t in result["tests"]] == ["fail", "pass", "pass"]
        assert result["status"] == "fail"
        assert report["success"] is False

    def test_failed_onnx_export_is_reported_and_not_validated(self, tmp_path, monkeypatch):
        def broken_export(*args, **kwargs):
            raise RuntimeError("unsupported operator")

        monkeypatch.setattr(torch.onnx, "export", broken_export)
        report, _, artifact = _export(tmp_path)
        assert report["exports"]["onnx"] == {"status": "failed", "error": "unsupported operator"}
        assert "onnx_numerical" not in report["validation"]
        assert report["validation"]["torchscript_numerical"]["status"] == "pass"
        assert report["success"] is False
        assert json.loads((artifact / "export_report.json").read_text())["success"] is False

    def test_failed_torchscript_export_is_reported_and_not_validated(self, tmp_path, monkeypatch):
        def broken_trace(*args, **kwargs):
            raise RuntimeError("cannot trace")

        monkeypatch.setattr(torch.jit, "trace", broken_trace)
        report, _, _ = _export(tmp_path, formats=["torchscript"])
        assert report["exports"]["torchscript"]["status"] == "failed"
        assert "torchscript_numerical" not in report["validation"]
        assert report["success"] is False

    def test_parity_check_that_cannot_run_is_not_a_pass(
        self, tmp_path, monkeypatch, abmil_artifact
    ):
        _reuse_onnx_export(monkeypatch, abmil_artifact)
        monkeypatch.setitem(sys.modules, "onnxruntime", None)  # import now fails
        report, _, _ = _export(tmp_path, formats=["onnx"])
        assert report["exports"]["onnx"]["status"] == "success"
        assert report["validation"]["onnx_numerical"]["status"] == "skipped"
        assert report["success"] is False

    def test_artifact_that_raises_at_another_bag_size_is_reported(self, tmp_path, monkeypatch):
        _register(monkeypatch, "_traced_bag_reshape", _TracedBagReshapeMIL)
        report, _, _ = _export(
            tmp_path,
            "_traced_bag_reshape",
            {"embed_dim": 8},
            formats=["torchscript"],
            validation_n_patches=[10],
        )
        assert report["validation"]["torchscript_numerical"]["status"] == "fail"
        assert report["success"] is False


def test_config_fingerprint_is_stable_and_order_independent():
    a = {"model": {"arch": "abmil", "embed_dim": 8}, "lr": 1e-3}
    b = {"lr": 1e-3, "model": {"embed_dim": 8, "arch": "abmil"}}
    assert config_fingerprint(a) == config_fingerprint(b)
    assert len(config_fingerprint(a)) == 8
    assert config_fingerprint(a) != config_fingerprint({**a, "lr": 1e-4})


# ═════════════════════════════════════════════════════════════════════════════
# Model card
# ═════════════════════════════════════════════════════════════════════════════


class TestModelCard:
    def test_card_written_next_to_the_artifact(self, abmil_artifact):
        card = json.loads((abmil_artifact.dir / "model_card.json").read_text())
        assert card["schema_version"] == "1.0"
        assert card["experiment_name"] == "tiny_abmil"
        assert card["config_fingerprint"] == config_fingerprint(CARD_CFG)
        assert card["model"]["architecture"] == "abmil"
        assert card["model"]["in_dim"] == IN_DIM
        assert card["model"]["total_params"] == abmil_artifact.report["model_params"]
        assert card["export"] == {
            "checkpoint_hash": abmil_artifact.report["checkpoint_hash"],
            "formats": ["onnx", "torchscript", "checkpoint"],
            "validation_passed": True,
        }
        assert card["serving"]["input_schema"]["features"]["shape"] == ["batch", "n_patches", 16]
        assert card["metrics"] == {"auroc": 0.91}
        assert card["custom"] == {"note": "test"}

    def test_missing_config_fields_become_none(self):
        card = ModelCard({}).build()
        assert card["experiment_name"] == "unknown"
        assert card["model"]["architecture"] == "unknown"
        assert card["encoder"]["feature_dim"] is None
        assert card["export"]["validation_passed"] is None
        assert "custom" not in card

    def test_metric_and_threshold_loaders(self, tmp_path):
        assert load_oof_metrics(str(tmp_path)) == {}
        (tmp_path / "eval").mkdir()
        (tmp_path / "eval" / "evaluation_report.json").write_text(
            json.dumps({"metrics": {"auroc": 0.8}})
        )
        assert load_oof_metrics(str(tmp_path)) == {"auroc": 0.8}
        (tmp_path / "oof_metrics.json").write_text(json.dumps({"auroc": 0.85}))
        assert load_oof_metrics(str(tmp_path)) == {"auroc": 0.85}
        assert load_threshold_info(str(tmp_path)) == {}
        (tmp_path / "threshold_analysis.json").write_text(json.dumps({"optimal_f1": 0.4}))
        assert load_threshold_info(str(tmp_path)) == {"optimal_f1": 0.4}


# ═════════════════════════════════════════════════════════════════════════════
# Inference backend and FastAPI app
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def clients(abmil_artifact):
    """One TestClient per backend. Server errors come back as 500 responses."""
    return {
        backend: TestClient(
            create_app(str(abmil_artifact.dir), backend=backend), raise_server_exceptions=False
        )
        for backend in BACKENDS
    }


def _post_raw(client: TestClient, path: str, payload: dict):
    """POST JSON that may contain NaN / Infinity tokens (Python's json emits them)."""
    return client.post(
        path, content=json.dumps(payload), headers={"content-type": "application/json"}
    )


class TestInferenceBackend:
    def test_auto_prefers_onnx_then_torchscript_then_checkpoint(self, abmil_artifact, tmp_path):
        artifact = tmp_path / "artifact"
        shutil.copytree(abmil_artifact.dir, artifact)
        assert InferenceBackend(str(artifact)).backend_name == "onnx"
        (artifact / "model.onnx").unlink()
        assert InferenceBackend(str(artifact)).backend_name == "torchscript"
        (artifact / "model.pt").unlink()
        assert InferenceBackend(str(artifact)).backend_name == "pytorch"
        (artifact / "model.ckpt").unlink()
        with pytest.raises(FileNotFoundError, match="No loadable model"):
            InferenceBackend(str(artifact))

    def test_unknown_backend_rejected(self, abmil_artifact):
        with pytest.raises(ValueError, match="Unknown backend"):
            InferenceBackend(str(abmil_artifact.dir), backend="tensorrt")

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_expected_feature_dim(self, abmil_artifact, backend):
        assert InferenceBackend(str(abmil_artifact.dir), backend=backend).in_dim == IN_DIM

    def test_feature_dim_unknown_without_a_card_for_torchscript(self, abmil_artifact, tmp_path):
        artifact = tmp_path / "artifact"
        artifact.mkdir()
        shutil.copy(abmil_artifact.dir / "model.pt", artifact / "model.pt")
        assert InferenceBackend(str(artifact), backend="torchscript").in_dim is None

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_predict_matches_pytorch(self, abmil_artifact, backend):
        engine = InferenceBackend(str(abmil_artifact.dir), backend=backend)
        bag = _bag(33, seed=1)
        logits, probs, embedding = _reference(abmil_artifact.model, bag)
        out = engine.predict(bag)
        np.testing.assert_allclose(out["logits"], logits, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(out["probabilities"], probs, atol=1e-6, rtol=1e-4)
        np.testing.assert_allclose(out["embedding"], embedding, atol=1e-5, rtol=1e-4)
        assert out["pred_class"] == int(np.argmax(probs))
        assert out["pred_prob"] == pytest.approx(float(probs.max()), abs=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
class TestServerEndpoints:
    def test_health(self, clients, backend):
        resp = clients[backend].get("/health")
        assert resp.status_code == 200
        assert resp.json() == {
            "status": "healthy",
            "backend": backend,
            "experiment": "tiny_abmil",
            "model_arch": "abmil",
            "validation_passed": True,
        }

    def test_model_card(self, clients, backend, abmil_artifact):
        resp = clients[backend].get("/model-card")
        assert resp.status_code == 200
        assert resp.json() == json.loads((abmil_artifact.dir / "model_card.json").read_text())

    @pytest.mark.parametrize("n_patches", [1, 40])
    def test_predict_valid_bag(self, clients, backend, abmil_artifact, n_patches):
        bag = _bag(n_patches, seed=n_patches)
        resp = clients[backend].post("/predict", json={"features": bag.tolist(), "slide_id": "s1"})
        assert resp.status_code == 200
        body = resp.json()
        logits, probs, _ = _reference(abmil_artifact.model, bag)
        assert body["slide_id"] == "s1"
        assert body["n_patches"] == n_patches
        np.testing.assert_allclose(body["logits"], logits, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(body["probabilities"], probs, atol=1e-6, rtol=1e-4)
        assert sum(body["probabilities"]) == pytest.approx(1.0, abs=1e-5)
        assert body["pred_class"] == int(np.argmax(probs))
        assert body["pred_prob"] == pytest.approx(float(probs.max()), abs=1e-6)
        assert body["latency_ms"] >= 0

    def test_predict_wrong_feature_dim(self, clients, backend):
        resp = clients[backend].post("/predict", json={"features": _bag(5)[:, :-1].tolist()})
        assert resp.status_code == 400
        assert "expected feature dim 16, got 15" in resp.json()["detail"]

    @pytest.mark.parametrize("features", [[], [[]], [[], []]])
    def test_predict_empty_bag(self, clients, backend, features):
        resp = clients[backend].post("/predict", json={"features": features})
        assert resp.status_code == 400
        assert "empty bag" in resp.json()["detail"]

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_predict_non_finite_values(self, clients, backend, bad):
        bag = _bag(5).tolist()
        bag[2][3] = bad
        resp = _post_raw(clients[backend], "/predict", {"features": bag})
        assert resp.status_code == 400
        assert "NaN or infinite" in resp.json()["detail"]

    def test_predict_ragged_rows(self, clients, backend):
        bag = _bag(3).tolist()
        bag[1] = bag[1][:-2]
        resp = clients[backend].post("/predict", json={"features": bag})
        assert resp.status_code == 400
        assert "different lengths" in resp.json()["detail"]

    def test_predict_schema_violations_are_422(self, clients, backend):
        client = clients[backend]
        assert client.post("/predict", json={}).status_code == 422
        assert client.post("/predict", json={"features": [[["x"]]]}).status_code == 422
        assert client.post("/predict", json={"features": [1.0, 2.0]}).status_code == 422

    def test_predict_batch(self, clients, backend, abmil_artifact):
        bags = {"a": _bag(12, seed=1), "b": _bag(70, seed=2)}
        payload = {"slides": [{"features": b.tolist(), "slide_id": k} for k, b in bags.items()]}
        resp = clients[backend].post("/predict/batch", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert [r["slide_id"] for r in body] == ["a", "b"]
        assert [r["n_patches"] for r in body] == [12, 70]
        for r in body:
            single = clients[backend].post(
                "/predict", json={"features": bags[r["slide_id"]].tolist()}
            )
            np.testing.assert_allclose(r["logits"], single.json()["logits"], atol=1e-6)
            np.testing.assert_allclose(
                r["logits"], _reference(abmil_artifact.model, bags[r["slide_id"]])[0], atol=1e-5
            )

    def test_predict_batch_rejects_a_bad_slide(self, clients, backend):
        payload = {
            "slides": [
                {"features": _bag(4).tolist(), "slide_id": "ok"},
                {"features": _bag(4)[:, :8].tolist(), "slide_id": "short"},
            ]
        }
        resp = clients[backend].post("/predict/batch", json=payload)
        assert resp.status_code == 400
        assert resp.json()["detail"].startswith("Slide short:")


def test_backends_agree(clients):
    """ONNX, TorchScript and the PyTorch checkpoint give the same predictions."""
    bags = [_bag(n, seed=n) for n in (1, 9, 128, 777)]
    results = {
        backend: [
            clients[backend].post("/predict", json={"features": bag.tolist()}).json()
            for bag in bags
        ]
        for backend in BACKENDS
    }
    for backend in ("torchscript", "pytorch"):
        for ref, other in zip(results["onnx"], results[backend], strict=True):
            np.testing.assert_allclose(other["logits"], ref["logits"], atol=1e-5, rtol=1e-4)
            np.testing.assert_allclose(
                other["probabilities"], ref["probabilities"], atol=1e-6, rtol=1e-4
            )
            assert other["pred_class"] == ref["pred_class"]


def test_model_card_endpoint_404_without_a_card(abmil_artifact, tmp_path):
    artifact = tmp_path / "artifact"
    shutil.copytree(abmil_artifact.dir, artifact)
    (artifact / "model_card.json").unlink()
    client = TestClient(create_app(str(artifact), backend="torchscript"))
    assert client.get("/model-card").status_code == 404
    health = client.get("/health").json()
    assert health["experiment"] == "unknown"
    assert health["validation_passed"] is None
    assert client.post("/predict", json={"features": _bag(3).tolist()}).status_code == 200


def test_health_reports_a_failed_export(tmp_path, monkeypatch):
    """An artifact whose parity check failed is served, but /health says so."""
    _register(monkeypatch, "_bag_size_branch", _BagSizeBranchMIL)
    report, _, artifact = _export(
        tmp_path,
        "_bag_size_branch",
        {"embed_dim": 8},
        formats=["torchscript"],
        validation_n_patches=[10, 100],
    )
    ModelCard({"exp_name": "branchy"}, report).write(str(artifact / "model_card.json"))
    client = TestClient(create_app(str(artifact), backend="torchscript"))
    assert client.get("/health").json()["validation_passed"] is False


def test_single_logit_model_serves_sigmoid_as_two_classes(tmp_path):
    """A BCE head emits one logit per slide; a softmax over it would run across the batch
    and always say 1.0, so the export wrappers apply a sigmoid instead."""
    ckpt = tmp_path / "model.ckpt"
    model = _write_checkpoint(ckpt, num_classes=1)
    report = Exporter(
        checkpoint_path=str(ckpt),
        output_dir=str(tmp_path / "artifact"),
        in_dim=IN_DIM,
        num_classes=1,
        validation_n_patches=[1, 17],
    ).run()
    assert report["success"] is True
    bag = _bag(9, seed=4)
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(bag).unsqueeze(0)).logits).item()
    for backend in BACKENDS:
        client = TestClient(create_app(str(tmp_path / "artifact"), backend=backend))
        result = client.post("/predict", json={"features": bag.tolist()}).json()
        np.testing.assert_allclose(result["probabilities"], [1 - p, p], atol=1e-5)
        assert len(result["logits"]) == 1
        assert result["pred_class"] == int(p > 0.5)


# ═════════════════════════════════════════════════════════════════════════════
# POST /predict/npy
# ═════════════════════════════════════════════════════════════════════════════


def _npy(array: np.ndarray, allow_pickle: bool = False) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=allow_pickle)
    return buf.getvalue()


@pytest.mark.parametrize("backend", BACKENDS)
class TestPredictNpy:
    def test_matches_json_predict(self, clients, backend):
        bag = _bag(33, seed=3)
        expected = clients[backend].post("/predict", json={"features": bag.tolist()}).json()
        result = clients[backend].post("/predict/npy?slide_id=s7", content=_npy(bag)).json()
        assert result["slide_id"] == "s7"
        assert result["n_patches"] == 33
        np.testing.assert_allclose(result["probabilities"], expected["probabilities"], atol=1e-6)
        assert result["pred_class"] == expected["pred_class"]

    def test_float64_is_accepted(self, clients, backend):
        resp = clients[backend].post("/predict/npy", content=_npy(_bag(5).astype(np.float64)))
        assert resp.status_code == 200

    @pytest.mark.parametrize("body", [b"", b"not an array"])
    def test_body_that_is_not_npy(self, clients, backend, body):
        resp = clients[backend].post("/predict/npy", content=body)
        assert resp.status_code == 400
        assert "not a .npy array" in resp.json()["detail"]

    def test_pickled_object_array_is_refused(self, clients, backend):
        body = _npy(np.array([{"a": 1}], dtype=object), allow_pickle=True)
        resp = clients[backend].post("/predict/npy", content=body)
        assert resp.status_code == 400

    def test_integer_array_is_refused(self, clients, backend):
        resp = clients[backend].post(
            "/predict/npy", content=_npy(np.ones((4, IN_DIM), dtype=np.int64))
        )
        assert resp.status_code == 400
        assert "float" in resp.json()["detail"]

    @pytest.mark.parametrize(
        ("array", "message"),
        [
            (np.zeros((5, IN_DIM - 1), dtype=np.float32), "expected feature dim 16, got 15"),
            (np.zeros((0, IN_DIM), dtype=np.float32), "empty bag"),
            (np.zeros(IN_DIM, dtype=np.float32), "expected 2D"),
            (np.full((3, IN_DIM), np.nan, dtype=np.float32), "NaN or infinite"),
        ],
    )
    def test_bag_the_model_cannot_take(self, clients, backend, array, message):
        resp = clients[backend].post("/predict/npy", content=_npy(array))
        assert resp.status_code == 400
        assert message in resp.json()["detail"]
