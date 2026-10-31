"""
Tests for the serving entry points: ``python -m oceanpath.serving``
(``oceanpath.serving.__main__``) and the random-weight demo artifact
(``oceanpath.serving.demo``).

No server is started: ``uvicorn.run`` is replaced by a recorder and the app it
was handed is exercised through FastAPI's TestClient. The demo artifact is
exported to TorchScript only (``_TorchScriptOnly``); test_serving.py covers the
ONNX path, whose first dynamo export in a process costs ~15 s.

Covers:
  * export_random_model: artifact layout and model card, the temporary
    checkpoint removed, weights reproducible per seed, the artifact served
  * main(): flags and environment defaults reach create_app and uvicorn.run;
    usage errors exit without starting a server
"""

import json

import numpy as np
import pytest
import torch

pytest.importorskip("fastapi", reason="serving tests need the 'serve' extra")
uvicorn = pytest.importorskip("uvicorn", reason="serving tests need the 'serve' extra")

from fastapi.testclient import TestClient  # noqa: E402

from oceanpath.serving import demo, server  # noqa: E402
from oceanpath.serving.__main__ import main  # noqa: E402
from oceanpath.serving.exporter import Exporter  # noqa: E402

IN_DIM = 16


class _TorchScriptOnly(Exporter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, formats=["torchscript"], **kwargs)


@pytest.fixture(scope="module")
def demo_artifacts(tmp_path_factory):
    """Seed 0 exported twice and seed 1 once, as export_random_model writes them."""
    root = tmp_path_factory.mktemp("demo")
    reports = {}
    with pytest.MonkeyPatch.context() as mp, torch.random.fork_rng():  # it seeds torch
        mp.setattr(demo, "Exporter", _TorchScriptOnly)
        mp.setattr(torch.cuda, "is_available", lambda: False)
        for name, seed in (("seed0", 0), ("seed0_again", 0), ("seed1", 1)):
            reports[name] = demo.export_random_model(
                root / name, in_dim=IN_DIM, seed=seed, validation_n_patches=[1, 9]
            )
    return root, reports


def _probabilities(artifact_dir, bag: np.ndarray) -> np.ndarray:
    client = TestClient(server.create_app(str(artifact_dir), backend="torchscript"))
    response = client.post("/predict", json={"features": bag.tolist()})
    assert response.status_code == 200
    return np.array(response.json()["probabilities"])


# ═════════════════════════════════════════════════════════════════════════════
# Demo artifact
# ═════════════════════════════════════════════════════════════════════════════


class TestDemoArtifact:
    def test_artifact_layout_and_model_card(self, demo_artifacts):
        root, reports = demo_artifacts
        report = reports["seed0"]
        assert report["success"] is True
        assert report["validation"]["torchscript_numerical"]["status"] == "pass"
        assert (report["in_dim"], report["num_classes"]) == (IN_DIM, 2)
        artifact = root / "seed0"
        names = {p.name for p in artifact.iterdir()}
        assert {"model.pt", "model.ckpt", "export_report.json", "model_card.json"} <= names
        # The random-init checkpoint is written next to the artifact, then removed.
        assert not list(root.glob(".*.random-init.ckpt"))

        card = json.loads((artifact / "model_card.json").read_text())
        assert card["experiment_name"] == "demo_random_abmil"
        assert card["model"]["architecture"] == "abmil"
        assert card["model"]["in_dim"] == IN_DIM
        assert card["export"]["validation_passed"] is True
        assert "note" in card["custom"]

    def test_weights_are_reproducible_per_seed(self, demo_artifacts):
        root, _ = demo_artifacts
        weights = {
            name: torch.load(root / name / "model.ckpt", weights_only=False)["state_dict"]
            for name in ("seed0", "seed0_again", "seed1")
        }
        assert weights["seed0"].keys() == weights["seed1"].keys()
        for key, value in weights["seed0"].items():
            assert torch.equal(value, weights["seed0_again"][key]), key
        assert not all(torch.equal(v, weights["seed1"][k]) for k, v in weights["seed0"].items())

    def test_artifact_serves_two_class_probabilities(self, demo_artifacts):
        root, _ = demo_artifacts
        bag = np.random.default_rng(0).standard_normal((12, IN_DIM)).astype(np.float32)
        probs = {name: _probabilities(root / name, bag) for name in ("seed0", "seed1")}
        assert probs["seed0"].shape == (2,)
        assert probs["seed0"].sum() == pytest.approx(1.0, abs=1e-6)
        assert not np.allclose(probs["seed0"], probs["seed1"])


# ═════════════════════════════════════════════════════════════════════════════
# python -m oceanpath.serving
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def uvicorn_run(monkeypatch):
    """Record what main() would serve instead of starting a server."""
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))
    return calls


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv("OCEANPATH_ARTIFACT_DIR", raising=False)
    monkeypatch.delenv("OCEANPATH_THREADS", raising=False)


class TestCommandLine:
    def test_serves_the_artifact_on_the_given_address(self, demo_artifacts, uvicorn_run, no_env):
        root, _ = demo_artifacts
        main(
            [
                "--artifact-dir",
                str(root / "seed0"),
                "--backend",
                "torchscript",
                "--host",
                "0.0.0.0",
                "--port",
                "9100",
            ]
        )
        ((app, kwargs),) = uvicorn_run
        assert kwargs == {"host": "0.0.0.0", "port": 9100, "log_level": "info"}
        client = TestClient(app)
        health = client.get("/health").json()
        assert (health["backend"], health["experiment"]) == ("torchscript", "demo_random_abmil")
        bag = np.zeros((3, IN_DIM), np.float32)
        assert client.post("/predict", json={"features": bag.tolist()}).status_code == 200

    def test_flags_and_environment_defaults_reach_create_app(
        self, monkeypatch, uvicorn_run, no_env
    ):
        created = []
        monkeypatch.setattr(server, "create_app", lambda **kwargs: created.append(kwargs) or "app")

        main(["--artifact-dir", "/a", "--backend", "onnx", "--device", "cuda:1", "--threads", "3"])
        monkeypatch.setenv("OCEANPATH_ARTIFACT_DIR", "/from/env")
        monkeypatch.setenv("OCEANPATH_THREADS", "2")
        main([])

        assert created == [
            {"artifact_dir": "/a", "backend": "onnx", "device": "cuda:1", "num_threads": 3},
            {"artifact_dir": "/from/env", "backend": "auto", "device": "cpu", "num_threads": 2},
        ]
        assert [kwargs for _, kwargs in uvicorn_run] == [
            {"host": "127.0.0.1", "port": 8000, "log_level": "info"}
        ] * 2

    @pytest.mark.parametrize(
        "argv",
        [
            [],
            ["--artifact-dir", "/a", "--backend", "tensorrt"],
            ["--artifact-dir", "/a", "--port", "x"],
        ],
        ids=["no_artifact_dir", "unknown_backend", "bad_port"],
    )
    def test_usage_errors_exit_without_serving(self, argv, uvicorn_run, no_env, capsys):
        with pytest.raises(SystemExit) as exit_info:
            main(argv)
        assert exit_info.value.code == 2
        assert uvicorn_run == []
        assert "usage: python -m oceanpath.serving" in capsys.readouterr().err
