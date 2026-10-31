"""
Tests for oceanpath.eval.umap_viz — UMAP views of slide embeddings (Stage 7).

Importing umap-learn and running its first fit costs ~15 s on CPU (numba JIT),
so the default run checks compute_umap / plot_train_test_umap's own work
(standardise -> PCA cap -> one joint projection, parameters passed through,
train/test legend) with a stand-in reducer, and the real UMAP fit runs only
under the ``slow`` marker.

Covers:
  - load_oof_embeddings / load_test_embeddings on the embeddings_{val,test}.npz
    files MILTrainModule.save_embeddings writes per fold: OOF concatenation, test
    embeddings averaged over folds, missing folds, nothing found
  - plot_umap on precomputed coordinates: class names (partial mapping, auto,
    named classes without slides and unlabeled slides left out), per-class counts
    and colours, highlighted slides, SVM decision regions (none under ten slides)
  - save_umap_data: Parquet round trip with metadata joined on slide_id
  - compute_umap / plot_train_test_umap with the stand-in reducer: PCA capped at
    N - 1 components, parameters passed through, one joint train+test fit, legend
    counts per class and split
  - compute_umap and plot_train_test_umap with the real umap-learn (slow)
"""

import sys
import types

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from matplotlib.colors import to_rgb
from matplotlib.contour import ContourSet
from PIL import Image

from oceanpath.eval.umap_viz import (
    compute_umap,
    load_oof_embeddings,
    load_test_embeddings,
    plot_train_test_umap,
    plot_umap,
    save_umap_data,
)
from oceanpath.models import MILOutput
from oceanpath.modules.train_module import MILTrainModule

EMBED_DIM = 16


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _save_fold_embeddings(fold_dir, prefix: str, slide_ids: list[str], embeddings: np.ndarray):
    """Write embeddings_{prefix}.npz through the training module's own accumulator."""
    module = MILTrainModule(
        arch="abmil", in_dim=4, num_classes=2, model_cfg={"embed_dim": 4, "attn_dim": 4}
    )
    acc = module._val_embeddings if prefix == "val" else module._test_embeddings
    output = MILOutput(slide_embedding=torch.from_numpy(embeddings), logits=None, extras={})
    module._accumulate_embeddings(acc, output, {"slide_ids": slide_ids})
    fold_dir.mkdir(parents=True, exist_ok=True)
    assert module.save_embeddings(str(fold_dir), prefix=prefix) is not None


def _clusters(n_per_class: int = 8, n_classes: int = 2, seed: int = 0):
    """Well-separated Gaussian clusters in EMBED_DIM dimensions."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 6, (n_classes, EMBED_DIM))
    labels = np.repeat(np.arange(n_classes), n_per_class)
    return centers[labels] + rng.normal(0, 1, (len(labels), EMBED_DIM)), labels


def _assert_png(path):
    assert path.is_file()
    with Image.open(path) as img:
        img.verify()
    assert path.stat().st_size > 1000


@pytest.fixture(autouse=True, scope="module")
def _headless_backend():
    """plot_umap draws with whatever backend is active; never open a window."""
    matplotlib.use("Agg")


@pytest.fixture
def axes(monkeypatch):
    """Axes created by the plotting code, readable after the figure is saved."""
    created = []
    real_subplots = plt.subplots

    def spy(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        created.append(ax)
        return fig, ax

    monkeypatch.setattr(plt, "subplots", spy)
    return created


@pytest.fixture
def reducer_calls(monkeypatch):
    """Replace umap.umap_.UMAP with a reducer that records what it was given and
    returns the first two input columns."""
    calls = []

    class RecordingUMAP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls.append(self)

        def fit_transform(self, x):
            self.x = np.asarray(x)
            return self.x[:, :2].copy()

    umap_module = types.ModuleType("umap.umap_")
    umap_module.UMAP = RecordingUMAP
    package = types.ModuleType("umap")
    package.umap_ = umap_module
    monkeypatch.setitem(sys.modules, "umap", package)
    monkeypatch.setitem(sys.modules, "umap.umap_", umap_module)
    return calls


def _legend_labels(ax) -> list[str]:
    return ax.get_legend_handles_labels()[1]


# ═════════════════════════════════════════════════════════════════════════════
# Loading the embeddings training wrote
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def train_dir(tmp_path_factory):
    """Three folds: OOF embeddings for folds 0 and 1 only, test embeddings for the
    same three slides from every fold."""
    root = tmp_path_factory.mktemp("train_run")
    rng = np.random.default_rng(0)
    oof = {
        0: (["s0", "s1", "s2"], rng.normal(size=(3, EMBED_DIM)).astype(np.float32)),
        1: (["s3", "s4"], rng.normal(size=(2, EMBED_DIM)).astype(np.float32)),
    }
    test_ids = ["t0", "t1", "t2"]
    test = {k: rng.normal(size=(3, EMBED_DIM)).astype(np.float32) for k in range(3)}
    for k, (ids, emb) in oof.items():
        _save_fold_embeddings(root / f"fold_{k}", "val", ids, emb)
    for k, emb in test.items():
        _save_fold_embeddings(root / f"fold_{k}", "test", test_ids, emb)
    return types.SimpleNamespace(path=root, oof=oof, test=test, test_ids=test_ids)


class TestLoadEmbeddings:
    def test_oof_embeddings_concatenate_the_folds(self, train_dir):
        df = load_oof_embeddings(str(train_dir.path), n_folds=3)  # fold 2 has none
        assert df["slide_id"].tolist() == ["s0", "s1", "s2", "s3", "s4"]
        assert df["fold"].tolist() == [0, 0, 0, 1, 1]
        assert all(isinstance(s, str) for s in df["slide_id"])
        stored = np.concatenate([train_dir.oof[0][1], train_dir.oof[1][1]])
        np.testing.assert_array_equal(np.stack(df["embedding"].to_numpy()), stored)

    def test_test_embeddings_are_averaged_over_folds(self, train_dir):
        df = load_test_embeddings(str(train_dir.path), n_folds=4)  # fold 3 does not exist
        assert df["slide_id"].tolist() == train_dir.test_ids
        assert (df["fold"] == -1).all()
        expected = np.mean([train_dir.test[k] for k in range(3)], axis=0)
        np.testing.assert_allclose(np.stack(df["embedding"].to_numpy()), expected, rtol=1e-6)

    def test_nothing_written_gives_none(self, tmp_path):
        assert load_oof_embeddings(str(tmp_path), n_folds=2) is None
        assert load_test_embeddings(str(tmp_path), n_folds=2) is None


# ═════════════════════════════════════════════════════════════════════════════
# Plotting precomputed coordinates
# ═════════════════════════════════════════════════════════════════════════════


class TestPlotUmap:
    def test_classes_counts_colours_and_highlights(self, tmp_path, axes):
        coords = np.arange(18, dtype=float).reshape(9, 2)
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 2, -1])  # -1: no label
        slide_ids = [f"s{i}" for i in range(9)]
        out = tmp_path / "umap" / "umap.png"
        plot_umap(
            coords,
            labels,
            str(out),
            class_names={0: "benign", 3: "absent"},  # 1 and 2 named automatically
            colors={0: "#ff0000"},
            highlight_slide_ids=["s1", "s5", "not_a_slide"],
            all_slide_ids=slide_ids,
            save_dpi=50,
        )
        _assert_png(out)
        (ax,) = axes
        handles, names = ax.get_legend_handles_labels()
        assert names == ["benign (n=4)", "1 (n=3)", "2 (n=1)", "Highlighted"]  # no "absent"
        assert tuple(handles[0].get_facecolor()[0][:3]) == pytest.approx(to_rgb("#ff0000"))
        np.testing.assert_array_equal(handles[-1].get_offsets(), coords[[1, 5]])
        drawn = sum(len(h.get_offsets()) for h in handles[:-1])
        assert drawn == 8  # the unlabeled slide is not drawn

    def test_decision_regions_need_enough_slides(self, tmp_path, axes):
        coords, labels = _clusters(n_per_class=8)
        coords = coords[:, :2]
        plot_umap(coords, labels, str(tmp_path / "a.png"), show_decision_boundary=True, save_dpi=50)
        plot_umap(  # both classes, but only eight slides: no regions
            coords[4:12],
            labels[4:12],
            str(tmp_path / "b.png"),
            show_decision_boundary=True,
            save_dpi=50,
        )
        enough, too_few = axes
        assert any(isinstance(c, ContourSet) for c in enough.collections)
        assert not any(isinstance(c, ContourSet) for c in too_few.collections)
        _assert_png(tmp_path / "a.png")
        _assert_png(tmp_path / "b.png")


def test_save_umap_data_round_trip(tmp_path):
    coords = np.array([[0.5, 1.5], [2.0, -1.0], [3.0, 0.0]])
    metadata = pd.DataFrame(
        {
            "slide_id": ["s0", "s2", "other"],
            "patient_id": ["P0", "P2", "PX"],
            "site": ["a", "b", "c"],
        }
    )
    out = tmp_path / "umap" / "umap.parquet"
    returned = save_umap_data(coords, ["s0", "s1", "s2"], np.array([1, 0, 1]), str(out), metadata)
    saved = pd.read_parquet(out)
    assert saved.columns.tolist() == ["slide_id", "umap_1", "umap_2", "label", "patient_id", "site"]
    core = ["slide_id", "umap_1", "umap_2", "label"]
    pd.testing.assert_frame_equal(saved[core], returned[core])
    assert saved["slide_id"].tolist() == ["s0", "s1", "s2"]  # left join keeps every slide
    np.testing.assert_array_equal(saved[["umap_1", "umap_2"]].to_numpy(), coords)
    assert saved["label"].tolist() == [1, 0, 1]
    assert saved["patient_id"].tolist()[0::2] == ["P0", "P2"]
    assert saved["site"].tolist()[0::2] == ["a", "b"]
    assert pd.isna(saved.loc[1, "patient_id"]) and pd.isna(saved.loc[1, "site"])


# ═════════════════════════════════════════════════════════════════════════════
# Projection wiring (stand-in reducer)
# ═════════════════════════════════════════════════════════════════════════════


class TestProjectionWiring:
    def test_compute_umap_feeds_a_capped_pca_to_the_reducer(self, reducer_calls):
        """12 slides in 16 dimensions: PCA keeps min(20, 16, 12 - 1) = 11 components."""
        embeddings, _ = _clusters(n_per_class=6)
        coords = compute_umap(embeddings, n_pca=20, n_neighbors=5, min_dist=0.3, seed=7)
        (reducer,) = reducer_calls
        assert reducer.kwargs == {
            "n_neighbors": 5,
            "min_dist": 0.3,
            "metric": "cosine",
            "random_state": 7,
        }
        assert reducer.x.shape == (12, 11)
        np.testing.assert_allclose(reducer.x.mean(0), 0, atol=1e-9)  # centred
        variances = reducer.x.var(0)
        assert np.all(np.diff(variances) <= 1e-9)  # principal components in order
        np.testing.assert_array_equal(coords, reducer.x[:, :2])

    def test_train_and_test_share_one_projection(self, tmp_path, reducer_calls, axes):
        """One joint fit over both splits; the legend counts each class per split
        and has no entry for a class the test set lacks."""
        train, train_labels = _clusters(n_per_class=5, seed=1)
        test, test_labels = _clusters(n_per_class=2, seed=2)
        test_labels = np.zeros_like(test_labels)  # no tumour in the test set
        out = tmp_path / "umap_train_test.png"
        plot_train_test_umap(
            train,
            train_labels,
            [f"tr{i}" for i in range(10)],
            test,
            test_labels,
            [f"te{i}" for i in range(4)],
            str(out),
            class_names={0: "benign", 1: "tumour"},
            n_pca=8,
            n_neighbors=4,
            save_dpi=50,
        )
        (reducer,) = reducer_calls  # one joint fit, not one per split
        assert reducer.x.shape == (14, 8)
        assert reducer.kwargs["n_neighbors"] == 4
        (ax,) = axes
        assert _legend_labels(ax) == [
            "benign (train, n=5)",
            "benign (test, n=4)",
            "tumour (train, n=5)",
        ]
        _assert_png(out)


# ═════════════════════════════════════════════════════════════════════════════
# Real UMAP (slow: umap-learn import + numba JIT)
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestRealUmap:
    def test_compute_umap_keeps_clusters_apart_and_is_seeded(self):
        embeddings, labels = _clusters(n_per_class=10, n_classes=3, seed=3)
        coords = compute_umap(embeddings, n_neighbors=8, seed=0)
        assert coords.shape == (30, 2)
        assert np.isfinite(coords).all()
        np.testing.assert_allclose(compute_umap(embeddings, n_neighbors=8, seed=0), coords)
        # Every slide's nearest neighbour on the map is from its own cluster.
        dist = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
        np.fill_diagonal(dist, np.inf)
        assert (labels[dist.argmin(1)] == labels).all()

    def test_plot_train_test_umap_writes_the_figure(self, tmp_path):
        train, train_labels = _clusters(n_per_class=8, seed=4)
        test, test_labels = _clusters(n_per_class=3, seed=5)
        out = tmp_path / "umap_train_test.png"
        plot_train_test_umap(
            train,
            train_labels,
            [f"tr{i}" for i in range(16)],
            test,
            test_labels,
            [f"te{i}" for i in range(6)],
            str(out),
            n_neighbors=5,
            save_dpi=50,
        )
        _assert_png(out)
