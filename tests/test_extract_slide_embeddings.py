"""
Tests for slide-embedding extraction (``oceanpath.modules.extract_slide_embeddings``).

Covers:
  - Manifest slide IDs: known extensions stripped, explicit column, duplicates rejected, and
    the same IDs as the linear probe resolves from the same manifest
  - Pooling over a memmap patch store: mean / max / mean+max equal the numpy reductions of each
    slide's features at any batch size (padding never leaks), bag caps, spatial sampling
  - Aggregator mode: checkpoint state cleaning (training prefixes stripped, heads dropped,
    ``aggregator_state_dict`` preferred), exact weight loading, batched extraction equal to
    per-slide forward passes, positional-forward and dict-output models
  - Embedding store: write_embedding_mmap -> linear_probing_sklearn.load_embedding_mmap round
    trip (fp32 exact, fp16 rounded, split across chunks, rewrites drop stale chunks), metadata
  - CLI main() end to end for mmap and npz outputs
"""

import json
import logging
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from oceanpath.data.mmap_builder import MMAP_SCHEMA_VERSION, validate_mmap_dir
from oceanpath.models import build_aggregator
from oceanpath.modules import extract_slide_embeddings as ese
from oceanpath.modules import linear_probing_sklearn as lp

CPU = torch.device("cpu")
FEAT_DIM = 8
PATCH_COUNTS = [5, 9, 3, 12, 7]
AGG_CFG = {"embed_dim": 8, "attn_dim": 8, "dropout": 0.0}


@pytest.fixture(scope="module")
def store(build_feature_store):
    return build_feature_store(PATCH_COUNTS, feat_dim=FEAT_DIM, seed=10)


@pytest.fixture(autouse=True)
def _detach_cli_log_handlers():
    """main() attaches handlers to its named logger; drop them after each test."""
    yield
    for name in ("extract_slide_embeddings", "linear_probe"):
        log = logging.getLogger(name)
        for handler in log.handlers[:]:
            handler.close()
            log.removeHandler(handler)


def _cli_args(monkeypatch, *argv: str):
    """The Namespace the CLI builds for ``argv`` (all defaults included)."""
    monkeypatch.setattr(sys, "argv", ["extract_slide_embeddings", *argv])
    return ese.parse_args()


def _store_args(monkeypatch, *, precision: int = 32, max_chunk_gb: float = 9.5):
    return _cli_args(
        monkeypatch,
        *("--mode", "mean_pool", "--mmap_dir", "patches", "--manifest_csv", "m.csv"),
        *("--output_path", "out", "--feat_precision", str(precision)),
        *("--max_chunk_gb", str(max_chunk_gb)),
    )


def _write_manifest(path, slide_ids, column="filename", suffix=".h5"):
    pd.DataFrame({column: [f"{sid}{suffix}" for sid in slide_ids]}).to_csv(path, index=False)
    return path


def _loader(store, *, mode="mean_pool", batch_size=1, max_instances=None, **kwargs):
    return ese.build_loader(
        mmap_dir=str(store.mmap_dir),
        slide_ids=store.slide_ids,
        mode=mode,
        batch_size=batch_size,
        num_workers=0,
        max_instances=max_instances,
        **kwargs,
    )


def _source_aggregator(seed: int = 7) -> torch.nn.Module:
    torch.manual_seed(seed)
    return build_aggregator("abmil", in_dim=FEAT_DIM, model_cfg=AGG_CFG).eval()


# ═════════════════════════════════════════════════════════════════════════════
# Manifest slide IDs
# ═════════════════════════════════════════════════════════════════════════════


class TestLoadSlideIds:
    FILENAMES = [
        "a.svs",
        "sub\\b.H5",
        "TCGA-AA-0001.8923A151-ABCD.svs",
        "c.ndpi",
        "plain_id",
        "d.tiff",
    ]
    EXPECTED = ["a", "sub/b", "TCGA-AA-0001.8923A151-ABCD", "c", "plain_id", "d"]

    def test_filenames_lose_known_extensions_only(self, tmp_path):
        csv = tmp_path / "m.csv"
        pd.DataFrame({"filename": self.FILENAMES}).to_csv(csv, index=False)
        assert ese.load_slide_ids(csv, None, "filename") == self.EXPECTED

    def test_probe_resolves_the_same_ids_from_the_same_manifest(self, tmp_path):
        """Extraction and probing must agree on IDs, or the probe cannot find embeddings."""
        csv = tmp_path / "m.csv"
        pd.DataFrame(
            {
                "filename": self.FILENAMES,
                "label": [0, 1] * 3,
                "patient_id": list("pqrstu"),
            }
        ).to_csv(csv, index=False)
        manifest = lp.load_manifest(csv, None, "filename", "label", "patient_id")
        assert manifest["slide_id"].tolist() == ese.load_slide_ids(csv, None, "filename")

    def test_explicit_slide_id_column_wins(self, tmp_path):
        csv = tmp_path / "m.csv"
        pd.DataFrame({"filename": ["x.svs", "y.svs"], "sid": ["s1.h5", "s2"]}).to_csv(
            csv, index=False
        )
        assert ese.load_slide_ids(csv, "sid", "filename") == ["s1", "s2"]

    def test_duplicate_ids_after_normalisation_rejected(self, tmp_path):
        csv = tmp_path / "m.csv"
        pd.DataFrame({"filename": ["a.svs", "a.h5", "b.svs"]}).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="duplicate slide_ids"):
            ese.load_slide_ids(csv, None, "filename")

    @pytest.mark.parametrize(("slide_col", "file_col"), [("sid", "filename"), (None, "path")])
    def test_missing_id_column_rejected(self, tmp_path, slide_col, file_col):
        csv = tmp_path / "m.csv"
        pd.DataFrame({"filename": ["a.svs"]}).to_csv(csv, index=False)
        with pytest.raises(ValueError, match=r"Columns: \['filename'\]"):
            ese.load_slide_ids(csv, slide_col, file_col)


# ═════════════════════════════════════════════════════════════════════════════
# Pooling over a memmap patch store
# ═════════════════════════════════════════════════════════════════════════════


class TestPooling:
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_mean_pool_equals_numpy_mean_of_each_slide(self, store, batch_size):
        emb, ids = ese.extract_pooling(_loader(store, batch_size=batch_size), CPU, "mean_pool")
        legacy, legacy_ids = ese.extract_mean_pool(_loader(store, batch_size=batch_size), CPU)

        assert ids == legacy_ids == store.slide_ids
        assert emb.dtype == np.float32
        expected = np.stack([store.features[sid].mean(axis=0) for sid in ids])
        np.testing.assert_allclose(emb, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(legacy, expected, rtol=1e-5, atol=1e-6)

    def test_max_and_mean_max_ignore_padding(self, store):
        """Zero padding must never win a max: the short slide has all-negative columns."""
        short = store.slide_ids[PATCH_COUNTS.index(min(PATCH_COUNTS))]
        assert (store.features[short].max(axis=0) < 0).any()

        mx, _ = ese.extract_pooling(_loader(store, batch_size=len(PATCH_COUNTS)), CPU, "max_pool")
        both, _ = ese.extract_pooling(
            _loader(store, batch_size=len(PATCH_COUNTS)), CPU, "mean_max_pool"
        )

        expected_max = np.stack([store.features[s].max(axis=0) for s in store.slide_ids])
        expected_mean = np.stack([store.features[s].mean(axis=0) for s in store.slide_ids])
        np.testing.assert_allclose(mx, expected_max, rtol=1e-6)
        assert both.shape == (len(PATCH_COUNTS), 2 * FEAT_DIM)
        np.testing.assert_allclose(both[:, :FEAT_DIM], expected_mean, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(both[:, FEAT_DIM:], expected_max, rtol=1e-6)

    def test_bag_cap_keeps_the_first_patches(self, store):
        emb, ids = ese.extract_pooling(
            _loader(store, batch_size=2, max_instances=4), CPU, "mean_pool"
        )
        expected = np.stack([store.features[sid][:4].mean(axis=0) for sid in ids])
        np.testing.assert_allclose(emb, expected, rtol=1e-5, atol=1e-6)

    def test_spatial_sampling_reads_coords_even_for_pooling(self, store):
        loader = _loader(store, max_instances=4, sampling_mode="spatial_stratified")
        for batch in loader:
            sid = batch["slide_ids"][0]
            n = int(batch["mask"].sum())
            assert n == min(4, len(store.features[sid]))
            rows = batch["features"][0, :n, 0].long().numpy()  # column 0 = row index
            assert len(set(rows.tolist())) == n
            np.testing.assert_array_equal(batch["coords"][0, :n].numpy(), store.coords[sid][rows])

    def test_unknown_pooling_mode_rejected(self, store):
        with pytest.raises(ValueError, match="Unknown pooling mode"):
            ese.extract_pooling(_loader(store), CPU, "median_pool")

    @pytest.mark.parametrize(
        "extract",
        [
            lambda loader: ese.extract_mean_pool(loader, CPU),
            lambda loader: ese.extract_pooling(loader, CPU, "max_pool"),
            lambda loader: ese.extract_aggregator(loader, _source_aggregator(), CPU),
        ],
        ids=["mean_pool", "pooling", "aggregator"],
    )
    def test_empty_loader_raises(self, extract):
        with pytest.raises(RuntimeError, match="No slides were extracted"):
            extract([])

    def test_loader_rejects_manifest_without_matching_slides(self, store):
        with pytest.raises(RuntimeError, match="No slide_ids matched"):
            ese.build_loader(
                mmap_dir=str(store.mmap_dir),
                slide_ids=["unknown_a", "unknown_b"],
                mode="mean_pool",
                batch_size=1,
                num_workers=0,
                max_instances=None,
            )


# ═════════════════════════════════════════════════════════════════════════════
# Aggregator checkpoints and extraction
# ═════════════════════════════════════════════════════════════════════════════

_HEAD_KEYS = {
    "projector.0.weight": torch.zeros(2),
    "predictor.net.weight": torch.zeros(2),
    "target_network.ema_model.patch_embed.0.weight": torch.zeros(2),
    "teacher.head.weight": torch.zeros(2),
    "classification_head.weight": torch.zeros(2),
    "classifier.bias": torch.zeros(2),
    "loss_fn.center": torch.zeros(2),
    "_online_for_ema.weight": torch.zeros(2),
}


class TestCheckpointState:
    @pytest.mark.parametrize(
        "prefix",
        ["", "aggregator.", "model.aggregator.", "trunk.", "online_encoder.", "backbone."],
    )
    def test_training_prefixes_stripped_and_heads_dropped(self, prefix):
        agg = _source_aggregator()
        raw = {"state_dict": {f"{prefix}{k}": v for k, v in agg.state_dict().items()}}
        raw["state_dict"].update(_HEAD_KEYS)

        cleaned = ese._clean_checkpoint_state(raw)

        expected = agg.state_dict()
        assert set(cleaned) == set(expected)
        for key, value in expected.items():
            assert torch.equal(cleaned[key], value), key

    def test_aggregator_state_dict_preferred_over_full_state(self):
        agg_state = _source_aggregator().state_dict()
        raw = {"state_dict": {"projector.w": torch.ones(1)}, "aggregator_state_dict": agg_state}
        assert ese._clean_checkpoint_state(raw) is agg_state


class TestAggregatorExtraction:
    @pytest.fixture
    def ckpt_args(self, tmp_path, monkeypatch):
        """A Lightning-style checkpoint whose keys carry the SSL module's prefixes."""
        source = _source_aggregator(seed=11)
        state = {f"model.aggregator.{k}": v for k, v in source.state_dict().items()}
        state.update(_HEAD_KEYS)
        path = tmp_path / "pretrain.ckpt"
        torch.save({"state_dict": state, "epoch": 3}, path)
        args = _cli_args(
            monkeypatch,
            *("--mode", "aggregator", "--arch", "abmil", "--ckpt", str(path)),
            *("--mmap_dir", "x", "--manifest_csv", "m.csv", "--output_path", "o"),
            *("--in_dim", str(FEAT_DIM), "--embed_dim", "8", "--attn_dim", "8"),
            *("--dropout", "0.0"),
        )
        return source, args

    def test_checkpoint_weights_load_exactly(self, ckpt_args):
        source, args = ckpt_args
        model = ese.build_aggregator_from_args(args)

        loaded, expected = model.state_dict(), source.state_dict()
        assert set(loaded) == set(expected)
        for key, value in expected.items():
            torch.testing.assert_close(loaded[key], value, msg=key)

    def test_without_checkpoint_the_aggregator_is_randomly_initialised(self, ckpt_args):
        source, args = ckpt_args
        args.ckpt = None
        torch.manual_seed(42)
        model = ese.build_aggregator_from_args(args)
        assert any(
            not torch.equal(model.state_dict()[k], v) for k, v in source.state_dict().items()
        )

    def test_batched_extraction_equals_per_slide_forward(self, store, ckpt_args):
        """Padding and masking inside a batch must not change any slide's embedding."""
        source, args = ckpt_args
        model = ese.build_aggregator_from_args(args)

        emb, ids = ese.extract_aggregator(
            _loader(store, mode="aggregator", batch_size=3), model, CPU
        )

        assert ids == store.slide_ids
        assert emb.shape == (len(ids), AGG_CFG["embed_dim"])
        with torch.no_grad():
            expected = np.stack(
                [
                    source(torch.from_numpy(store.features[sid])[None]).slide_embedding[0].numpy()
                    for sid in ids
                ]
            )
        np.testing.assert_allclose(emb, expected, rtol=1e-5, atol=1e-6)

    def test_positional_forward_and_dict_output_supported(self, store):
        class _PositionalMean(torch.nn.Module):
            def forward(self, x, mask=None, coords=None):
                assert coords is not None
                m = mask.unsqueeze(-1)
                return {"slide_embedding": (x * m).sum(1) / m.sum(1)}

        emb, ids = ese.extract_aggregator(
            _loader(store, mode="aggregator", batch_size=2), _PositionalMean(), CPU
        )
        expected = np.stack([store.features[sid].mean(axis=0) for sid in ids])
        np.testing.assert_allclose(emb, expected, rtol=1e-5, atol=1e-6)

    def test_output_without_slide_embedding_rejected(self, store):
        class _NoEmbedding(torch.nn.Module):
            def forward(self, features, mask=None, coords=None):
                return {"logits": features.mean(1)}

        with pytest.raises(ValueError, match="slide_embedding"):
            ese.extract_aggregator(_loader(store, mode="aggregator"), _NoEmbedding(), CPU)


# ═════════════════════════════════════════════════════════════════════════════
# Embedding store round trip
# ═════════════════════════════════════════════════════════════════════════════


class TestEmbeddingStore:
    IDS = ["s0", "dir/s1", "s2", "s3", "s4"]

    @pytest.fixture
    def embeddings(self):
        return np.random.default_rng(0).standard_normal((len(self.IDS), 6)).astype(np.float32)

    def test_fp32_store_reads_back_identically_through_the_probe_loader(
        self, tmp_path, monkeypatch, embeddings
    ):
        out = tmp_path / "emb"
        ese.write_embedding_mmap(out, self.IDS, embeddings, _store_args(monkeypatch))

        store = lp.load_embedding_mmap(out)

        assert store.slide_ids == self.IDS
        np.testing.assert_array_equal(store.embeddings, embeddings)
        meta = validate_mmap_dir(out)
        assert meta["n_slides"] == len(self.IDS)
        assert meta["feat_dim"] == 6
        assert meta["feat_dtype"] == "float32"
        assert (out / ".schema_version").read_text() == str(MMAP_SCHEMA_VERSION)

    def test_fp16_store_rounds_to_half_precision(self, tmp_path, monkeypatch, embeddings):
        out = tmp_path / "emb16"
        ese.write_embedding_mmap(out, self.IDS, embeddings, _store_args(monkeypatch, precision=16))

        store = lp.load_embedding_mmap(out)

        assert validate_mmap_dir(out)["feat_dtype"] == "float16"
        np.testing.assert_array_equal(
            store.embeddings, embeddings.astype(np.float16).astype(np.float32)
        )

    def test_tiny_chunks_split_rows_across_files(self, tmp_path, monkeypatch, embeddings):
        """A 2-row chunk budget spreads 5 rows over 3 feature files that still read back."""
        out = tmp_path / "chunked"
        row_bytes = embeddings[0].nbytes
        args = _store_args(monkeypatch, max_chunk_gb=2.5 * row_bytes / 1024**3)

        ese.write_embedding_mmap(out, self.IDS, embeddings, args)

        assert sorted(p.name for p in out.glob("features_*.bin")) == [
            "features_000.bin",
            "features_001.bin",
            "features_002.bin",
        ]
        np.testing.assert_array_equal(lp.load_embedding_mmap(out).embeddings, embeddings)

    def test_rewrite_drops_stale_chunks(self, tmp_path, monkeypatch, embeddings):
        out = tmp_path / "emb"
        tiny = _store_args(monkeypatch, max_chunk_gb=1e-9)
        ese.write_embedding_mmap(out, self.IDS, embeddings, tiny)
        assert len(list(out.glob("features_*.bin"))) == len(self.IDS)

        ese.write_embedding_mmap(out, self.IDS[:2], embeddings[:2] * 2, _store_args(monkeypatch))

        assert [p.name for p in out.glob("features_*.bin")] == ["features_000.bin"]
        assert [p.name for p in out.glob("coords_*.bin")] == ["coords_000.bin"]
        store = lp.load_embedding_mmap(out)
        assert store.slide_ids == self.IDS[:2]
        np.testing.assert_array_equal(store.embeddings, embeddings[:2] * 2)

    def test_metadata_describes_the_extraction(self, tmp_path, monkeypatch, embeddings):
        out = tmp_path / "emb"
        args = _store_args(monkeypatch)
        ese.write_embedding_mmap(out, self.IDS, embeddings, args)

        meta = json.loads((out / "embedding_meta.json").read_text())
        assert meta["mode"] == "mean_pool"
        assert meta["n_slides"] == len(self.IDS)
        assert meta["feat_dim"] == 6
        assert meta["feat_precision"] == 32
        assert (out / ".source_hash").read_text() == ese.compute_source_hash(
            self.IDS, embeddings, args
        )

    def test_source_hash_tracks_ids_mode_and_checkpoint(self, monkeypatch, embeddings):
        args = _store_args(monkeypatch)
        base = ese.compute_source_hash(self.IDS, embeddings, args)

        assert len(base) == 20
        assert base == ese.compute_source_hash(list(self.IDS), embeddings.copy(), args)
        assert base != ese.compute_source_hash(self.IDS[::-1], embeddings, args)
        args.mode = "max_pool"
        assert base != ese.compute_source_hash(self.IDS, embeddings, args)
        args.mode, args.ckpt = "mean_pool", "ssl.ckpt"
        assert base != ese.compute_source_hash(self.IDS, embeddings, args)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════


class TestMain:
    def test_mean_pool_cli_writes_a_probe_ready_store(self, store, tmp_path, monkeypatch):
        manifest = _write_manifest(tmp_path / "m.csv", store.slide_ids, suffix=".svs")
        out = tmp_path / "embeddings" / "meanpool"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "extract_slide_embeddings",
                *("--mode", "mean_pool", "--mmap_dir", str(store.mmap_dir)),
                *("--manifest_csv", str(manifest), "--output_path", str(out)),
                *("--num_workers", "0", "--batch_size", "2", "--device", "cpu"),
            ],
        )

        ese.main()

        emb_store = lp.load_embedding_mmap(out)
        assert emb_store.slide_ids == store.slide_ids
        expected = np.stack([store.features[sid].mean(axis=0) for sid in store.slide_ids])
        np.testing.assert_allclose(emb_store.embeddings, expected, rtol=1e-5, atol=1e-6)

    def test_aggregator_cli_writes_npz_from_the_checkpoint(self, store, tmp_path, monkeypatch):
        source = _source_aggregator(seed=5)
        ckpt = tmp_path / "aggregator_weights.pt"
        torch.save(source.state_dict(), ckpt)
        manifest = _write_manifest(tmp_path / "m.csv", store.slide_ids[:3])
        out = tmp_path / "emb.npz"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "extract_slide_embeddings",
                *("--mode", "aggregator", "--arch", "abmil", "--ckpt", str(ckpt)),
                *("--mmap_dir", str(store.mmap_dir), "--manifest_csv", str(manifest)),
                *("--output_path", str(out), "--output_format", "npz"),
                *("--in_dim", str(FEAT_DIM), "--embed_dim", "8", "--attn_dim", "8"),
                *("--dropout", "0.0", "--num_workers", "0", "--device", "cpu"),
            ],
        )

        ese.main()

        data = np.load(out, allow_pickle=True)
        assert data["slide_ids"].tolist() == store.slide_ids[:3]
        with torch.no_grad():
            expected = np.stack(
                [
                    source(torch.from_numpy(store.features[sid])[None]).slide_embedding[0].numpy()
                    for sid in store.slide_ids[:3]
                ]
            )
        np.testing.assert_allclose(data["embeddings"], expected, rtol=1e-5, atol=1e-6)
