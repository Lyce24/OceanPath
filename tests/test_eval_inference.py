"""
Tests for final-model inference: oceanpath.eval.inference (Stage 6 test-set
predictions) and oceanpath.eval.ensemble (EnsembleWrapper, Stage 7).

A synthetic memmap store (conftest's build_feature_store) holds eight slides;
splits.parquet marks the last four as the held-out test set (fold -1). The
final/ directory is laid out the way modules/finalize.py writes it
(best_fold/model.ckpt, ensemble/fold_K.ckpt, refit/model.ckpt, each with an
info.json), with untrained ABMIL checkpoints saved through Trainer.save_checkpoint
as training does. Every expected probability, embedding and attention map is a
direct forward pass of the in-memory model that was saved.

Covers:
  - run_inference: single model (batch 1), fold ensemble = mean of member
    probabilities (batched, padded bags), three-class and single-logit heads,
    folds that failed or are missing on disk, explicit test slide IDs, a scheme
    without a test set, and the errors for missing files
  - get_test_slide_ids
  - EnsembleWrapper: loading per strategy, predict_slide averages (probs, logits,
    softmax attention, embedding), masked padding, single-logit heads,
    get_all_embeddings over an MmapDataset, agreement with run_inference, cleanup
"""

import json
from types import SimpleNamespace

import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch

from oceanpath.data.dataset import MmapDataset
from oceanpath.eval.ensemble import EnsembleWrapper
from oceanpath.eval.inference import get_test_slide_ids, run_inference
from oceanpath.modules.train_module import MILTrainModule

IN_DIM = 16
ABMIL_CFG = {"embed_dim": 8, "attn_dim": 8, "dropout": 0.0}
PATCH_COUNTS = [12, 30, 7, 21, 15, 9, 26, 18]
LABELS = [0, 1, 0, 1, 1, 0, 1, 0]
FOLDS = [0, 1, 0, 1, -1, -1, -1, -1]  # the last four slides are the held-out test set
N_MEMBERS = 3


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures and helpers
# ═════════════════════════════════════════════════════════════════════════════


def _write_checkpoint(path, num_classes: int = 2, seed: int = 0):
    """Save an untrained MILTrainModule the way training does (Trainer.save_checkpoint)."""
    torch.manual_seed(seed)
    module = MILTrainModule(
        arch="abmil",
        in_dim=IN_DIM,
        num_classes=num_classes,
        model_cfg=dict(ABMIL_CFG),
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


def _model_dir(root, name: str, info: dict | None):
    path = root / name
    path.mkdir(parents=True)
    if info is not None:
        (path / "info.json").write_text(json.dumps(info))
    return path


@pytest.fixture(scope="module")
def cohort(build_feature_store, tmp_path_factory):
    store = build_feature_store(PATCH_COUNTS, feat_dim=IN_DIM, seed=11)
    root = tmp_path_factory.mktemp("eval_cohort")

    csv = root / "labels.csv"
    pd.DataFrame({"filename": [f"{s}.h5" for s in store.slide_ids], "label": LABELS}).to_csv(
        csv, index=False
    )
    splits_dir = root / "splits"
    splits_dir.mkdir()
    pd.DataFrame({"slide_id": store.slide_ids, "fold": FOLDS}).to_parquet(
        splits_dir / "splits.parquet", index=False
    )

    final = root / "final"
    best_fold = _model_dir(final, "best_fold", {"strategy": "best_fold", "source_fold": 1})
    ensemble = _model_dir(
        final, "ensemble", {"strategy": "ensemble", "folds": [{"fold": k} for k in range(3)]}
    )
    refit = _model_dir(final, "refit", {"strategy": "refit"})
    single_logit = _model_dir(final, "single_logit", {"strategy": "best_fold"})

    return SimpleNamespace(
        store=store,
        csv=csv,
        splits_dir=splits_dir,
        final=final,
        test_ids=[s for s, f in zip(store.slide_ids, FOLDS, strict=True) if f == -1],
        labels=dict(zip(store.slide_ids, LABELS, strict=True)),
        best_fold=best_fold,
        best_fold_model=_write_checkpoint(best_fold / "model.ckpt", seed=0),
        ensemble=ensemble,
        members=[
            _write_checkpoint(ensemble / f"fold_{k}.ckpt", seed=10 + k) for k in range(N_MEMBERS)
        ],
        refit=refit,
        refit_model=_write_checkpoint(refit / "model.ckpt", num_classes=3, seed=20),
        single_logit=single_logit,
        single_logit_model=_write_checkpoint(single_logit / "model.ckpt", num_classes=1, seed=30),
    )


def _infer(cohort, model_dir, splits_dir=None, **kwargs):
    return run_inference(
        model_dir=model_dir,
        mmap_dir=str(cohort.store.mmap_dir),
        splits_dir=str(splits_dir or cohort.splits_dir),
        csv_path=str(cohort.csv),
        label_column="label",
        filename_column="filename",
        scheme="kfold",
        num_workers=0,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        **kwargs,
    )


def _forward(model, features: np.ndarray):
    """Direct forward pass on one whole slide: (softmax probs, logits, embedding, attention)."""
    with torch.no_grad():
        out = model(torch.from_numpy(features).unsqueeze(0), return_attention=True)
    logits = out.logits[0]
    return SimpleNamespace(
        probs=torch.softmax(logits, -1).numpy(),
        logits=logits.numpy(),
        embedding=out.slide_embedding[0].numpy(),
        attention=torch.softmax(out.extras["attention_weights"][0], -1).numpy(),
    )


def _expected_frame(cohort, models, slide_ids=None) -> pd.DataFrame:
    """Mean softmax probabilities of ``models`` per test slide, in prob_c columns."""
    rows = []
    for sid in slide_ids or cohort.test_ids:
        probs = np.mean([_forward(m, cohort.store.features[sid]).probs for m in models], axis=0)
        rows.append({"slide_id": sid, **{f"prob_{c}": p for c, p in enumerate(probs)}})
    return pd.DataFrame(rows).set_index("slide_id")


@pytest.fixture(scope="module")
def ensemble_preds(cohort):
    """Stage 6 ensemble inference with two slides per batch (padded, masked bags)."""
    return _infer(cohort, cohort.ensemble, batch_size=2)


# ═════════════════════════════════════════════════════════════════════════════
# run_inference
# ═════════════════════════════════════════════════════════════════════════════


class TestRunInference:
    def test_single_model_matches_a_direct_forward_pass(self, cohort):
        # scripts/evaluate.py passes the split's test IDs explicitly
        preds = _infer(cohort, cohort.best_fold, batch_size=1, test_slide_ids=cohort.test_ids)
        assert list(preds.columns) == ["slide_id", "label", "prob_1"]
        assert preds["slide_id"].tolist() == cohort.test_ids
        assert preds["label"].tolist() == [cohort.labels[s] for s in cohort.test_ids]
        expected = _expected_frame(cohort, [cohort.best_fold_model])
        np.testing.assert_allclose(
            preds["prob_1"], expected.loc[cohort.test_ids, "prob_1"], atol=1e-6
        )

    def test_ensemble_is_the_mean_of_its_members(self, cohort, ensemble_preds):
        assert sorted(ensemble_preds["slide_id"]) == sorted(cohort.test_ids)
        preds = ensemble_preds.set_index("slide_id").loc[cohort.test_ids]
        assert preds["label"].tolist() == [cohort.labels[s] for s in cohort.test_ids]
        expected = _expected_frame(cohort, cohort.members)
        np.testing.assert_allclose(preds["prob_1"], expected["prob_1"], atol=1e-6)
        # ... and not just one of the members
        for member in cohort.members:
            single = _expected_frame(cohort, [member])
            assert np.abs(preds["prob_1"] - single["prob_1"]).max() > 1e-4

    def test_three_class_model_returns_a_probability_per_class(self, cohort):
        preds = _infer(cohort, cohort.refit, batch_size=4).set_index("slide_id")
        assert sorted(preds.columns) == ["label", "prob_0", "prob_1", "prob_2"]
        expected = _expected_frame(cohort, [cohort.refit_model])
        cols = ["prob_0", "prob_1", "prob_2"]
        np.testing.assert_allclose(preds.loc[cohort.test_ids, cols], expected[cols], atol=1e-6)
        np.testing.assert_allclose(preds[cols].sum(1), 1.0, atol=1e-6)

    def test_single_logit_model_gives_the_sigmoid(self, cohort):
        preds = _infer(cohort, cohort.single_logit, batch_size=4).set_index("slide_id")
        for sid in cohort.test_ids:
            logit = _forward(cohort.single_logit_model, cohort.store.features[sid]).logits
            assert preds.loc[sid, "prob_1"] == pytest.approx(1 / (1 + np.exp(-logit.item())))

    def test_failed_and_missing_folds_are_left_out(self, cohort, tmp_path):
        """info.json lists fold 1 as failed and fold 2's file is gone: the
        ensemble is fold 0 alone."""
        model_dir = _model_dir(
            tmp_path,
            "ensemble",
            {
                "strategy": "ensemble",
                "folds": [{"fold": 0}, {"fold": 1, "error": "checkpoint not found"}, {"fold": 2}],
            },
        )
        for k in (0, 1):
            (model_dir / f"fold_{k}.ckpt").write_bytes(
                (cohort.ensemble / f"fold_{k}.ckpt").read_bytes()
            )
        preds = _infer(cohort, model_dir).set_index("slide_id").loc[cohort.test_ids]
        expected = _expected_frame(cohort, cohort.members[:1])
        np.testing.assert_allclose(preds["prob_1"], expected["prob_1"], atol=1e-6)

    def test_explicit_test_slide_ids_replace_the_split(self, cohort):
        """Two training slides and one test slide, named explicitly."""
        wanted = [cohort.store.slide_ids[0], cohort.store.slide_ids[3], cohort.test_ids[1]]
        preds = _infer(cohort, cohort.best_fold, test_slide_ids=wanted)
        assert preds["slide_id"].tolist() == wanted
        expected = _expected_frame(cohort, [cohort.best_fold_model], slide_ids=wanted)
        np.testing.assert_allclose(preds["prob_1"], expected.loc[wanted, "prob_1"], atol=1e-6)

    def test_scheme_without_a_test_set_gives_no_predictions(self, cohort, tmp_path):
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()
        pd.DataFrame({"slide_id": cohort.store.slide_ids, "fold": [0, 1] * 4}).to_parquet(
            splits_dir / "splits.parquet", index=False
        )
        assert _infer(cohort, cohort.best_fold, splits_dir=splits_dir).empty
        assert _infer(cohort, cohort.ensemble, splits_dir=splits_dir).empty

    def test_missing_files_raise(self, cohort, tmp_path):
        with pytest.raises(FileNotFoundError, match="info.json"):
            _infer(cohort, _model_dir(tmp_path, "no_info", None))
        with pytest.raises(FileNotFoundError, match="model.ckpt"):
            _infer(cohort, _model_dir(tmp_path, "refit", {"strategy": "refit"}))
        all_failed = {"strategy": "ensemble", "folds": [{"fold": 0, "error": "no checkpoint"}]}
        with pytest.raises(ValueError, match="No valid fold"):
            _infer(cohort, _model_dir(tmp_path, "failed", all_failed))
        no_files = {"strategy": "ensemble", "folds": [{"fold": 0}, {"fold": 1}]}
        with pytest.raises(ValueError, match="No successful fold"):
            _infer(cohort, _model_dir(tmp_path, "no_files", no_files))


def test_get_test_slide_ids_reads_the_held_out_fold(cohort):
    assert get_test_slide_ids(str(cohort.splits_dir), scheme="kfold") == cohort.test_ids


# ═════════════════════════════════════════════════════════════════════════════
# EnsembleWrapper
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def wrapper(cohort):
    return EnsembleWrapper(cohort.ensemble, device="cpu")


class TestEnsembleWrapper:
    def test_loads_every_fold_checkpoint(self, wrapper):
        assert wrapper.strategy == "ensemble"
        assert wrapper.n_models == N_MEMBERS

    def test_predict_slide_averages_the_members(self, cohort, wrapper):
        sid = cohort.test_ids[0]
        features = cohort.store.features[sid]
        result = wrapper.predict_slide(torch.from_numpy(features))
        members = [_forward(m, features) for m in cohort.members]

        np.testing.assert_allclose(result.probs, np.mean([m.probs for m in members], 0), atol=1e-6)
        np.testing.assert_allclose(
            result.logits, np.mean([m.logits for m in members], 0), atol=1e-6
        )
        np.testing.assert_allclose(
            result.embedding, np.mean([m.embedding for m in members], 0), atol=1e-6
        )
        # Attention: softmax per member, then the mean -> one weight per patch
        assert result.attention.shape == (len(features),)
        np.testing.assert_allclose(
            result.attention, np.mean([m.attention for m in members], 0), atol=1e-6
        )
        assert result.attention.sum() == pytest.approx(1.0, abs=1e-5)

        assert result.n_models == N_MEMBERS
        assert [d["fold"] for d in result.fold_details] == list(range(N_MEMBERS))
        for detail, member in zip(result.fold_details, members, strict=True):
            np.testing.assert_allclose(detail["probs"], member.probs, atol=1e-6)
            assert detail["has_attention"] is True
        assert result.pred_class == int(np.argmax(result.probs))
        assert result.pred_prob == pytest.approx(float(result.probs.max()))

    def test_masked_padding_does_not_change_the_prediction(self, cohort, wrapper):
        features = torch.from_numpy(cohort.store.features[cohort.test_ids[1]])
        n = len(features)
        padded = torch.cat([features, torch.zeros(5, IN_DIM)])
        mask = torch.cat([torch.ones(n), torch.zeros(5)])

        plain = wrapper.predict_slide(features)
        masked = wrapper.predict_slide(padded.unsqueeze(0), mask=mask)
        np.testing.assert_allclose(masked.probs, plain.probs, atol=1e-6)
        np.testing.assert_allclose(masked.embedding, plain.embedding, atol=1e-6)
        np.testing.assert_allclose(masked.attention[:n], plain.attention, atol=1e-6)
        assert np.all(masked.attention[n:] == 0)

    def test_agrees_with_stage_6_ensemble_inference(self, cohort, wrapper, ensemble_preds):
        preds = ensemble_preds.set_index("slide_id")
        for sid in cohort.test_ids:
            result = wrapper.predict_slide(torch.from_numpy(cohort.store.features[sid]))
            assert result.probs[1] == pytest.approx(preds.loc[sid, "prob_1"], abs=1e-6)

    def test_get_all_embeddings_averages_members_per_slide(self, cohort, wrapper):
        dataset = MmapDataset(
            mmap_dir=str(cohort.store.mmap_dir),
            slide_ids=cohort.test_ids,
            labels=cohort.labels,
            max_instances=8000,
            is_train=False,
        )
        embeddings, slide_ids, labels = wrapper.get_all_embeddings(dataset)
        assert slide_ids == cohort.test_ids
        assert labels.tolist() == [cohort.labels[s] for s in cohort.test_ids]
        assert embeddings.shape == (len(cohort.test_ids), ABMIL_CFG["embed_dim"])
        for row, sid in zip(embeddings, slide_ids, strict=True):
            members = [_forward(m, cohort.store.features[sid]) for m in cohort.members]
            np.testing.assert_allclose(row, np.mean([m.embedding for m in members], 0), atol=1e-6)


class TestEnsembleWrapperSingleModel:
    @pytest.mark.parametrize("strategy", ["best_fold", None])
    def test_single_checkpoint_directory(self, cohort, tmp_path, strategy):
        """best_fold/refit, or no info.json at all, load model.ckpt alone."""
        info = None if strategy is None else {"strategy": strategy}
        model_dir = _model_dir(tmp_path, "model", info)
        (model_dir / "model.ckpt").write_bytes((cohort.best_fold / "model.ckpt").read_bytes())

        wrapper = EnsembleWrapper(model_dir, device="cpu")
        assert wrapper.strategy == (strategy or "unknown")
        assert wrapper.n_models == 1
        features = cohort.store.features[cohort.test_ids[2]]
        result = wrapper.predict_slide(torch.from_numpy(features))
        expected = _forward(cohort.best_fold_model, features)
        np.testing.assert_allclose(result.probs, expected.probs, atol=1e-6)
        np.testing.assert_allclose(result.attention, expected.attention, atol=1e-6)

    def test_cleanup_drops_the_models(self, cohort, monkeypatch):
        wrapper = EnsembleWrapper(cohort.best_fold, device="cpu")
        monkeypatch.setattr("oceanpath.eval.ensemble.gc.collect", lambda: 0)  # ~0.3 s, not checked
        wrapper.cleanup()
        assert wrapper.n_models == 0

    def test_single_logit_head_gives_two_class_sigmoid_probabilities(self, cohort):
        """A BCE head emits one logit per slide; a softmax over it is always 1.0."""
        wrapper = EnsembleWrapper(cohort.single_logit, device="cpu")
        for sid in cohort.test_ids:
            features = cohort.store.features[sid]
            logit = _forward(cohort.single_logit_model, features).logits.item()
            p = 1 / (1 + np.exp(-logit))
            result = wrapper.predict_slide(torch.from_numpy(features))
            np.testing.assert_allclose(result.probs, [1 - p, p], atol=1e-6)
            np.testing.assert_allclose(result.logits, [logit], atol=1e-6)
            assert result.pred_class == int(p > 0.5)

    def test_missing_checkpoints_raise(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="fold_"):
            EnsembleWrapper(_model_dir(tmp_path, "ensemble", {"strategy": "ensemble"}), "cpu")
        with pytest.raises(FileNotFoundError, match="model.ckpt"):
            EnsembleWrapper(_model_dir(tmp_path, "refit", {"strategy": "refit"}), "cpu")
