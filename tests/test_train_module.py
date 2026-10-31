"""
Tests for the supervised training module (``oceanpath.modules.train_module``)
and the fold loop that ``scripts/train.py`` builds around it.

One tiny fold is trained once per module (``fold_run``): ABMIL with 8-dim
embeddings, fit for three epochs on CPU over a synthetic memmap store through
the real ``MILDataModule``, with the checkpointing and the
``oceanpath.modules.callbacks`` that ``run_fold`` attaches (fold timing,
batch-index logging, the W&B fold summary, the metadata writer and the bag-size
curriculum). After the fit the fixture does what ``run_fold`` does: reload the
best checkpoint, re-validate it, save the validation predictions and embeddings,
then run the held-out test set. A recording logger stands in for the W&B run, so
nothing reaches the network.

Covers:
  * the fold fit: logged train / val / test metric keys, metric values that
    agree with sklearn on the saved predictions, the best and last checkpoints
    (reload to the same weights, epoch parsed from the checkpoint path),
    preds_*.parquet and embeddings_*.npz contents
  * the callbacks inside that fit: batch_indices.npy, metadata.json, fold wall
    time, the best-metric W&B summary, the per-epoch bag-size curriculum
  * losses: ce / bce / focal, with and without class weights, against
    hand-computed values; logit clamping
  * probabilities and predicted classes for one-logit, two-logit and
    multiclass heads; the multiclass predictions file
  * a non-finite training loss leaves the weights finite
  * configure_optimizers: parameter groups (differential LR, frozen aggregator)
    and the cosine / warmup / step / plateau / none schedules
  * the cross-rank gather that de-duplicates DDP sampler padding
"""

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.logger import Logger
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from oceanpath.data.datamodule import MILDataModule
from oceanpath.modules.callbacks import (
    BagCurriculumCallback,
    BatchIndexLogger,
    FoldTimingCallback,
    MetadataWriter,
    WandbFoldSummary,
)
from oceanpath.modules.finalize import parse_best_epoch_from_checkpoint
from oceanpath.modules.train_module import FocalLoss, MILTrainModule

IN_DIM = 16
TINY_ABMIL = {"embed_dim": 8, "attn_dim": 8, "dropout": 0.0}
# What OmegaConf.to_container(cfg.model) hands MILTrainModule for configs/model/abmil.yaml
MODEL_CFG = {
    "name": "abmil",
    "arch": "abmil",
    "embed_dim": 8,
    "num_fc_layers": 1,
    "attn_dim": 8,
    "gate": True,
    "dropout": 0.0,
    "gradient_checkpointing": False,
}
# Parameters of MODEL_CFG at in_dim=16 with a two-class head, counted by hand:
#   patch_embed Linear(16, 8)                 16*8 + 8 = 136
#   gated attention a, b: Linear(8, 8)        2 * (8*8 + 8) = 144
#   attention_c Linear(8, 1)                  8 + 1 = 9
#   head Linear(8, 2)                         8*2 + 2 = 18
N_PARAMS = 136 + 144 + 9 + 18

# (n_patches, label, fold); fold -1 is the fixed held-out test set
COHORT = [
    (9, 0, 0),
    (30, 1, 0),
    (12, 0, 0),
    (14, 1, 1),
    (20, 0, 1),
    (13, 1, 2),
    (16, 0, 2),
    (25, 1, 2),
    (11, 0, -1),
    (17, 1, -1),
]
EPOCHS = 3
BATCH_SIZE = 2
CURRICULUM = {"start_instances": 4, "end_instances": 12, "warmup_epochs": 2}
CURRICULUM_CAPS = [4, 8, 12]  # bag cap the curriculum sets at epochs 0, 1, 2
VAL_KEYS = {
    "val/loss",
    "val/acc",
    "val/f1",
    "val/balanced_acc",
    "val/precision",
    "val/recall",
    "val/auroc",
}
TEST_KEYS = {k.replace("val/", "test/") for k in VAL_KEYS}


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures and helpers
# ═════════════════════════════════════════════════════════════════════════════


class RecordingLogger(Logger):
    """Keeps every ``log_metrics`` call; ``experiment`` stands in for a W&B run."""

    def __init__(self):
        super().__init__()
        self.records: list[dict] = []
        self.run = SimpleNamespace(summary={})

    @property
    def name(self):
        return "recording"

    @property
    def version(self):
        return 0

    @property
    def experiment(self):
        return self.run

    def log_metrics(self, metrics, step=None):
        self.records.append(dict(metrics))

    def log_hyperparams(self, params, *args, **kwargs):
        pass


class BagSizeProbe(L.Callback):
    """Records the largest training bag the model receives in each epoch."""

    def __init__(self):
        self.max_len: dict[int, int] = {}

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        epoch = trainer.current_epoch
        self.max_len[epoch] = max(self.max_len.get(epoch, 0), int(batch["lengths"].max()))


def write_cohort(root: Path, slide_ids: list[str], cohort=COHORT) -> SimpleNamespace:
    """Manifest CSV (filename + label) and a kfold splits.parquet for the store's slides."""
    labels = {sid: label for sid, (_, label, _) in zip(slide_ids, cohort, strict=True)}
    folds = {sid: fold for sid, (_, _, fold) in zip(slide_ids, cohort, strict=True)}
    csv_path = root / "manifest.csv"
    pd.DataFrame(
        {"filename": [f"{s}.svs" for s in slide_ids], "label": list(labels.values())}
    ).to_csv(csv_path, index=False)
    splits_dir = root / "splits"
    splits_dir.mkdir()
    pd.DataFrame({"slide_id": slide_ids, "fold": list(folds.values())}).to_parquet(
        splits_dir / "splits.parquet", index=False
    )
    return SimpleNamespace(
        csv=csv_path,
        splits_dir=splits_dir,
        labels=labels,
        val_ids=sorted(s for s, f in folds.items() if f == 0),
        train_ids=sorted(s for s, f in folds.items() if f > 0),
        test_ids=sorted(s for s, f in folds.items() if f == -1),
    )


def snapshot(metrics) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.items()}


@pytest.fixture(scope="module")
def fold_run(build_feature_store, tmp_path_factory):
    """Train fold 0 the way scripts/train.py's run_fold does, at toy scale."""
    store = build_feature_store([n for n, _, _ in COHORT], feat_dim=IN_DIM, seed=3)
    root = tmp_path_factory.mktemp("fold_run")
    cohort = write_cohort(root, store.slide_ids)
    fold_dir = root / "fold_0"
    fold_dir.mkdir()

    L.seed_everything(0, workers=True)
    dm = MILDataModule(
        mmap_dir=str(store.mmap_dir),
        splits_dir=str(cohort.splits_dir),
        csv_path=str(cohort.csv),
        label_column="label",
        filename_column="filename",
        scheme="kfold",
        fold=0,
        batch_size=BATCH_SIZE,
        max_instances=64,
        num_workers=0,
        class_weighted_sampling=True,
        verify_splits=False,
        use_preallocated_collator=True,
    )
    module = MILTrainModule(
        arch="abmil",
        in_dim=IN_DIM,
        num_classes=dm.num_classes,
        model_cfg=dict(MODEL_CFG),
        lr=5e-3,
        lr_scheduler="cosine",
        warmup_epochs=1,
        max_epochs=EPOCHS,
        canary_interval=1,
    )
    config = {"exp_name": "toy_abmil", "model": dict(MODEL_CFG), "training": {"lr": 5e-3}}
    logger = RecordingLogger()
    probe = BagSizeProbe()
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        callbacks=[
            ModelCheckpoint(
                dirpath=str(fold_dir / "checkpoints"),
                filename="best-{epoch}-{val/loss:.4f}",
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                save_last=True,
            ),
            FoldTimingCallback(),
            BatchIndexLogger(output_dir=str(fold_dir)),
            WandbFoldSummary(fold=0, monitor_metric="val/loss", monitor_mode="min"),
            MetadataWriter(output_dir=str(fold_dir), config=config),
            BagCurriculumCallback(**CURRICULUM),
            probe,
        ],
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(fold_dir),
    )
    start = time.monotonic()
    trainer.fit(module, datamodule=dm)
    fit_seconds = time.monotonic() - start

    fit_records = list(logger.records)
    fit_metrics = snapshot(trainer.callback_metrics)
    final_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
    ckpt_cb = trainer.checkpoint_callback
    best_path, last_path = ckpt_cb.best_model_path, ckpt_cb.last_model_path
    best_score = float(ckpt_cb.best_model_score)

    # Post-fit evaluation, as run_fold does it
    best = MILTrainModule.load_from_checkpoint(best_path, weights_only=False)
    best.eval()
    trainer.validate(best, datamodule=dm)
    val_metrics = snapshot(trainer.callback_metrics)
    preds_val = best.save_predictions(str(fold_dir), prefix="val")
    emb_val = best.save_embeddings(str(fold_dir), prefix="val")
    dm.setup(stage="test")
    trainer.test(best, datamodule=dm)
    test_metrics = snapshot(trainer.callback_metrics)
    preds_test = best.save_predictions(str(fold_dir), prefix="test")
    emb_test = best.save_embeddings(str(fold_dir), prefix="test")

    return SimpleNamespace(
        store=store,
        cohort=cohort,
        fold_dir=fold_dir,
        config=config,
        best=best,
        summary=logger.run.summary,
        probe=probe,
        fit_seconds=fit_seconds,
        fit_records=fit_records,
        fit_metrics=fit_metrics,
        final_state=final_state,
        best_path=best_path,
        last_path=last_path,
        best_score=best_score,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        preds_val=preds_val,
        emb_val=emb_val,
        preds_test=preds_test,
        emb_test=emb_test,
    )


def epoch_val_losses(run) -> list[float]:
    """val/loss of each training epoch, in order (the sanity check never reaches loggers)."""
    return [r["val/loss"] for r in run.fit_records if "val/loss" in r]


def make_module(**kwargs) -> MILTrainModule:
    torch.manual_seed(0)
    defaults = {
        "arch": "abmil",
        "in_dim": IN_DIM,
        "num_classes": 2,
        "model_cfg": dict(TINY_ABMIL),
        "canary_interval": 0,
    }
    defaults.update(kwargs)
    return MILTrainModule(**defaults)


def bag_batch(n_patches=6, labels=(0, 1), slide_ids=("a", "b"), nan=False, seed=0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    features = torch.randn(len(labels), n_patches, IN_DIM, generator=gen)
    if nan:
        features[0, 0, 0] = float("nan")
    return {
        "features": features,
        "mask": torch.ones(len(labels), n_patches),
        "labels": torch.tensor(labels),
        "slide_ids": list(slide_ids),
    }


# ═════════════════════════════════════════════════════════════════════════════
# The fold fit
# ═════════════════════════════════════════════════════════════════════════════


class TestFoldFit:
    def test_train_loss_is_logged_every_step_and_val_metrics_every_epoch(self, fold_run):
        steps_per_epoch = math.ceil(len(fold_run.cohort.train_ids) / BATCH_SIZE)
        step_losses = [r for r in fold_run.fit_records if "train/loss_step" in r]
        assert len(step_losses) == EPOCHS * steps_per_epoch
        epoch_records = [r for r in fold_run.fit_records if "train/loss_epoch" in r]
        assert len(epoch_records) == EPOCHS
        assert all("train/acc" in r for r in epoch_records)
        val_records = [r for r in fold_run.fit_records if "val/loss" in r]
        assert len(val_records) == EPOCHS
        for record in val_records:
            assert record.keys() >= VAL_KEYS
            assert all(math.isfinite(record[k]) for k in VAL_KEYS)

    def test_callback_metrics_after_fit_hold_train_and_val_keys(self, fold_run):
        assert fold_run.fit_metrics.keys() >= {"train/loss", "train/acc"} | VAL_KEYS

    def test_last_checkpoint_reloads_to_the_final_weights(self, fold_run):
        assert Path(fold_run.last_path).name == "last.ckpt"
        reloaded = MILTrainModule.load_from_checkpoint(fold_run.last_path, weights_only=False)
        state = reloaded.state_dict()
        assert state.keys() == fold_run.final_state.keys()
        for key, value in fold_run.final_state.items():
            assert torch.equal(state[key], value), key

    def test_best_checkpoint_is_the_epoch_with_the_lowest_val_loss(self, fold_run):
        losses = epoch_val_losses(fold_run)
        best_epoch = int(np.argmin(losses))
        assert fold_run.best_score == pytest.approx(min(losses), abs=1e-6)
        # ModelCheckpoint names it best-epoch=E-val/loss=X.ckpt; the "/" makes a
        # sub-directory, and the epoch count must still be recovered from the path.
        assert f"epoch={best_epoch}" in fold_run.best_path
        assert parse_best_epoch_from_checkpoint(fold_run.best_path) == best_epoch + 1

    def test_best_checkpoint_rebuilds_the_model_from_its_hparams(self, fold_run):
        hp = fold_run.best.hparams
        assert (hp.arch, hp.in_dim, hp.num_classes) == ("abmil", IN_DIM, 2)
        assert hp.model_cfg == MODEL_CFG

    def test_revalidating_the_best_checkpoint_reproduces_its_score(self, fold_run):
        assert fold_run.val_metrics["val/loss"] == pytest.approx(fold_run.best_score, abs=1e-5)

    def test_val_predictions_hold_one_row_per_val_slide(self, fold_run):
        df = pd.read_parquet(fold_run.preds_val)
        assert Path(fold_run.preds_val) == fold_run.fold_dir / "preds_val.parquet"
        assert list(df.columns) == ["slide_id", "label", "prob_1"]
        assert sorted(df["slide_id"]) == fold_run.cohort.val_ids
        assert dict(zip(df["slide_id"], df["label"], strict=True)) == {
            s: fold_run.cohort.labels[s] for s in fold_run.cohort.val_ids
        }
        assert df["prob_1"].between(0, 1).all()

    def test_logged_val_metrics_agree_with_the_saved_predictions(self, fold_run):
        df = pd.read_parquet(fold_run.preds_val)
        y, p1 = df["label"].to_numpy(), df["prob_1"].to_numpy()
        pred = (p1 > 0.5).astype(int)
        p_true = np.where(y == 1, p1, 1 - p1)
        m = fold_run.val_metrics
        # Lightning weights each batch's mean loss by its size: the slide-level mean
        assert m["val/loss"] == pytest.approx(-np.log(p_true).mean(), rel=1e-5)
        assert m["val/acc"] == pytest.approx((pred == y).mean())
        assert m["val/auroc"] == pytest.approx(roc_auc_score(y, p1))
        assert m["val/balanced_acc"] == pytest.approx(balanced_accuracy_score(y, pred))

    def test_saved_predictions_and_embeddings_match_the_model_on_unpadded_bags(self, fold_run):
        """Batches were padded to 64 patches; masking must make that invisible."""
        df = pd.read_parquet(fold_run.preds_val).set_index("slide_id")
        npz = np.load(fold_run.emb_val, allow_pickle=True)
        assert Path(fold_run.emb_val) == fold_run.fold_dir / "embeddings_val.npz"
        assert sorted(npz["slide_ids"].tolist()) == fold_run.cohort.val_ids
        assert npz["embeddings"].shape == (len(fold_run.cohort.val_ids), MODEL_CFG["embed_dim"])
        embeddings = dict(zip(npz["slide_ids"].tolist(), npz["embeddings"], strict=True))
        with torch.no_grad():
            for sid in fold_run.cohort.val_ids:
                out = fold_run.best(torch.from_numpy(fold_run.store.features[sid]))
                prob_1 = torch.softmax(out.logits, dim=-1)[0, 1].item()
                assert df.loc[sid, "prob_1"] == pytest.approx(prob_1, rel=1e-5)
                np.testing.assert_allclose(
                    embeddings[sid], out.slide_embedding[0].numpy(), rtol=1e-5, atol=1e-6
                )

    def test_test_stage_reports_test_metrics_and_writes_test_files(self, fold_run):
        assert fold_run.test_metrics.keys() >= TEST_KEYS
        df = pd.read_parquet(fold_run.preds_test)
        assert sorted(df["slide_id"]) == fold_run.cohort.test_ids
        y, p1 = df["label"].to_numpy(), df["prob_1"].to_numpy()
        assert fold_run.test_metrics["test/auroc"] == pytest.approx(roc_auc_score(y, p1))
        npz = np.load(fold_run.emb_test, allow_pickle=True)
        assert sorted(npz["slide_ids"].tolist()) == fold_run.cohort.test_ids

    def test_saving_empties_the_accumulators(self, fold_run):
        """A second save after a fold must not write the same rows again."""
        assert fold_run.best.save_predictions(str(fold_run.fold_dir), prefix="val") is None
        assert fold_run.best.save_embeddings(str(fold_run.fold_dir), prefix="test") is None


class TestCallbacksInFit:
    def test_batch_index_file_lists_every_training_draw(self, fold_run):
        ids = np.load(fold_run.fold_dir / "batch_indices.npy", allow_pickle=True).tolist()
        # class-weighted sampling draws len(train) slides per epoch, with replacement
        assert len(ids) == EPOCHS * len(fold_run.cohort.train_ids)
        assert set(ids) <= set(fold_run.cohort.train_ids)

    def test_metadata_json_records_the_config_and_parameter_counts(self, fold_run):
        meta = json.loads((fold_run.fold_dir / "metadata.json").read_text())
        assert meta["config"] == fold_run.config
        assert meta["num_parameters"] == N_PARAMS
        assert meta["num_trainable"] == N_PARAMS
        assert meta["torch_version"] == torch.__version__

    def test_fold_timing_logs_the_wall_time_of_the_fit(self, fold_run):
        times = [r["fold/wall_time_s"] for r in fold_run.fit_records if "fold/wall_time_s" in r]
        assert len(times) == 1
        assert 0 < times[0] <= fold_run.fit_seconds

    def test_wandb_summary_holds_the_best_val_loss_and_its_epoch(self, fold_run):
        losses = epoch_val_losses(fold_run)
        assert fold_run.summary["fold_0/best_val/loss"] == pytest.approx(min(losses))
        assert fold_run.summary["fold_0/best_epoch"] == int(np.argmin(losses))

    def test_bag_curriculum_caps_training_bags_epoch_by_epoch(self, fold_run):
        # every training slide has more than 12 patches, so each cap is reached exactly
        assert [fold_run.probe.max_len[e] for e in range(EPOCHS)] == CURRICULUM_CAPS


# ═════════════════════════════════════════════════════════════════════════════
# Losses
# ═════════════════════════════════════════════════════════════════════════════


def _ce_terms(logits, labels):
    """Per-sample -log softmax(logits)[label], computed with plain floats."""
    return [
        math.log(sum(math.exp(v) for v in row)) - row[y]
        for row, y in zip(logits, labels, strict=True)
    ]


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class TestLosses:
    LOGITS = [[2.0, 1.0, 0.1], [0.5, 2.5, -1.0]]
    LABELS = [0, 2]

    def _loss(self, module, logits, labels):
        return module._compute_loss(torch.tensor(logits), torch.tensor(labels)).item()

    def test_ce_is_the_mean_negative_log_softmax(self):
        module = make_module(num_classes=3)
        expected = sum(_ce_terms(self.LOGITS, self.LABELS)) / 2
        assert self._loss(module, self.LOGITS, self.LABELS) == pytest.approx(expected, rel=1e-6)

    def test_ce_class_weights_give_a_weighted_mean(self):
        module = make_module(num_classes=3, class_weights=[1.0, 2.0, 3.0])
        t0, t1 = _ce_terms(self.LOGITS, self.LABELS)
        expected = (1.0 * t0 + 3.0 * t1) / (1.0 + 3.0)
        assert self._loss(module, self.LOGITS, self.LABELS) == pytest.approx(expected, rel=1e-6)

    def test_bce_needs_a_single_logit_head(self):
        with pytest.raises(ValueError, match="num_classes=1"):
            make_module(num_classes=2, loss_type="bce")

    def test_bce_uses_the_class_weight_ratio_as_pos_weight(self):
        module = make_module(num_classes=1, loss_type="bce", class_weights=[1.0, 4.0])
        logits, labels = [0.3, -1.2, 2.0], [1, 0, 1]
        terms = [
            -(4.0 * y * math.log(_sigmoid(x)) + (1 - y) * math.log(1 - _sigmoid(x)))
            for x, y in zip(logits, labels, strict=True)
        ]
        assert self._loss(module, logits, labels) == pytest.approx(sum(terms) / 3, rel=1e-6)

    def test_focal_with_gamma_zero_is_cross_entropy(self):
        module = make_module(num_classes=3, loss_type="focal", focal_gamma=0.0)
        expected = sum(_ce_terms(self.LOGITS, self.LABELS)) / 2
        assert self._loss(module, self.LOGITS, self.LABELS) == pytest.approx(expected, rel=1e-6)

    def test_focal_scales_each_term_by_one_minus_p_true_to_the_gamma(self):
        logits, labels = [[2.0, -1.0], [0.2, 0.4]], [0, 1]
        p = [_sigmoid(3.0), _sigmoid(0.2)]  # softmax probability of the true class
        expected = sum((1 - pt) ** 2 * -math.log(pt) for pt in p) / 2
        loss = FocalLoss(gamma=2.0)(torch.tensor(logits), torch.tensor(labels)).item()
        assert loss == pytest.approx(expected, rel=1e-6)

    def test_focal_class_weights_are_alpha_balancing(self):
        """Lin et al.: FL = -alpha_t (1 - p_t)^gamma log(p_t), p_t the model's probability."""
        logits, labels = [[2.0, -1.0], [0.2, 0.4]], [0, 1]
        alpha = [1.0, 3.0]
        p = [_sigmoid(3.0), _sigmoid(0.2)]
        expected = (
            sum(alpha[y] * (1 - pt) ** 2 * -math.log(pt) for pt, y in zip(p, labels, strict=True))
            / 2
        )
        module = make_module(loss_type="focal", focal_gamma=2.0, class_weights=alpha)
        assert self._loss(module, logits, labels) == pytest.approx(expected, rel=1e-6)

    def test_unknown_loss_type_raises(self):
        with pytest.raises(ValueError, match="Unknown loss_type"):
            make_module(loss_type="hinge")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_extreme_logits_are_clamped_before_the_loss(self, dtype):
        module = make_module()
        logits = torch.tensor([[6.0e4, -6.0e4]], dtype=dtype)
        loss = module._compute_loss(logits, torch.tensor([1]))
        # clamped to [100, -100]: -log softmax = 200 + log(1 + e^-200)
        assert loss.dtype == torch.float32
        assert loss.item() == pytest.approx(200.0)


# ═════════════════════════════════════════════════════════════════════════════
# Probabilities, predicted classes, prediction files
# ═════════════════════════════════════════════════════════════════════════════


class TestPredictions:
    def test_two_logit_head_reports_the_positive_class_probability(self):
        module = make_module(num_classes=2)
        logits = torch.tensor([[0.0, 2.0], [1.0, -1.0]])
        torch.testing.assert_close(
            module._get_probs(logits), torch.tensor([_sigmoid(2.0), _sigmoid(-2.0)])
        )
        assert module._get_preds(logits).tolist() == [1, 0]

    @pytest.mark.parametrize("shape", [(3,), (3, 1)])
    def test_single_logit_head_uses_the_sigmoid(self, shape):
        module = make_module(num_classes=1, loss_type="bce")
        logits = torch.tensor([0.5, -0.3, 0.0]).reshape(shape)
        expected = torch.tensor([_sigmoid(0.5), _sigmoid(-0.3), 0.5])
        torch.testing.assert_close(module._get_probs(logits), expected)
        assert module._get_preds(logits).tolist() == [1, 0, 0]

    def test_multiclass_probabilities_are_softmax_rows(self):
        module = make_module(num_classes=3)
        logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, -1.0]])
        probs = module._get_probs(logits)
        torch.testing.assert_close(probs.sum(dim=1), torch.ones(2))
        assert probs.argmax(dim=1).tolist() == module._get_preds(logits).tolist() == [0, 1]

    @pytest.mark.filterwarnings("ignore:You are trying to `self.log")
    def test_multiclass_prediction_file_has_a_column_per_class(self, tmp_path):
        """Exercised through validation_step on a non-ABMIL aggregator (no fp32 override)."""
        module = make_module(
            arch="static", num_classes=3, model_cfg={"embed_dim": 8, "dropout": 0.0}
        )
        module.eval()
        batch = bag_batch(labels=(2, 0, 1), slide_ids=("s2", "s0", "s1"))
        with torch.no_grad():
            module.validation_step(batch, 0)
            expected = torch.softmax(module(batch["features"], mask=batch["mask"]).logits, dim=-1)
        df = pd.read_parquet(module.save_predictions(str(tmp_path), prefix="val"))
        assert list(df.columns) == ["slide_id", "label", "prob_0", "prob_1", "prob_2"]
        assert df["slide_id"].tolist() == ["s2", "s0", "s1"]
        assert df["label"].tolist() == [2, 0, 1]
        np.testing.assert_allclose(
            df[["prob_0", "prob_1", "prob_2"]].to_numpy(), expected.numpy(), rtol=1e-5
        )
        npz = np.load(module.save_embeddings(str(tmp_path), prefix="val"), allow_pickle=True)
        assert npz["slide_ids"].tolist() == ["s2", "s0", "s1"]
        assert npz["embeddings"].shape == (3, 8)

    def test_nothing_to_save_returns_none_and_writes_nothing(self, tmp_path):
        module = make_module()
        assert module.save_predictions(str(tmp_path), prefix="test") is None
        assert module.save_embeddings(str(tmp_path), prefix="val") is None
        assert list(tmp_path.iterdir()) == []


# ═════════════════════════════════════════════════════════════════════════════
# Non-finite loss
# ═════════════════════════════════════════════════════════════════════════════


class TestNonFiniteLoss:
    def test_nan_batch_leaves_the_weights_finite_and_training_continues(self):
        module = make_module(lr=1e-2, weight_decay=0.0, lr_scheduler="none")
        initial = {k: v.clone() for k, v in module.state_dict().items()}
        batches = [bag_batch(nan=True), bag_batch(seed=1), bag_batch(seed=2)]
        trainer = L.Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            precision="32-true",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            gradient_clip_val=1.0,
        )
        trainer.fit(module, train_dataloaders=DataLoader(batches, batch_size=None))
        assert module._nan_count == 1  # only the poisoned batch was non-finite
        for name, param in module.named_parameters():
            assert torch.isfinite(param).all(), name
        assert any(not torch.equal(initial[k], v) for k, v in module.state_dict().items())


# ═════════════════════════════════════════════════════════════════════════════
# Optimizer and LR schedule
# ═════════════════════════════════════════════════════════════════════════════


def _lr_trajectory(module, n_epochs):
    """LR of the first param group at the start of each epoch, plus after the last."""
    config = module.configure_optimizers()
    optimizer, scheduler = config["optimizer"], config["lr_scheduler"]
    lrs = []
    for _ in range(n_epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    return lrs


def _cosine(base, eta_min, k, t_max):
    return eta_min + (base - eta_min) * (1 + math.cos(math.pi * k / t_max)) / 2


class TestConfigureOptimizers:
    def _trainable_ids(self, module):
        return {id(p) for p in module.parameters() if p.requires_grad}

    def test_one_group_with_the_base_lr_by_default(self):
        module = make_module(lr=3e-4, weight_decay=0.05, lr_scheduler="none")
        optimizer = module.configure_optimizers()
        assert isinstance(optimizer, torch.optim.AdamW)
        (group,) = optimizer.param_groups
        assert (group["lr"], group["weight_decay"]) == (3e-4, 0.05)
        assert {id(p) for p in group["params"]} == self._trainable_ids(module)

    def test_differential_lr_splits_aggregator_and_head(self):
        module = make_module(lr=1e-3, aggregator_lr=1e-5, head_lr=1e-4, lr_scheduler="none")
        agg, head = module.configure_optimizers().param_groups
        assert (agg["lr"], head["lr"]) == (1e-5, 1e-4)
        agg_ids, head_ids = {id(p) for p in agg["params"]}, {id(p) for p in head["params"]}
        assert agg_ids == {id(p) for p in module.model.aggregator.parameters()}
        assert head_ids == {id(p) for p in module.model.head.parameters()}

    def test_head_lr_alone_keeps_the_base_lr_for_the_aggregator(self):
        module = make_module(lr=1e-3, head_lr=1e-2, lr_scheduler="none")
        agg, head = module.configure_optimizers().param_groups
        assert (agg["lr"], head["lr"]) == (1e-3, 1e-2)

    def test_equal_aggregator_and_head_lr_keep_one_group(self):
        module = make_module(lr=1e-3, aggregator_lr=1e-4, head_lr=1e-4, lr_scheduler="none")
        (group,) = module.configure_optimizers().param_groups
        assert group["lr"] == 1e-3  # identical rates: no split, base lr applies

    def test_a_frozen_aggregator_is_left_out_of_the_optimizer(self):
        module = make_module(freeze_aggregator=True, lr_scheduler="none")
        (group,) = module.configure_optimizers().param_groups
        assert {id(p) for p in group["params"]} == {id(p) for p in module.model.head.parameters()}

    def test_cosine_with_linear_warmup(self):
        base, eta_min = 1e-3, 1e-7
        module = make_module(lr=base, lr_scheduler="cosine", warmup_epochs=2, max_epochs=6)
        expected = [
            base * 0.5,  # LinearLR starts at 1/warmup
            base * 0.75,  # and climbs linearly to the base lr at epoch `warmup`
            *[_cosine(base, eta_min, k, 4) for k in range(4)],  # cosine over the remaining 4
            eta_min,
        ]
        assert _lr_trajectory(module, 6) == pytest.approx(expected, rel=1e-6)

    def test_cosine_without_warmup_anneals_to_eta_min(self):
        module = make_module(lr=1e-3, lr_scheduler="cosine", warmup_epochs=0, max_epochs=4)
        lrs = _lr_trajectory(module, 4)
        assert lrs == pytest.approx([_cosine(1e-3, 1e-7, k, 4) for k in range(5)], rel=1e-6)

    def test_step_schedule_halves_the_lr_every_ten_epochs(self):
        module = make_module(lr=1e-3, lr_scheduler="step", max_epochs=30)
        lrs = _lr_trajectory(module, 21)
        assert lrs[0] == lrs[9] == pytest.approx(1e-3)
        assert lrs[10] == lrs[19] == pytest.approx(5e-4)
        assert lrs[20] == pytest.approx(2.5e-4)

    def test_plateau_schedule_monitors_the_configured_metric(self):
        module = make_module(
            lr_scheduler="plateau", monitor_metric="val/auroc", monitor_mode="max", warmup_epochs=3
        )
        config = module.configure_optimizers()
        sched = config["lr_scheduler"]
        assert sched["monitor"] == "val/auroc"
        assert sched["interval"] == "epoch"
        assert isinstance(sched["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert sched["scheduler"].mode == "max"

    def test_unknown_scheduler_raises(self):
        with pytest.raises(ValueError, match="Unknown lr_scheduler"):
            make_module(lr_scheduler="onecycle").configure_optimizers()


# ═════════════════════════════════════════════════════════════════════════════
# Gathering predictions across DDP ranks
# ═════════════════════════════════════════════════════════════════════════════


class TestGatherAcrossRanks:
    RANK0 = [{"slide_id": "a", "prob_1": 0.1}, {"slide_id": "b", "prob_1": 0.2}]
    # DistributedSampler pads the last rank with repeats so every rank sees as many items
    RANK1 = [{"slide_id": "c", "prob_1": 0.3}, {"slide_id": "a", "prob_1": 0.1}]

    def test_without_ddp_items_come_back_unchanged(self):
        items = list(self.RANK0)
        assert MILTrainModule._gather_objects_across_ranks(items, dedupe_key="slide_id") is items

    def test_ddp_union_drops_the_sampler_padding(self, monkeypatch):
        dist = torch.distributed

        def all_gather_object(out, obj):
            out[0], out[1] = self.RANK0, self.RANK1

        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: 2)
        monkeypatch.setattr(dist, "all_gather_object", all_gather_object)
        merged = MILTrainModule._gather_objects_across_ranks(self.RANK0, dedupe_key="slide_id")
        assert [r["slide_id"] for r in merged] == ["a", "b", "c"]
        undeduped = MILTrainModule._gather_objects_across_ranks(self.RANK0)
        assert [r["slide_id"] for r in undeduped] == ["a", "b", "c", "a"]
