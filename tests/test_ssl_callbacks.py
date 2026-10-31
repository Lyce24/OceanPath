"""
Tests for the in-training probe and quality callbacks (``oceanpath.ssl.modules.callbacks``).

tests/test_ssl_pretrain.py covers RankMe / alpha-ReQ maths and the combined quality callback
in a Trainer; this file covers LinearProbeEvalCallback and the quality callbacks' schedules,
and tests/test_ssl_distributed.py their DDP paths. The SSL module is a tiny VICReg
SSLPretrainModule fit on CPU; the probe tasks read memmap stores built with the repo's
builder from patches with a planted class signal.

Covers:
  - LinearProbeEvalCallback in a Trainer fit: runs at fit start and on scheduled validation
    epochs only, writes one summary per task and tag, logs lp/<task>/* to the loggers, puts
    the weighted composite and its sources in callback_metrics, leaves the aggregator
    training, and reports exactly what the standalone sklearn runner computes from the same
    aggregator on the full labeled set (label fractions are never wired through it)
  - Protocols through trainer.validate: grouped CV, predefined split and external test;
    tasks sharing a memmap reuse one extraction; a failing task raises, or is skipped
    (and the composite withheld) when fail_on_error=False
  - Configuration: argument checks, default composite weights, schedule, W&B allow-lists,
    store filtering
  - Quality callbacks: compute-every-n schedules, too few embeddings, non-finite embeddings
"""

import dataclasses
import gc
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import h5py
import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch
from lightning.pytorch.loggers import Logger
from torch.utils.data import DataLoader

from oceanpath.data.mmap_builder import MmapBuildConfig, build_mmap
from oceanpath.modules import extract_slide_embeddings as ese
from oceanpath.modules import linear_probing_sklearn as lp
from oceanpath.ssl.modules.callbacks import (
    AlphaReQCallback,
    LinearProbeEvalCallback,
    LinearProbeTask,
    RankMeCallback,
    SSLQualityCallback,
)
from oceanpath.ssl.modules.pretrain_module import SSLPretrainModule

D_IN = 8
EMBED = 16
SEED = 42

# ═════════════════════════════════════════════════════════════════════════════
# Fixtures and helpers
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module", autouse=True)
def _cheap_gc_collect():
    """The callbacks call gc.collect() after every hook; with torch, Lightning and sklearn
    loaded a full collection takes ~0.25 s. Freezing the objects that exist before these
    tests limits each collection to the objects the tests create."""
    gc.freeze()
    yield
    gc.unfreeze()


def _build_labeled_store(root, n_patients: int, *, seed: int) -> tuple[str, str]:
    """Two slides per patient; class-1 patients' patch features are shifted by +1.5.

    Returns (mmap_dir, manifest_csv). The manifest also carries a train/test split
    column (first two thirds of the patients train).
    """
    rng = np.random.default_rng(seed)
    h5_dir = root / "h5"
    h5_dir.mkdir(parents=True)
    grid = np.stack(np.meshgrid(np.arange(8), np.arange(8), indexing="ij"), -1).reshape(-1, 2)
    rows = []
    for p in range(n_patients):
        label = p % 2
        for s in range(2):
            slide_id = f"{root.name}_p{p:02d}_s{s}"
            n = int(rng.integers(6, 13))
            feats = rng.standard_normal((n, D_IN)).astype(np.float32) + 1.5 * label
            coords = (grid[rng.choice(len(grid), size=n, replace=False)] * 256).astype(np.int32)
            with h5py.File(h5_dir / f"{slide_id}.h5", "w") as f:
                f["features"] = feats
                f["coords"] = coords
            rows.append(
                {
                    "filename": f"{slide_id}.h5",
                    "label": label,
                    "patient_id": f"{root.name}_p{p:02d}",
                    "split": "train" if p < 2 * n_patients // 3 else "test",
                }
            )
    mmap_dir = root / "mmap"
    build_mmap(
        MmapBuildConfig(
            h5_dir=str(h5_dir),
            output_dir=str(mmap_dir),
            feat_precision=32,
            save_coverage_thumbnails=False,
        )
    )
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return str(mmap_dir), str(manifest)


@pytest.fixture(scope="module")
def cohorts(tmp_path_factory):
    src = _build_labeled_store(tmp_path_factory.mktemp("src"), 12, seed=0)
    tgt = _build_labeled_store(tmp_path_factory.mktemp("tgt"), 6, seed=1)
    return SimpleNamespace(src_mmap=src[0], src_csv=src[1], tgt_mmap=tgt[0], tgt_csv=tgt[1])


def _task(cohorts, name: str, protocol: str = "grouped_cv", **overrides) -> LinearProbeTask:
    params = {
        "name": name,
        "protocol": protocol,
        "mmap_dir": cohorts.src_mmap,
        "manifest_csv": cohorts.src_csv,
        "label_column": "label",
        "patient_column": "patient_id",
        "n_splits": 3,
        "inner_splits": 0,
        "c_grid": (1.0,),
        "batch_size": 4,
        "num_workers": 0,
        "seed": SEED,
    }
    if protocol == "external_test":
        params.update(test_mmap_dir=cohorts.tgt_mmap, test_manifest_csv=cohorts.tgt_csv)
    if protocol == "predefined_split":
        params["split_column"] = "split"
    params.update(overrides)
    return LinearProbeTask(**params)


def _module(seed: int = 0) -> SSLPretrainModule:
    torch.manual_seed(seed)
    return SSLPretrainModule(
        ssl_method="vicreg",
        arch="abmil",
        in_dim=D_IN,
        model_cfg={"embed_dim": EMBED, "attn_dim": 8, "dropout": 0.0},
        ssl_cfg={"proj_hidden_dim": 16, "proj_out_dim": 16, "proj_num_layers": 2},
        lr=1e-3,
        warmup_epochs=0,
        max_epochs=3,
        lr_scheduler="none",
    )


def _views(n_batches: int, *, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    batches = [
        {
            "view1": torch.randn(4, 10, D_IN, generator=g),
            "view2": torch.randn(4, 10, D_IN, generator=g),
        }
        for _ in range(n_batches)
    ]
    return DataLoader(batches, batch_size=None)


class _RecordingLogger(Logger):
    """Keeps every log_metrics call, as a W&B logger would receive it."""

    def __init__(self):
        super().__init__()
        self.records: list[tuple[int | None, dict]] = []

    @property
    def name(self):
        return "recording"

    @property
    def version(self):
        return 0

    def log_hyperparams(self, params, *args, **kwargs):
        pass

    def log_metrics(self, metrics, step=None):
        self.records.append((step, dict(metrics)))

    def direct(self, key: str) -> list[tuple[int | None, dict]]:
        """Records the callback sent itself (Lightning adds 'epoch' to its own)."""
        return [(step, m) for step, m in self.records if key in m and "epoch" not in m]

    def direct_steps(self, key: str) -> list[int | None]:
        return [step for step, _ in self.direct(key)]

    def epochs_with(self, key: str) -> list[int]:
        """Epochs at which Lightning flushed a value logged with pl_module.log()."""
        return sorted({int(m["epoch"]) for _, m in self.records if key in m and "epoch" in m})

    def last(self, key: str) -> float:
        return [float(m[key]) for _, m in self.records if key in m][-1]


class _AggregatorModes(L.Callback):
    def __init__(self):
        self.train_batch_modes: list[bool] = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.train_batch_modes.append(pl_module.aggregator.training)


def _trainer(logger, callbacks, **kwargs) -> L.Trainer:
    params = {
        "accelerator": "cpu",
        "devices": 1,
        "logger": logger,
        "callbacks": callbacks,
        "num_sanity_val_steps": 0,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    params.update(kwargs)
    return L.Trainer(**params)


def _summary(path) -> dict:
    return json.loads(path.read_text())


# ═════════════════════════════════════════════════════════════════════════════
# LinearProbeEvalCallback inside a Trainer fit
# ═════════════════════════════════════════════════════════════════════════════


class TestLinearProbeEvalInFit:
    """One 3-epoch fit: probe at fit start and every 2nd epoch, quality every 2nd epoch."""

    WEIGHTS = (0.25, 0.75)

    @pytest.fixture(scope="class")
    def fit(self, cohorts, tmp_path_factory):
        out = tmp_path_factory.mktemp("lp_eval")
        probe = LinearProbeEvalCallback(
            tasks=[_task(cohorts, "cv"), _task(cohorts, "xfer", "external_test")],
            every_n_epochs=2,
            output_dir=out,
            run_at_fit_start=True,
            composite_metric_name="lp/composite",
            composite_metric_sources=["lp/cv/mean_auroc", "lp/xfer/auroc"],
            composite_metric_weights=list(self.WEIGHTS),
        )
        modes = _AggregatorModes()
        logger = _RecordingLogger()
        module = _module()
        trainer = _trainer(
            logger, [probe, SSLQualityCallback(compute_every_n_epochs=2), modes], max_epochs=3
        )
        trainer.fit(module, train_dataloaders=_views(1), val_dataloaders=_views(1, seed=1))
        return SimpleNamespace(
            trainer=trainer, module=module, logger=logger, modes=modes, out=out, probe=probe
        )

    def test_probe_runs_at_fit_start_and_on_scheduled_epochs_only(self, fit):
        assert sorted(p.name for p in fit.out.iterdir()) == ["epoch_002", "fit_start"]
        for tag in ("fit_start", "epoch_002"):
            assert _summary(fit.out / tag / "cv" / "summary.json")["protocol"] == "grouped_cv"
            assert _summary(fit.out / tag / "xfer" / "summary.json")["protocol"] == "external_test"
        # Sent straight to the loggers at fit start and after epoch 2 (the last step).
        final_step = fit.trainer.global_step
        assert fit.logger.direct_steps("lp/cv/mean_auroc") == [0, final_step]
        assert fit.logger.direct_steps("lp/xfer/auroc") == [0, final_step]
        assert fit.logger.direct_steps("lp/xfer/label_fraction") == [0, final_step]

    def test_composite_is_the_weighted_sum_and_reaches_callback_metrics(self, fit):
        w_cv, w_xfer = self.WEIGHTS
        direct = fit.logger.direct("lp/composite")
        assert len(direct) == 2
        for _, metrics in direct:
            expected = w_cv * metrics["lp/cv/mean_auroc"] + w_xfer * metrics["lp/xfer/auroc"]
            assert metrics["lp/composite"] == pytest.approx(expected)

        # pl_module.log (what ModelCheckpoint reads) is only allowed on validation epochs.
        assert fit.logger.epochs_with("lp/composite") == [2]
        cm = fit.trainer.callback_metrics
        assert float(cm["lp/composite"]) == pytest.approx(fit.logger.last("lp/composite"))
        assert float(cm["lp/cv/mean_auroc"]) == pytest.approx(fit.logger.last("lp/cv/mean_auroc"))
        assert float(cm["lp/xfer/auroc"]) == pytest.approx(fit.logger.last("lp/xfer/auroc"))

    def test_separable_cohorts_probe_near_perfectly(self, fit):
        assert fit.logger.last("lp/cv/mean_auroc") >= 0.9
        assert fit.logger.last("lp/xfer/auroc") >= 0.9

    def test_probe_matches_the_standalone_runner_on_the_full_labeled_set(
        self, fit, cohorts, tmp_path
    ):
        """The epoch-2 probe saw the final weights; the standalone runner on embeddings
        extracted from the same aggregator must report the same numbers, and both use
        every labeled patient (label-efficiency runs are the standalone runner's job)."""
        manifest = lp.load_manifest(cohorts.src_csv, None, "filename", "label", "patient_id")
        loader = ese.build_loader(
            mmap_dir=cohorts.src_mmap,
            slide_ids=sorted(manifest["slide_id"]),
            mode="aggregator",
            batch_size=4,
            num_workers=0,
            max_instances=None,
        )
        emb, ids = ese.extract_aggregator(loader, fit.module.aggregator, torch.device("cpu"))
        df = lp.merge_manifest_and_embeddings(manifest, lp.EmbeddingStore(ids, emb))
        args = lp.make_probe_args(
            output_dir=str(tmp_path), n_splits=3, inner_splits=0, c_grid=[1.0], seed=SEED
        )
        expected = lp.run_grouped_cv(args, df)

        run_dir = fit.out / "epoch_002" / "cv"
        pd.testing.assert_frame_equal(
            pd.read_csv(run_dir / "patient_predictions_oof.csv"),
            pd.read_csv(tmp_path / "patient_predictions_oof.csv"),
            rtol=1e-5,
        )
        logged = _summary(run_dir / "summary.json")
        assert logged["mean_auroc"] == pytest.approx(expected["mean_auroc"])
        assert logged["label_fraction"] == expected["label_fraction"] == 1.0
        assert logged["mean_train_labeled_patient_fraction"] == 1.0
        assert "oof_auroc_ci_low" not in logged  # no bootstrap inside the training loop
        xfer = _summary(fit.out / "epoch_002" / "xfer" / "summary.json")
        assert xfer["train_labeled_patient_fraction"] == 1.0
        assert xfer["n_train_patients"] == xfer["n_train_patients_full"] == 12

    def test_aggregator_keeps_training_after_the_fit_start_probe(self, fit):
        assert fit.modes.train_batch_modes == [True, True, True]

    def test_quality_metrics_follow_their_own_schedule(self, fit):
        assert fit.logger.epochs_with("ssl/rankme") == [0, 2]
        assert fit.logger.epochs_with("ssl/alpha_req") == [0, 2]
        assert 1.0 <= fit.logger.last("ssl/rankme") <= EMBED


# ═════════════════════════════════════════════════════════════════════════════
# Protocols and failures through trainer.validate
# ═════════════════════════════════════════════════════════════════════════════


class TestLinearProbeEvalInValidate:
    @staticmethod
    def _validate(callback, logger=None):
        trainer = _trainer(logger or _RecordingLogger(), [callback])
        trainer.validate(_module(), dataloaders=_views(1), verbose=False)
        return trainer

    def test_all_protocols_share_one_extraction_per_store(self, cohorts, tmp_path, monkeypatch):
        calls = []
        real_extract = ese.extract_aggregator

        def counting_extract(loader, model, device):
            calls.append(len(loader.dataset))
            return real_extract(loader, model, device)

        monkeypatch.setattr(ese, "extract_aggregator", counting_extract)
        tasks = [
            _task(cohorts, "cv"),
            _task(cohorts, "split", "predefined_split"),
            _task(cohorts, "xfer", "external_test"),
        ]
        logger = _RecordingLogger()

        self._validate(
            LinearProbeEvalCallback(tasks, output_dir=tmp_path, run_at_epoch0=True), logger
        )

        # The source store is read once for both of its tasks (24 slides), the target once.
        assert sorted(calls) == [12, 24]
        epoch0 = tmp_path / "epoch_000"
        assert _summary(epoch0 / "cv" / "summary.json")["protocol"] == "grouped_cv"
        split = _summary(epoch0 / "split" / "summary.json")
        assert split["n_train_patients"] == 8  # patients 0-7 are "train"
        assert split["n_test_patients"] == 4
        xfer = _summary(epoch0 / "xfer" / "summary.json")
        assert xfer["n_train_patients"] == 12
        assert xfer["n_test_patients"] == 6
        for key in ("lp/cv/mean_auroc", "lp/split/auroc", "lp/xfer/auroc"):
            assert logger.direct_steps(key) == [0]

    def test_failing_task_raises_by_default(self, cohorts, tmp_path):
        too_many_folds = _task(cohorts, "bad", n_splits=13)  # 12 patients
        callback = LinearProbeEvalCallback(
            [too_many_folds, _task(cohorts, "cv")], output_dir=tmp_path, run_at_epoch0=True
        )
        with pytest.raises(RuntimeError, match="task='bad'.*Need at least 13 unique groups"):
            self._validate(callback)
        assert not (tmp_path / "epoch_000" / "cv").exists()  # stopped at the failure

    def test_failing_task_is_skipped_and_the_composite_withheld(self, cohorts, tmp_path):
        callback = LinearProbeEvalCallback(
            [_task(cohorts, "bad", n_splits=13), _task(cohorts, "cv")],
            output_dir=tmp_path,
            run_at_epoch0=True,
            fail_on_error=False,
            composite_metric_name="lp/composite",
            composite_metric_sources=["lp/bad/mean_auroc", "lp/cv/mean_auroc"],
        )
        logger = _RecordingLogger()

        trainer = self._validate(callback, logger)

        assert logger.direct_steps("lp/cv/mean_auroc") == [0]
        assert logger.direct_steps("lp/bad/mean_auroc") == []
        assert "lp/composite" not in trainer.callback_metrics
        assert not any("lp/composite" in m for _, m in logger.records)

    def test_external_test_without_a_test_cohort_fails(self, cohorts, tmp_path):
        task = _task(cohorts, "xfer", "external_test", test_mmap_dir=None)
        callback = LinearProbeEvalCallback([task], output_dir=tmp_path, run_at_epoch0=True)
        with pytest.raises(RuntimeError, match="test_mmap_dir or test_manifest_csv is missing"):
            self._validate(callback)

    def test_unknown_protocol_fails(self, cohorts, tmp_path):
        task = _task(cohorts, "loso", "leave_one_site_out")
        callback = LinearProbeEvalCallback([task], output_dir=tmp_path, run_at_epoch0=True)
        with pytest.raises(RuntimeError, match="Unknown protocol='leave_one_site_out'"):
            self._validate(callback)


# ═════════════════════════════════════════════════════════════════════════════
# Configuration, schedule, metric allow-lists, store filtering
# ═════════════════════════════════════════════════════════════════════════════


class TestLinearProbeEvalConfig:
    def test_invalid_settings_rejected(self):
        with pytest.raises(ValueError, match="every_n_epochs must be positive"):
            LinearProbeEvalCallback([], every_n_epochs=0)
        with pytest.raises(ValueError, match="without composite_metric_sources"):
            LinearProbeEvalCallback([], composite_metric_name="lp/c")
        with pytest.raises(ValueError, match="has 1 entries but composite_metric_sources has 2"):
            LinearProbeEvalCallback(
                [],
                composite_metric_name="lp/c",
                composite_metric_sources=["a", "b"],
                composite_metric_weights=[1.0],
            )

    def test_label_fraction_is_never_a_task_setting(self, cohorts, tmp_path):
        """In-training probes always use every labeled patient and skip bootstrap CIs."""
        assert "label_fraction" not in {f.name for f in dataclasses.fields(LinearProbeTask)}
        cb = LinearProbeEvalCallback([_task(cohorts, "t")])
        args = cb._build_probe_args(cb.tasks[0], tmp_path)
        assert args.label_fraction is None
        assert args.n_boot == 0

    def test_composite_weights_default_to_a_uniform_mean(self):
        cb = LinearProbeEvalCallback(
            [], composite_metric_name="lp/c", composite_metric_sources=["a", "b", "c", "d"]
        )
        assert cb.composite_metric_weights == [0.25] * 4

    @pytest.mark.parametrize(
        ("epoch", "sanity", "run_at_epoch0", "expected"),
        [
            (0, False, False, False),
            (0, False, True, True),
            (0, True, True, False),
            (4, False, False, False),
            (5, False, False, True),
            (10, False, False, True),
            (10, True, False, False),
        ],
    )
    def test_schedule(self, epoch, sanity, run_at_epoch0, expected):
        cb = LinearProbeEvalCallback([], every_n_epochs=5, run_at_epoch0=run_at_epoch0)
        trainer = SimpleNamespace(current_epoch=epoch, sanity_checking=sanity)
        assert cb._should_run_validation_epoch(trainer) is expected

    SUMMARY = {
        "protocol": "grouped_cv",
        "n_folds": 3,
        "mean_auroc": 0.9,
        "std_auroc": 0.05,
        "mean_auprc": 0.8,
        "mean_balanced_acc": 0.7,
        "mean_macro_f1": 0.6,
        "mean_auroc_macro_ovr": 0.85,
        "fold_metrics": [{"auroc": 0.9}],
    }

    def test_every_numeric_summary_field_is_logged_by_default(self, cohorts):
        flat = LinearProbeEvalCallback._flatten_summary(_task(cohorts, "t"), self.SUMMARY)
        assert flat == {
            f"lp/t/{k}": float(v)
            for k, v in self.SUMMARY.items()
            if k not in {"protocol", "fold_metrics"}
        }

    def test_allow_list_keeps_only_the_named_metrics(self, cohorts):
        task = _task(cohorts, "t", wandb_log_metrics=("AUROC", "acc"))
        flat = LinearProbeEvalCallback._flatten_summary(task, self.SUMMARY)
        assert flat == {
            "lp/t/mean_auroc": 0.9,
            "lp/t/mean_auroc_macro_ovr": 0.85,
            "lp/t/mean_balanced_acc": 0.7,
        }

    def test_unknown_allow_list_name_rejected(self, cohorts):
        task = _task(cohorts, "t", wandb_log_metrics=("auroc", "kappa"))
        with pytest.raises(ValueError, match="Unknown LP metric name 'kappa'"):
            LinearProbeEvalCallback._flatten_summary(task, self.SUMMARY)

    def test_store_filter_follows_the_manifest(self):
        store = lp.EmbeddingStore(
            slide_ids=["a", "b", "c"], embeddings=np.arange(6, dtype=np.float32).reshape(3, 2)
        )

        same = LinearProbeEvalCallback._filter_store(
            store, pd.DataFrame({"slide_id": ["a", "b", "c"]})
        )
        picked = LinearProbeEvalCallback._filter_store(
            store, pd.DataFrame({"slide_id": ["c", "x", "a"]})
        )
        empty = LinearProbeEvalCallback._filter_store(store, pd.DataFrame({"slide_id": ["x"]}))

        assert same is store
        assert picked.slide_ids == ["c", "a"]
        np.testing.assert_array_equal(picked.embeddings, [[4, 5], [0, 1]])
        assert empty.slide_ids == []
        assert empty.embeddings.shape == (0, 2)


# ═════════════════════════════════════════════════════════════════════════════
# Quality callbacks: schedules and degenerate embeddings
# ═════════════════════════════════════════════════════════════════════════════

QUALITY_CALLBACKS = [RankMeCallback, AlphaReQCallback, SSLQualityCallback]


def _feed(cb, epoch: int, embeddings: list[torch.Tensor]):
    trainer = MagicMock(current_epoch=epoch, world_size=1, is_global_zero=True)
    pl_module = MagicMock()
    for i, z in enumerate(embeddings):
        cb.on_validation_batch_end(trainer, pl_module, {"embeddings": z}, None, i)
    buffered = len(cb._embeddings)
    cb.on_validation_epoch_end(trainer, pl_module)
    return buffered, {c.args[0]: c.args[1] for c in pl_module.log.call_args_list}


class TestQualityCallbacks:
    @pytest.mark.parametrize("cls", QUALITY_CALLBACKS)
    def test_off_schedule_epochs_collect_and_log_nothing(self, cls):
        cb = cls(compute_every_n_epochs=3)
        z = [torch.randn(8, 16)]

        assert _feed(cb, 4, z) == (0, {})
        buffered, logged = _feed(cb, 6, z)
        assert buffered == 1
        assert logged
        assert cb._embeddings == []

    @pytest.mark.parametrize("cls", QUALITY_CALLBACKS)
    def test_fewer_than_two_embeddings_log_nothing(self, cls):
        cb = cls(compute_every_n_epochs=1)
        assert _feed(cb, 0, []) == (0, {})
        assert _feed(cb, 0, [torch.randn(1, 16)]) == (1, {})

    def test_max_samples_caps_the_embeddings_used(self):
        """Four one-hot rows centre to a regular simplex with three equal singular values,
        so their effective rank is exactly 3; the 60 rows after them must not count."""
        z = torch.cat([torch.eye(4, 16), torch.randn(60, 16)])
        _, capped = _feed(RankMeCallback(compute_every_n_epochs=1, max_samples=4), 0, [z])
        _, full = _feed(RankMeCallback(compute_every_n_epochs=1, max_samples=64), 0, [z])
        assert capped["ssl/rankme"] == pytest.approx(3.0, rel=1e-4)
        assert capped["ssl/rankme_ratio"] == pytest.approx(3.0 / 16, rel=1e-4)
        assert full["ssl/rankme"] > 10

    def test_non_finite_embeddings_report_zero_instead_of_failing(self):
        z = torch.randn(32, 16)
        z[3, 5] = float("nan")
        _, logged = _feed(SSLQualityCallback(compute_every_n_epochs=1), 0, [z])
        assert logged["ssl/rankme"] == 0.0
        assert logged["ssl/rankme_ratio"] == 0.0
        assert logged["ssl/alpha_req"] == 0.0
