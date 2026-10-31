"""
Tests for the DDP code paths of the SSL losses and callbacks, on two CPU ranks.

Each test forks two processes that join a gloo process group through a file store, runs
the scenario on both ranks and compares their results with a single-process computation on
the concatenated batch. Forked ranks keep the parent's imports, so a group costs ~0.2 s.
Coverage is not recorded inside the forked ranks.

Covers:
  - VICReg: with uneven local batches the gathered loss equals the single-process loss on
    the global batch, and each rank's gradient is world_size times its slice of the
    single-process gradient (so DDP's gradient averaging restores the global-batch gradient)
  - SIGReg, LeJEPA, LeJEPA-MC: the all-reduced characteristic function gives the
    single-process statistic on uneven batches; in training the loss scales SIGReg by
    world_size (and logs it unscaled), which makes the averaged gradient exact
  - Quality callbacks: embeddings are gathered across ranks (also from an empty rank) and
    every rank logs the metric of the full validation set
  - LinearProbeEvalCallback: slides sharded across ranks are gathered into the same store a
    single process extracts; rank 0 probes and every rank logs the same composite; a
    failing task raises on every rank instead of leaving one waiting
"""

import gc
import os
import sys
import traceback
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from oceanpath.models import build_aggregator
from oceanpath.ssl.modules.callbacks import (
    AlphaReQCallback,
    LinearProbeEvalCallback,
    LinearProbeTask,
    RankMeCallback,
    SSLQualityCallback,
)
from oceanpath.ssl.modules.losses import LeJEPALoss, LeJEPAMCLoss, VICRegLoss, sigreg

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not dist.is_available() or not dist.is_gloo_available(),
    reason="needs fork and the gloo backend",
)

WORLD = 2
CPU = torch.device("cpu")
D_IN = 8


@pytest.fixture(scope="module", autouse=True)
def _frozen_heap():
    """Move the imported heap to gc's permanent generation before forking: the callbacks
    call gc.collect() after every hook, which would otherwise walk (and copy-on-write) the
    whole torch/Lightning/sklearn heap in every rank, ~0.25 s per call."""
    gc.freeze()
    yield
    gc.unfreeze()


def _rank_entry(rank, init_method, fn, args, results):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.set_num_threads(1)
    try:
        dist.init_process_group(
            "gloo",
            init_method=init_method,
            rank=rank,
            world_size=WORLD,
            timeout=timedelta(seconds=60),
        )
        try:
            payload = fn(rank, *args)
        finally:
            dist.destroy_process_group()
        results.put((rank, None, payload))
    except BaseException:
        results.put((rank, traceback.format_exc(), None))


def _run_ranks(tmp_path, fn, *args) -> list:
    """Run ``fn(rank, *args)`` on WORLD forked ranks of one gloo group; results by rank.

    Payloads travel through a queue, so ``fn`` returns numpy arrays rather than tensors.
    """
    ctx = mp.get_context("fork")
    results = ctx.Queue()
    init_method = (tmp_path / "rendezvous").as_uri()
    procs = [
        ctx.Process(target=_rank_entry, args=(rank, init_method, fn, args, results))
        for rank in range(WORLD)
    ]
    for p in procs:
        p.start()
    try:
        out = sorted((results.get(timeout=120) for _ in procs), key=lambda r: r[0])
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
    errors = [err for _, err, _ in out if err]
    assert not errors, "\n".join(errors)
    return [payload for _, _, payload in out]


def _rows(rank: int, sizes: list[int]) -> slice:
    start = sum(sizes[:rank])
    return slice(start, start + sizes[rank])


class _Host(torch.nn.Module):
    """What the callbacks use from the LightningModule: aggregator, device and log()."""

    def __init__(self, aggregator=None):
        super().__init__()
        self.aggregator = aggregator
        self.logged: dict[str, float] = {}

    @property
    def device(self):
        return CPU

    def log(self, name, value, **kwargs):
        self.logged[name] = float(value)


def _ddp_trainer(rank: int, epoch: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        world_size=WORLD,
        global_rank=rank,
        is_global_zero=rank == 0,
        current_epoch=epoch,
        sanity_checking=False,
        global_step=0,
        loggers=[],
        strategy=SimpleNamespace(barrier=lambda *a, **k: dist.barrier()),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Losses
# ═════════════════════════════════════════════════════════════════════════════


def _vicreg_rank(rank, z1, z2, sizes):
    rows = _rows(rank, sizes)
    a = z1[rows].clone().requires_grad_(True)
    b = z2[rows].clone().requires_grad_(True)
    out = VICRegLoss()(a, b)
    out["loss"].backward()
    terms = {k: float(v) for k, v in out.items()}
    return terms, a.grad.numpy(), b.grad.numpy()


def _sigreg_rank(rank, z1, z2, sizes, weight):
    rows = _rows(rank, sizes)
    a = z1[rows].clone().requires_grad_(True)
    b = z2[rows].clone().requires_grad_(True)
    values = {
        "sigreg": float(sigreg(a.detach(), global_step=5, num_slices=32)),
        "eval": LeJEPALoss(weight, num_slices=32).eval()(a.detach(), b.detach(), global_step=5),
        "mc": LeJEPAMCLoss(weight, num_slices=32).train()([a, b], num_global=1, global_step=5),
    }
    train = LeJEPALoss(weight, num_slices=32).train()(a, b, global_step=5)
    train["loss"].backward()
    values["train"] = train
    values = {
        k: v if isinstance(v, float) else {n: float(t) for n, t in v.items() if n[0] != "_"}
        for k, v in values.items()
    }
    return values, a.grad.numpy(), b.grad.numpy()


class TestDistributedLosses:
    def test_vicreg_sees_the_global_batch_and_scales_gradients(self, tmp_path):
        torch.manual_seed(0)
        z1, z2 = torch.randn(8, 6), torch.randn(8, 6)
        sizes = [3, 5]

        results = _run_ranks(tmp_path, _vicreg_rank, z1, z2, sizes)

        a, b = z1.clone().requires_grad_(True), z2.clone().requires_grad_(True)
        ref = VICRegLoss()(a, b)
        ref["loss"].backward()
        for rank, (terms, grad_a, grad_b) in enumerate(results):
            for key in ("loss", "inv_loss", "var_loss", "cov_loss"):
                assert terms[key] == pytest.approx(float(ref[key]), rel=1e-5), key
            rows = _rows(rank, sizes)
            np.testing.assert_allclose(grad_a, WORLD * a.grad[rows].numpy(), rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(grad_b, WORLD * b.grad[rows].numpy(), rtol=1e-4, atol=1e-6)

    def test_sigreg_is_the_global_statistic_on_uneven_batches(self, tmp_path):
        torch.manual_seed(1)
        z1, z2 = torch.randn(9, 6), torch.randn(9, 6)
        weight = 0.2

        results = _run_ranks(tmp_path, _sigreg_rank, z1, z2, [2, 7], weight)

        sig1 = float(sigreg(z1, 5, num_slices=32))
        ref_sig = 0.5 * (sig1 + float(sigreg(z2, 5, num_slices=32)))
        for values, _, _ in results:
            assert values["sigreg"] == pytest.approx(sig1, rel=1e-5)
            for key in ("train", "eval", "mc"):
                assert values[key]["sigreg_loss"] == pytest.approx(ref_sig, rel=1e-5)
            # Evaluation keeps the plain mix; training scales SIGReg by the world size.
            ev, tr = values["eval"], values["train"]
            assert ev["loss"] == pytest.approx(
                (1 - weight) * ev["pred_loss"] + weight * ev["sigreg_loss"], rel=1e-5
            )
            assert tr["loss"] == pytest.approx(
                (1 - weight) * tr["pred_loss"] + weight * WORLD * tr["sigreg_loss"], rel=1e-5
            )
            mc = values["mc"]
            assert mc["loss"] == pytest.approx(
                (1 - weight) * mc["pred_loss"] + weight * WORLD * mc["sigreg_loss"], rel=1e-5
            )

    def test_lejepa_world_size_scaling_makes_the_averaged_gradient_exact(self, tmp_path):
        """With equal local batches, rank r's gradient is world_size times its slice of the
        single-process gradient, so DDP's averaging reproduces the global-batch update."""
        torch.manual_seed(2)
        z1, z2 = torch.randn(8, 6), torch.randn(8, 6)
        sizes = [4, 4]

        results = _run_ranks(tmp_path, _sigreg_rank, z1, z2, sizes, 0.2)

        a, b = z1.clone().requires_grad_(True), z2.clone().requires_grad_(True)
        LeJEPALoss(0.2, num_slices=32).train()(a, b, global_step=5)["loss"].backward()
        for rank, (_, grad_a, grad_b) in enumerate(results):
            rows = _rows(rank, sizes)
            np.testing.assert_allclose(grad_a, WORLD * a.grad[rows].numpy(), rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(grad_b, WORLD * b.grad[rows].numpy(), rtol=1e-4, atol=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# Quality callbacks
# ═════════════════════════════════════════════════════════════════════════════


QUALITY_CALLBACKS = [SSLQualityCallback, RankMeCallback, AlphaReQCallback]


def _quality_rank(rank, splits):
    """Each callback's logs, once per split of four one-hot rows over the two ranks."""
    logged = []
    for split in splits:
        rows = torch.eye(4, 16)[_rows(rank, split)]
        for cls in QUALITY_CALLBACKS:
            trainer, host, cb = _ddp_trainer(rank), _Host(), cls(compute_every_n_epochs=1)
            if len(rows):
                cb.on_validation_batch_end(trainer, host, {"embeddings": rows}, None, 0)
            cb.on_validation_epoch_end(trainer, host)
            logged.append(host.logged)
    return logged


class TestDistributedQualityCallbacks:
    def test_every_rank_logs_the_metric_of_the_gathered_embeddings(self, tmp_path):
        """Four one-hot rows have effective rank 3; a rank alone would see rank <= 1."""
        results = _run_ranks(tmp_path, _quality_rank, [[2, 2], [4, 0]])

        expected = {
            "ssl/rankme": 3.0,
            "ssl/rankme_ratio": 3.0 / 16,
            "ssl/alpha_req": AlphaReQCallback()._compute_alpha_req(torch.eye(4, 16)),
        }
        keys_per_callback = [
            {"ssl/rankme", "ssl/rankme_ratio", "ssl/alpha_req"},
            {"ssl/rankme", "ssl/rankme_ratio"},
            {"ssl/alpha_req"},
        ]
        # Both ranks log, including the one that collected no embeddings, for both splits.
        assert results[0] == results[1]
        for logged, keys in zip(results[0], keys_per_callback * 2, strict=True):
            assert logged == pytest.approx({k: expected[k] for k in keys}, rel=1e-4, abs=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# LinearProbeEvalCallback
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def probe_setup(build_feature_store, tmp_path_factory):
    """A 12-slide store, a manifest of 6 two-slide patients and a fixed aggregator."""
    store = build_feature_store([4, 6, 5, 7, 3, 8, 5, 6, 4, 7, 6, 5], feat_dim=D_IN, seed=4)
    manifest = tmp_path_factory.mktemp("manifest") / "manifest.csv"
    pd.DataFrame(
        {
            "filename": [f"{sid}.h5" for sid in store.slide_ids],
            "label": [i // 2 % 2 for i in range(12)],
            "patient_id": [f"p{i // 2}" for i in range(12)],
        }
    ).to_csv(manifest, index=False)
    torch.manual_seed(0)
    aggregator = build_aggregator(
        "abmil", in_dim=D_IN, model_cfg={"embed_dim": 8, "attn_dim": 8, "dropout": 0.0}
    ).eval()
    return SimpleNamespace(store=store, manifest=str(manifest), aggregator=aggregator)


def _probe_task(setup, name="cv", **overrides) -> LinearProbeTask:
    params = {
        "name": name,
        "protocol": "grouped_cv",
        "mmap_dir": str(setup.store.mmap_dir),
        "manifest_csv": setup.manifest,
        "label_column": "label",
        "patient_column": "patient_id",
        "n_splits": 3,
        "inner_splits": 0,
        "c_grid": (1.0,),
        "batch_size": 2,
        "num_workers": 0,
    }
    params.update(overrides)
    return LinearProbeTask(**params)


def _composite_callback(setup, output_dir, **kwargs) -> LinearProbeEvalCallback:
    return LinearProbeEvalCallback(
        [_probe_task(setup)],
        every_n_epochs=1,
        output_dir=output_dir,
        composite_metric_name="lp/composite",
        composite_metric_sources=["lp/cv/mean_auroc", "lp/cv/mean_balanced_acc"],
        **kwargs,
    )


def _extract_rank(rank, setup, slide_ids):
    cb = LinearProbeEvalCallback([_probe_task(setup)])
    store = cb._extract_distributed(
        slide_ids=slide_ids,
        model=setup.aggregator,
        device=CPU,
        mmap_dir=str(setup.store.mmap_dir),
        batch_size=2,
        num_workers=0,
        max_instances=None,
        trainer=_ddp_trainer(rank),
    )
    return store.slide_ids, store.embeddings


def _probe_rank(rank, setup, output_dir):
    host = _Host(setup.aggregator)
    callback = _composite_callback(setup, output_dir / f"rank{rank}")
    callback.on_validation_epoch_end(_ddp_trainer(rank, 2), host)
    return host.logged


def _failing_probe_rank(rank, setup, output_dir):
    bad = _probe_task(setup, name="bad", n_splits=7)  # only 6 patients
    cb = LinearProbeEvalCallback([bad], every_n_epochs=1, output_dir=output_dir)
    try:
        cb.on_validation_epoch_end(_ddp_trainer(rank, 1), _Host(setup.aggregator))
    except RuntimeError as e:
        return str(e)
    return None


class TestDistributedLinearProbe:
    @pytest.mark.parametrize("n_slides", [12, 1], ids=["all_slides", "fewer_slides_than_ranks"])
    def test_sharded_extraction_gathers_the_single_process_store(
        self, probe_setup, tmp_path, n_slides
    ):
        slide_ids = probe_setup.store.slide_ids[:n_slides]

        results = _run_ranks(tmp_path, _extract_rank, probe_setup, slide_ids)

        cb = LinearProbeEvalCallback([_probe_task(probe_setup)])
        reference = cb._extract_all_embeddings(
            SimpleNamespace(world_size=1), probe_setup.aggregator, CPU
        )
        ref_store = next(iter(reference.values()))
        ref_by_id = dict(zip(ref_store.slide_ids, ref_store.embeddings, strict=True))
        for ids, embeddings in results:
            assert sorted(ids) == sorted(slide_ids)
            for sid, emb in zip(ids, embeddings, strict=True):
                np.testing.assert_allclose(emb, ref_by_id[sid], rtol=1e-5, atol=1e-6)

    def test_every_rank_logs_the_single_process_composite(self, probe_setup, tmp_path):
        results = _run_ranks(tmp_path, _probe_rank, probe_setup, tmp_path / "ddp")

        host = _Host(probe_setup.aggregator)
        single = SimpleNamespace(**{**vars(_ddp_trainer(0, 2)), "world_size": 1})
        _composite_callback(probe_setup, tmp_path / "single").on_validation_epoch_end(single, host)

        assert set(host.logged) == {"lp/composite", "lp/cv/mean_auroc", "lp/cv/mean_balanced_acc"}
        for logged in results:
            assert logged == pytest.approx(host.logged)
        # Only rank 0 runs the sklearn probe and writes its results.
        assert (tmp_path / "ddp" / "rank0" / "epoch_002" / "cv" / "summary.json").is_file()
        assert not (tmp_path / "ddp" / "rank1").exists()

    def test_a_failing_task_raises_on_every_rank(self, probe_setup, tmp_path):
        results = _run_ranks(tmp_path, _failing_probe_rank, probe_setup, tmp_path / "out")

        assert results[0] == results[1]
        assert "LP callback failed for task='bad'" in results[0]
        assert "Need at least 7 unique groups" in results[0]
