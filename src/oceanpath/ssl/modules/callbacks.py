"""
SSL quality-monitoring callbacks: RankMe and alpha-ReQ.

These replace held-out classification accuracy as the primary quality
signal during self-supervised pretraining. Both measure representation
quality WITHOUT labels.

RankMe (Garrido et al., 2023)
═════════════════════════════
Effective rank of the embedding matrix via Shannon entropy of
singular values. Higher = more dimensions are being used = less
collapse.

    RankMe(Z) = exp(-Σᵢ pᵢ log pᵢ)

where pᵢ = σᵢ / Σⱼ σⱼ (normalized singular values of Z).

- RankMe = 1 → complete collapse (all embeddings identical)
- RankMe = D → full rank (all dimensions equally used)
- Good SSL: RankMe / D > 0.5 (using >50% of available dimensions)

alpha-ReQ (Agrawal et al., 2022)
═════════════════════════════
Alpha-Requiem: power-law exponent of the eigenvalue spectrum.
Fits alpha such that λₖ ∝ k^(-alpha). Measures spectral decay:

- alpha ≈ 0  → flat spectrum (good: all dimensions informative)
- alpha >> 1 → steep decay (bad: few dominant dimensions, near-collapse)
- Good SSL: alpha ∈ [0.5, 2.0]

Usage: Both callbacks compute metrics on validation embeddings every
N epochs. They're logged to the training logger (W&B/TensorBoard)
and can be used for early stopping or hyperparameter selection.
"""

from __future__ import annotations

import gc
import logging
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import lightning as L
import numpy as np
import torch

if TYPE_CHECKING:
    import argparse

    import pandas as pd

logger = logging.getLogger(__name__)


def _release_memory(*objs) -> None:
    """Drop references, force GC, and return CUDA cache to the driver.

    Called at the end of any callback hook that holds large transient state
    (extracted embeddings, validation buffers, intermediate DataFrames).
    """
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _trainer_world_size(trainer: L.Trainer) -> int:
    """Return a concrete world size even when tests pass a loose mock trainer."""
    try:
        return int(getattr(trainer, "world_size", 1))
    except (TypeError, ValueError):
        return 1


def _gather_embeddings_ddp(
    Z: torch.Tensor,
    trainer: L.Trainer,
    device: torch.device,
) -> torch.Tensor:
    """Gather variable-length embedding tensors from all DDP ranks.

    Handles uneven DistributedSampler splits where different ranks hold
    different numbers of validation embeddings.  Uses padded ``all_gather``
    + trim so that every rank ends up with the *full* concatenated tensor.
    """
    import torch.distributed as dist

    world_size = _trainer_world_size(trainer)

    # Resolve embed_dim: take max across ranks (some may have 0 samples)
    local_dim = Z.shape[1] if Z.ndim == 2 and Z.shape[0] > 0 else 0
    dim_t = torch.tensor([local_dim], device=device, dtype=torch.long)
    dist.all_reduce(dim_t, op=dist.ReduceOp.MAX)
    embed_dim = int(dim_t.item())

    if embed_dim == 0:
        return Z  # No rank produced any embeddings

    # Ensure Z is 2-D with the agreed embed_dim
    if Z.ndim != 2 or Z.shape[1] != embed_dim:
        Z = torch.empty(0, embed_dim, dtype=torch.float32)

    # Exchange per-rank sample counts
    local_n = torch.tensor([Z.shape[0]], device=device, dtype=torch.long)
    all_n = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(all_n, local_n)

    max_n = int(max(n.item() for n in all_n))
    if max_n == 0:
        return Z

    # Pad, gather, trim
    padded = torch.zeros(max_n, embed_dim, device=device, dtype=torch.float32)
    if Z.shape[0] > 0:
        padded[: Z.shape[0]] = Z.to(device=device, dtype=torch.float32)

    all_padded = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(all_padded, padded)

    parts: list[torch.Tensor] = []
    for i, n_t in enumerate(all_n):
        n = int(n_t.item())
        if n > 0:
            parts.append(all_padded[i][:n].cpu())

    return torch.cat(parts, dim=0) if parts else Z


class RankMeCallback(L.Callback):
    """
    Compute RankMe (effective rank) on validation embeddings.

    Runs every `compute_every_n_epochs` epochs on the validation set.
    Logs 'ssl/rankme' and 'ssl/rankme_ratio' (= rankme / embed_dim).

    Parameters
    ----------
    compute_every_n_epochs : int
        Frequency of computation (expensive for large val sets).
    max_samples : int
        Cap on number of embeddings to collect (memory safety).
    eps : float
        Numerical stability for log computation.
    """

    def __init__(
        self,
        compute_every_n_epochs: int = 5,
        max_samples: int = 2048,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.compute_every_n_epochs = compute_every_n_epochs
        self.max_samples = max_samples
        self.eps = eps
        self._embeddings: list[torch.Tensor] = []

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        """Collect slide embeddings from validation batches."""
        epoch = trainer.current_epoch
        if epoch % self.compute_every_n_epochs != 0:
            return

        # The pretrain module stores embeddings in outputs
        if isinstance(outputs, dict) and "embeddings" in outputs:
            emb = outputs["embeddings"].detach().cpu()
            self._embeddings.append(emb)

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        epoch = trainer.current_epoch
        if epoch % self.compute_every_n_epochs != 0:
            self._embeddings.clear()  # defensive: drop any stray appends
            return

        try:
            # Cat local embeddings (may be empty on some DDP ranks)
            if self._embeddings:
                Z = torch.cat(self._embeddings, dim=0)
            else:
                Z = torch.empty(0, dtype=torch.float32)
            self._embeddings.clear()

            # Gather across DDP ranks so the metric is computed on the
            # full validation set, not a per-rank shard.
            if _trainer_world_size(trainer) > 1:
                Z = _gather_embeddings_ddp(Z, trainer, pl_module.device)

            Z = Z[: self.max_samples]

            # Compute on every rank (Z is identical post-gather, SVD on
            # ≤max_samples is cheap) so pl_module.log() registers the
            # metric in callback_metrics on every rank — required for
            # ModelCheckpoint/EarlyStopping to find it as a monitor.
            self._compute_and_log(Z, trainer, pl_module)
        finally:
            self._embeddings.clear()
            _release_memory()

    def _compute_and_log(
        self,
        Z: torch.Tensor,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Compute RankMe from a pre-built embedding tensor and log results."""
        if Z.shape[0] < 2:
            return

        rankme, D = self._compute_rankme(Z)
        ratio = rankme / D

        # NOTE: do NOT pass rank_zero_only=True. Lightning excludes such
        # metrics from being usable as a ModelCheckpoint/EarlyStopping
        # monitor. All ranks must log the same value here.
        pl_module.log("ssl/rankme", rankme, prog_bar=True)
        pl_module.log("ssl/rankme_ratio", ratio)
        if trainer.is_global_zero:
            logger.info(
                f"[RankMe] epoch={trainer.current_epoch}: rankme={rankme:.1f}, ratio={ratio:.3f} (D={D})"
            )

    def _compute_rankme(self, Z: torch.Tensor) -> tuple[float, int]:
        """
        Compute effective rank via Shannon entropy of singular values.

        Z: [N, D] embedding matrix.
        Returns (rankme, D).
        """
        # Center embeddings (improves numerical stability)
        Z = Z.float() - Z.float().mean(dim=0, keepdim=True)

        # SVD (only need singular values)
        try:
            S = torch.linalg.svdvals(Z)  # (min(N, D),)
        except RuntimeError:
            logger.warning("[RankMe] SVD failed, returning 0")
            return 0.0, Z.shape[1]

        # Normalize to probability distribution
        S = S[self.eps < S]
        if len(S) == 0:
            return 0.0, Z.shape[1]

        p = S / S.sum()

        # Shannon entropy → effective rank
        entropy = -(p * torch.log(p + self.eps)).sum()
        rankme = torch.exp(entropy).item()

        return rankme, Z.shape[1]


class AlphaReQCallback(L.Callback):
    """
    Compute alpha-ReQ (power-law eigenvalue exponent) on validation embeddings.

    Fits alpha such that the eigenvalue spectrum λ_k ∝ k^(-alpha) via
    log-log linear regression.

    Logs 'ssl/alpha_req'. Good range: alpha ∈ [0.5, 2.0].
    alpha → 0: flat spectrum (good diversity).
    alpha >> 2: steep decay (near collapse).

    Parameters
    ----------
    compute_every_n_epochs : int
        Frequency of computation.
    max_samples : int
        Cap on embeddings to collect.
    """

    def __init__(
        self,
        compute_every_n_epochs: int = 5,
        max_samples: int = 2048,
    ):
        super().__init__()
        self.compute_every_n_epochs = compute_every_n_epochs
        self.max_samples = max_samples
        self._embeddings: list[torch.Tensor] = []

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        epoch = trainer.current_epoch
        if epoch % self.compute_every_n_epochs != 0:
            return

        if isinstance(outputs, dict) and "embeddings" in outputs:
            emb = outputs["embeddings"].detach().cpu()
            self._embeddings.append(emb)

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        epoch = trainer.current_epoch
        if epoch % self.compute_every_n_epochs != 0:
            self._embeddings.clear()
            return

        try:
            if self._embeddings:
                Z = torch.cat(self._embeddings, dim=0)
            else:
                Z = torch.empty(0, dtype=torch.float32)
            self._embeddings.clear()

            if _trainer_world_size(trainer) > 1:
                Z = _gather_embeddings_ddp(Z, trainer, pl_module.device)

            Z = Z[: self.max_samples]

            # Compute on every rank — see RankMeCallback for rationale.
            self._compute_and_log(Z, trainer, pl_module)
        finally:
            self._embeddings.clear()
            _release_memory()

    def _compute_and_log(
        self,
        Z: torch.Tensor,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Compute alpha-ReQ from a pre-built embedding tensor and log results."""
        if Z.shape[0] < 2:
            return

        alpha = self._compute_alpha_req(Z)

        # See RankMeCallback for why rank_zero_only is not used here.
        pl_module.log("ssl/alpha_req", alpha, prog_bar=True)
        if trainer.is_global_zero:
            logger.info(f"[alpha-ReQ] epoch={trainer.current_epoch}: alpha={alpha:.3f}")

    def _compute_alpha_req(self, Z: torch.Tensor) -> float:
        """
        Compute alpha via log-log regression on eigenvalue spectrum.

        Z: [N, D] embedding matrix.
        Returns alpha (power-law exponent).
        """
        Z = Z.float() - Z.float().mean(dim=0, keepdim=True)
        N, D = Z.shape

        # Covariance matrix eigenvalues
        # Use (Z^T Z) / (N-1) for efficiency when N > D
        if N >= D:
            cov = (Z.T @ Z) / max(N - 1, 1)
        else:
            cov = (Z @ Z.T) / max(N - 1, 1)

        try:
            eigenvalues = torch.linalg.eigvalsh(cov)  # ascending order
        except RuntimeError:
            logger.warning("[alpha-ReQ] Eigendecomposition failed, returning 0")
            return 0.0

        # Sort descending, keep positive eigenvalues
        eigenvalues = eigenvalues.flip(0)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        if len(eigenvalues) < 3:
            return 0.0

        # Fit the upper half of the spectrum instead of the full tail.
        # The tiny-noise floor can otherwise flatten the log-log slope and
        # underestimate alpha for near-collapsed representations.
        n_fit = max(3, len(eigenvalues) // 2)
        eigenvalues = eigenvalues[:n_fit]

        # Log-log regression: log(λ_k) = -alpha * log(k) + c
        k = torch.arange(1, n_fit + 1, dtype=torch.float32)
        log_k = torch.log(k)
        log_lambda = torch.log(eigenvalues)

        # OLS: alpha = -cov(log_k, log_λ) / var(log_k)
        log_k_c = log_k - log_k.mean()
        log_lambda_c = log_lambda - log_lambda.mean()

        var_log_k = (log_k_c * log_k_c).sum()
        if var_log_k < 1e-10:
            return 0.0

        alpha = -(log_k_c * log_lambda_c).sum() / var_log_k

        return alpha.item()


class SSLQualityCallback(L.Callback):
    """
    Convenience wrapper that computes both RankMe and alpha-ReQ.

    Maintains a **single** embedding buffer instead of one per sub-callback,
    halving peak memory during validation collection.

    Note: this wrapper composes ``RankMeCallback`` / ``AlphaReQCallback`` and
    invokes their ``_compute_and_log`` directly from a shared buffer. Their
    own ``on_validation_*`` hooks are intentionally NOT used here — register
    the wrapper in the trainer, not the inner callbacks (otherwise embeddings
    are collected twice and metrics are computed redundantly).

    Parameters
    ----------
    compute_every_n_epochs : int
        Compute metrics every N epochs.
    max_samples : int
        Max embeddings to collect.
    """

    def __init__(
        self,
        compute_every_n_epochs: int = 5,
        max_samples: int = 2048,
    ):
        super().__init__()
        self.compute_every_n_epochs = compute_every_n_epochs
        self.max_samples = max_samples
        self._embeddings: list[torch.Tensor] = []
        self.rankme = RankMeCallback(
            compute_every_n_epochs=compute_every_n_epochs,
            max_samples=max_samples,
        )
        self.alpha_req = AlphaReQCallback(
            compute_every_n_epochs=compute_every_n_epochs,
            max_samples=max_samples,
        )

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        epoch = trainer.current_epoch
        if epoch % self.compute_every_n_epochs != 0:
            return
        if isinstance(outputs, dict) and "embeddings" in outputs:
            emb = outputs["embeddings"].detach().cpu()
            self._embeddings.append(emb)

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        epoch = trainer.current_epoch
        if epoch % self.compute_every_n_epochs != 0:
            self._embeddings.clear()
            return

        try:
            if self._embeddings:
                Z = torch.cat(self._embeddings, dim=0)
            else:
                Z = torch.empty(0, dtype=torch.float32)
            self._embeddings.clear()

            if _trainer_world_size(trainer) > 1:
                Z = _gather_embeddings_ddp(Z, trainer, pl_module.device)

            Z = Z[: self.max_samples]

            # Compute on every rank from the shared, gathered tensor — the
            # inner _compute_and_log calls pl_module.log() WITHOUT
            # rank_zero_only=True so the metrics land in callback_metrics
            # on every rank and ModelCheckpoint can monitor them.
            self.rankme._compute_and_log(Z, trainer, pl_module)
            self.alpha_req._compute_and_log(Z, trainer, pl_module)
        finally:
            self._embeddings.clear()
            _release_memory()


@dataclass(frozen=True)
class LinearProbeTask:
    name: str
    protocol: str  # grouped_cv | external_test | predefined_split

    mmap_dir: str
    manifest_csv: str

    label_column: str
    patient_column: str
    filename_column: str = "filename"
    slide_id_column: str | None = None
    split_column: str | None = None

    # External test cohort, used only for protocol=external_test
    test_mmap_dir: str | None = None
    test_manifest_csv: str | None = None
    test_label_column: str | None = None
    test_patient_column: str | None = None
    test_filename_column: str | None = None
    test_slide_id_column: str | None = None

    # Probe settings
    n_splits: int = 5
    inner_splits: int = 3
    primary_metric: str | None = None
    c_grid: tuple[float, ...] = (
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
    )
    class_weight: str = "balanced"
    max_iter: int = 5000
    multi_class_mode: str = "auto"

    # Predefined split settings
    train_split_names: tuple[str, ...] = ("train", "tr")
    eval_split_names: tuple[str, ...] = ("test", "te")

    # Extraction settings
    max_instances: int | None = None
    batch_size: int = 1
    num_workers: int = 4

    # Subsampling strategy when max_instances < bag size.
    # contiguous: deterministic first-N (matches pretraining contiguous pre-cap).
    # random: uniform without replacement, deterministic per slide via hashed seed.
    # spatial_stratified: bin tokens into a grid over the bag bbox and sample
    #   evenly across cells (best tissue coverage at eval time).
    sampling_mode: str = "contiguous"
    sampling_seed: int = 42

    # Robustness
    allow_missing_mmap: bool = False
    allow_missing_manifest: bool = False
    seed: int = 42

    # Slide vs patient evaluation. False = compute metrics per slide while
    # still grouping by patient for CV folds (no patient leakage). Required
    # for cohorts where sections of the same case can have different labels
    # (e.g. GEJ Barrett's per-section severity).
    aggregate_to_patient: bool = True

    # Allow-list of friendly metric names to forward to W&B for this task.
    # ``None`` (default) preserves the legacy behaviour of logging every numeric
    # field returned by the sklearn probe. Setting this prunes the W&B stream
    # to only the metrics listed here. Recognised names: 'auroc', 'auprc',
    # 'f1', 'acc', 'qwk'. Each name is resolved to the actual sklearn key(s)
    # by ``LinearProbeEvalCallback``: e.g. for grouped CV 'auroc' →
    # ``mean_auroc``; for external_test 'auroc' → ``auroc``; for multi-class
    # tasks 'auroc' also matches ``mean_auroc_macro_ovr`` / ``auroc_macro_ovr``.
    wandb_log_metrics: tuple[str, ...] | None = None


class LinearProbeEvalCallback(L.Callback):
    """
    Periodically evaluates the current SSL aggregator with downstream sklearn probes.

    This is a monitoring callback only:
      - no gradient
      - no SSL loss interaction
      - rank-0 only under DDP
      - logs metrics under lp/{task_name}/{metric}
    """

    # Friendly metric name -> sklearn-probe summary key candidates. A friendly
    # name is satisfied if any candidate is present in the task's summary.
    # Both grouped-CV (mean_*) and external-test (bare) variants are listed
    # so a single allow-list works for both protocols. Multi-class tasks get
    # `*_auroc_macro_ovr` matched by the same `auroc` friendly name.
    _FRIENDLY_METRIC_KEYS: ClassVar[dict[str, tuple[str, ...]]] = {
        "auroc": ("mean_auroc", "auroc", "mean_auroc_macro_ovr", "auroc_macro_ovr"),
        "auprc": ("mean_auprc", "auprc"),
        "f1": ("mean_macro_f1", "macro_f1"),
        "acc": ("mean_balanced_acc", "balanced_acc"),
        "qwk": ("mean_qwk", "qwk"),
    }

    @classmethod
    def _resolve_metric_keys(cls, friendly_names: tuple[str, ...]) -> set[str]:
        out: set[str] = set()
        for n in friendly_names:
            key = n.strip().lower()
            try:
                out.update(cls._FRIENDLY_METRIC_KEYS[key])
            except KeyError as e:
                raise ValueError(
                    f"Unknown LP metric name {n!r}; "
                    f"expected one of {sorted(cls._FRIENDLY_METRIC_KEYS)}"
                ) from e
        return out

    def __init__(
        self,
        tasks: list[LinearProbeTask],
        every_n_epochs: int = 5,
        output_dir: str | Path = "outputs/pretrain_lp_eval",
        run_at_fit_start: bool = False,
        run_at_epoch0: bool = False,
        fail_on_error: bool = True,
        composite_metric_name: str | None = None,
        composite_metric_sources: list[str] | None = None,
        composite_metric_weights: list[float] | None = None,
    ):
        super().__init__()

        if every_n_epochs <= 0:
            raise ValueError("every_n_epochs must be positive.")

        self.tasks = tasks
        self.every_n_epochs = int(every_n_epochs)
        self.output_dir = Path(output_dir)
        self.run_at_fit_start = bool(run_at_fit_start)
        self.run_at_epoch0 = bool(run_at_epoch0)
        self.fail_on_error = bool(fail_on_error)

        # Composite metric: weighted sum of named LP metrics. Logged via
        # pl_module.log() so it lands in trainer.callback_metrics on every
        # rank — required for ModelCheckpoint(monitor=...) to find it.
        # If composite_metric_weights is None, falls back to a uniform mean.
        self.composite_metric_name = composite_metric_name
        self.composite_metric_sources = list(composite_metric_sources or [])
        if self.composite_metric_name and not self.composite_metric_sources:
            raise ValueError("composite_metric_name set without composite_metric_sources.")

        if composite_metric_weights is None:
            n = len(self.composite_metric_sources)
            self.composite_metric_weights = [1.0 / n] * n if n > 0 else []
        else:
            weights = [float(w) for w in composite_metric_weights]
            if len(weights) != len(self.composite_metric_sources):
                raise ValueError(
                    f"composite_metric_weights has {len(weights)} entries "
                    f"but composite_metric_sources has "
                    f"{len(self.composite_metric_sources)}."
                )
            self.composite_metric_weights = weights

        self._fit_start_done = False

    # ---------------------------------------------------------------------
    # Scheduling
    # ---------------------------------------------------------------------

    def _should_run_validation_epoch(self, trainer: L.Trainer) -> bool:
        if trainer.sanity_checking:
            return False

        epoch = int(trainer.current_epoch)

        # Important:
        # epoch == 0 at validation_epoch_end is usually AFTER the first train epoch.
        # For true random-init evaluation, use run_at_fit_start=True.
        if epoch == 0:
            return self.run_at_epoch0

        # Correct schedule: 5, 10, 15, ...
        # Not 4, 9, 14, ...
        return epoch % self.every_n_epochs == 0

    def on_fit_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if not self.run_at_fit_start or self._fit_start_done:
            return

        self._fit_start_done = True
        # pl_module.log() is forbidden inside on_fit_start by Lightning;
        # the composite + sources still go to W&B via the direct
        # lg.log_metrics() path inside _run_all_tasks.
        self._run_all_tasks(
            trainer=trainer,
            pl_module=pl_module,
            epoch_tag="fit_start",
            allow_pl_module_log=False,
        )

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if not self._should_run_validation_epoch(trainer):
            return

        self._run_all_tasks(
            trainer=trainer,
            pl_module=pl_module,
            epoch_tag=f"epoch_{int(trainer.current_epoch):03d}",
        )

    # ---------------------------------------------------------------------
    # Main execution
    # ---------------------------------------------------------------------

    def _run_all_tasks(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        epoch_tag: str,
        allow_pl_module_log: bool = True,
    ) -> None:
        is_distributed = _trainer_world_size(trainer) > 1

        # Entry barrier: all ranks synchronize
        if is_distributed:
            trainer.strategy.barrier()

        was_training = pl_module.training
        aggregator = pl_module.aggregator
        aggregator.eval()

        output_root = self.output_dir / epoch_tag
        all_metrics: dict[str, float] = {}
        lp_error: str | None = None
        store_cache: dict[tuple, Any] | None = None
        summary: dict[str, Any] | None = None

        try:
            with torch.inference_mode():
                # Phase 1: Embedding extraction (ALL ranks for DDP, rank-0 for single-GPU).
                # Caches by (mmap_dir, max_instances) to avoid redundant extraction
                # when multiple tasks share the same mmap.
                store_cache = self._extract_all_embeddings(
                    trainer=trainer,
                    model=aggregator,
                    device=pl_module.device,
                )

                # Phase 2: Sklearn linear probing (rank-0 only).
                if trainer.is_global_zero:
                    output_root.mkdir(parents=True, exist_ok=True)
                    for task in self.tasks:
                        try:
                            logger.info(
                                "[LP callback] start | tag=%s | task=%s | protocol=%s",
                                epoch_tag,
                                task.name,
                                task.protocol,
                            )

                            summary = self._run_task_from_stores(
                                task=task,
                                store_cache=store_cache,
                                output_dir=output_root / task.name,
                            )

                            task_metrics = self._flatten_summary(task, summary)
                            all_metrics.update(task_metrics)

                            logger.info(
                                "[LP callback] done | tag=%s | task=%s | metrics=%s",
                                epoch_tag,
                                task.name,
                                task_metrics,
                            )

                        except Exception as e:
                            logger.exception(
                                "[LP callback] failed | tag=%s | task=%s",
                                epoch_tag,
                                task.name,
                            )

                            if self.fail_on_error:
                                lp_error = (
                                    f"LP callback failed for task='{task.name}' "
                                    f"at tag='{epoch_tag}': {type(e).__name__}: {e}"
                                )
                                break
                        finally:
                            # Release per-task intermediates before next task
                            summary = None
                            _release_memory()

        finally:
            if was_training:
                pl_module.train()
            # Drop every cached EmbeddingStore (slide_ids list + np.float32
            # [N, D] array) and any task summaries before returning to
            # training, then reclaim CUDA cache filled by the aggregator
            # forward passes during extraction.
            if store_cache is not None:
                for k in list(store_cache.keys()):
                    store_cache[k] = None
                store_cache.clear()
            store_cache = None
            summary = None
            _release_memory()

        # ── DDP: broadcast all_metrics/error from rank 0 to every rank ───
        # Sklearn probing only ran on rank 0, so all_metrics is currently
        # populated only there. ModelCheckpoint runs on every rank and
        # reads from callback_metrics — every rank must therefore see the
        # composite metric. Broadcast the whole dict before computing the
        # composite so the rest of this method runs identically everywhere.
        #
        # Use torch.distributed.broadcast_object_list directly: Lightning's
        # strategy.broadcast returns the broadcasted object and should not be
        # treated as in-place mutation of a Python list. Missing this assignment
        # can make rank 0 see lp/composite_score while rank 1 does not, which
        # shifts checkpoint collectives and can deadlock NCCL.
        if is_distributed:
            import torch.distributed as dist

            payload = (
                [{"metrics": all_metrics, "error": lp_error}] if trainer.is_global_zero else [None]
            )
            dist.broadcast_object_list(payload, src=0)
            shared = payload[0] if payload[0] is not None else {}
            all_metrics = dict(shared.get("metrics") or {})
            lp_error = shared.get("error")

        if lp_error is not None and self.fail_on_error:
            # All ranks raise the same exception after the broadcast, so no
            # rank is left waiting in a different collective.
            if is_distributed:
                trainer.strategy.barrier()
            raise RuntimeError(lp_error)

        # Compute composite metric (weighted sum of named source metrics)
        # so that ModelCheckpoint can monitor a single number that combines
        # all downstream LP tasks. We only set a value when every source
        # metric is present — otherwise we'd write a partial value that
        # ModelCheckpoint would silently treat as "improving".
        composite_value: float | None = None
        if self.composite_metric_name and self.composite_metric_sources:
            missing = [k for k in self.composite_metric_sources if k not in all_metrics]
            if missing:
                if trainer.is_global_zero:
                    logger.warning(
                        "[LP callback] composite '%s' skipped — missing sources: %s",
                        self.composite_metric_name,
                        missing,
                    )
            else:
                vals = [float(all_metrics[k]) for k in self.composite_metric_sources]
                composite_value = float(
                    sum(w * v for w, v in zip(self.composite_metric_weights, vals))
                )
                all_metrics[self.composite_metric_name] = composite_value
                if trainer.is_global_zero:
                    breakdown = {
                        k: f"{w:.2f}*{float(all_metrics[k]):.4f}"
                        for k, w in zip(
                            self.composite_metric_sources,
                            self.composite_metric_weights,
                        )
                    }
                    logger.info(
                        "[LP callback] %s = %.4f from %s",
                        self.composite_metric_name,
                        composite_value,
                        breakdown,
                    )

        # Log to experiment tracker(s) directly (rank-0 only) — keeps the
        # raw all_metrics dict (incl. *_failed flags) out of Lightning's
        # callback_metrics machinery.
        if trainer.is_global_zero and all_metrics and trainer.loggers:
            step = trainer.global_step
            for lg in trainer.loggers:
                lg.log_metrics(all_metrics, step=step)

        # Additionally, surface the composite metric and its sources via
        # pl_module.log() so ModelCheckpoint(monitor=composite_metric_name)
        # can find them in callback_metrics. ALL ranks must call this with
        # the same value (no rank_zero_only) — otherwise Lightning won't
        # register the key on non-zero ranks and ModelCheckpoint raises.
        # `allow_pl_module_log` is False when called from on_fit_start
        # (Lightning forbids self.log there); the composite still goes to
        # W&B via the lg.log_metrics() path above.
        if allow_pl_module_log and self.composite_metric_name and composite_value is not None:
            for k in self.composite_metric_sources:
                pl_module.log(k, float(all_metrics[k]), prog_bar=False)
            pl_module.log(
                self.composite_metric_name,
                float(composite_value),
                prog_bar=True,
            )

        # Exit barrier
        if is_distributed:
            trainer.strategy.barrier()

    # ---------------------------------------------------------------------
    # Distributed embedding extraction + caching
    # ---------------------------------------------------------------------

    def _extract_all_embeddings(
        self,
        trainer: L.Trainer,
        model: torch.nn.Module,
        device: torch.device,
    ) -> dict[tuple, Any]:
        """Extract embeddings for all unique (mmap_dir, max_instances) across tasks.

        For DDP (world_size > 1), slide IDs are sharded across ranks and
        results are gathered via ``all_gather_object`` — giving near-linear
        speedup on the extraction bottleneck.  Within a single invocation,
        tasks sharing the same ``mmap_dir`` reuse a cached ``EmbeddingStore``
        so the forward pass is never run twice on the same data.

        Returns a dict keyed by
        ``(mmap_dir, max_instances, sampling_mode, sampling_seed)`` →
        ``EmbeddingStore``. Two tasks reading the same mmap with different
        sampling settings get separate cache entries.
        """
        from oceanpath.modules.extract_slide_embeddings import (
            build_loader,
            extract_aggregator,
        )
        from oceanpath.modules.linear_probing_sklearn import (
            EmbeddingStore,
            load_manifest,
        )

        # ── Collect unique extraction groups ──────────────────────────────
        groups: dict[tuple, dict] = {}

        for task in self.tasks:
            # Primary mmap
            key = (task.mmap_dir, task.max_instances, task.sampling_mode, task.sampling_seed)
            if key not in groups:
                groups[key] = {
                    "slide_ids": set(),
                    "batch_size": task.batch_size,
                    "num_workers": task.num_workers,
                }
            manifest_df = load_manifest(
                csv_path=task.manifest_csv,
                slide_id_column=task.slide_id_column,
                filename_column=task.filename_column,
                label_column=task.label_column,
                patient_column=task.patient_column,
                split_column=(task.split_column if task.protocol == "predefined_split" else None),
            )
            groups[key]["slide_ids"].update(manifest_df["slide_id"].tolist())

            # Test mmap for external_test protocol
            if task.protocol == "external_test" and task.test_mmap_dir and task.test_manifest_csv:
                test_key = (
                    task.test_mmap_dir,
                    task.max_instances,
                    task.sampling_mode,
                    task.sampling_seed,
                )
                if test_key not in groups:
                    groups[test_key] = {
                        "slide_ids": set(),
                        "batch_size": task.batch_size,
                        "num_workers": task.num_workers,
                    }
                test_manifest_df = load_manifest(
                    csv_path=task.test_manifest_csv,
                    slide_id_column=(task.test_slide_id_column or task.slide_id_column),
                    filename_column=(task.test_filename_column or task.filename_column),
                    label_column=(task.test_label_column or task.label_column),
                    patient_column=(task.test_patient_column or task.patient_column),
                    split_column=None,
                )
                groups[test_key]["slide_ids"].update(test_manifest_df["slide_id"].tolist())

        # ── Extract per group ─────────────────────────────────────────────
        store_cache: dict[tuple, EmbeddingStore] = {}
        for (mmap_dir, max_instances, sampling_mode, sampling_seed), info in groups.items():
            slide_ids = sorted(info["slide_ids"])
            batch_size = info["batch_size"]
            num_workers = info["num_workers"]

            logger.info(
                "[LP callback] extracting %d slides from %s "
                "(max_instances=%s, sampling_mode=%s, world_size=%d)",
                len(slide_ids),
                mmap_dir,
                max_instances,
                sampling_mode,
                _trainer_world_size(trainer),
            )

            if _trainer_world_size(trainer) > 1:
                store = self._extract_distributed(
                    slide_ids=slide_ids,
                    model=model,
                    device=device,
                    mmap_dir=mmap_dir,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    max_instances=max_instances,
                    sampling_mode=sampling_mode,
                    sampling_seed=sampling_seed,
                    trainer=trainer,
                )
            else:
                loader = build_loader(
                    mmap_dir=mmap_dir,
                    slide_ids=slide_ids,
                    mode="aggregator",
                    batch_size=batch_size,
                    num_workers=num_workers,
                    max_instances=max_instances,
                    sampling_mode=sampling_mode,
                    sampling_seed=sampling_seed,
                )
                embeddings, kept_ids = extract_aggregator(
                    loader=loader,
                    model=model,
                    device=device,
                )
                store = EmbeddingStore(
                    slide_ids=list(kept_ids),
                    embeddings=np.asarray(embeddings, dtype=np.float32),
                )

            store_cache[(mmap_dir, max_instances, sampling_mode, sampling_seed)] = store

        return store_cache

    def _extract_distributed(
        self,
        slide_ids: list[str],
        model: torch.nn.Module,
        device: torch.device,
        mmap_dir: str,
        batch_size: int,
        num_workers: int,
        max_instances: int | None,
        trainer: L.Trainer,
        sampling_mode: str = "contiguous",
        sampling_seed: int = 42,
    ) -> Any:
        """Shard ``slide_ids`` across DDP ranks, extract in parallel, gather."""
        import torch.distributed as dist

        from oceanpath.modules.extract_slide_embeddings import (
            build_loader,
            extract_aggregator,
        )
        from oceanpath.modules.linear_probing_sklearn import EmbeddingStore

        rank = trainer.global_rank
        world_size = trainer.world_size

        # DDP safety: verify all ranks agree on the slide_id list.
        # A mismatch here would cause each rank to shard differently
        # and the gathered result would be silently wrong.
        expected = torch.tensor(len(slide_ids), device=device, dtype=torch.long)
        actual_min = expected.clone()
        dist.all_reduce(actual_min, op=dist.ReduceOp.MIN)
        if int(actual_min.item()) != len(slide_ids):
            raise RuntimeError(
                f"DDP rank disagreement on slide_ids for {mmap_dir}: "
                f"rank {rank} has {len(slide_ids)}, "
                f"min across ranks is {int(actual_min.item())}"
            )

        my_slide_ids = slide_ids[rank::world_size]

        if my_slide_ids:
            loader = build_loader(
                mmap_dir=mmap_dir,
                slide_ids=my_slide_ids,
                mode="aggregator",
                batch_size=batch_size,
                num_workers=num_workers,
                max_instances=max_instances,
                sampling_mode=sampling_mode,
                sampling_seed=sampling_seed,
            )
            my_embeddings, my_kept_ids = extract_aggregator(
                loader=loader,
                model=model,
                device=device,
            )
            my_embeddings = np.asarray(my_embeddings, dtype=np.float32)
            my_kept_ids = list(my_kept_ids)
        else:
            # Edge case: fewer slides than ranks.  Use None as sentinel
            # so the gather loop can skip cleanly without shape issues.
            my_embeddings = None
            my_kept_ids = []

        # Gather across all ranks
        gathered: list = [None] * world_size
        dist.all_gather_object(gathered, (my_kept_ids, my_embeddings))

        # Free per-rank local copies — `gathered` now owns its own buffers.
        del my_embeddings, my_kept_ids

        all_slide_ids: list[str] = []
        all_embeddings: list[np.ndarray] = []
        for g_ids, g_emb in gathered:
            all_slide_ids.extend(g_ids)
            if g_emb is not None and g_emb.ndim == 2 and g_emb.shape[0] > 0:
                all_embeddings.append(g_emb)

        if not all_embeddings:
            del gathered
            _release_memory()
            raise RuntimeError(f"No embeddings extracted from any rank for mmap_dir={mmap_dir}")

        merged = np.concatenate(all_embeddings, axis=0)

        # Release the per-rank shards now that they're concatenated.
        all_embeddings.clear()
        del gathered
        _release_memory()

        return EmbeddingStore(
            slide_ids=all_slide_ids,
            embeddings=merged,
        )

    def _run_task_from_stores(
        self,
        task: LinearProbeTask,
        store_cache: dict[tuple, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Run sklearn probing for *task* using pre-extracted embedding stores."""
        output_dir.mkdir(parents=True, exist_ok=True)

        from oceanpath.modules.linear_probing_sklearn import (
            load_manifest,
            merge_manifest_and_embeddings,
            run_external_test,
            run_grouped_cv,
            run_predefined_split,
            save_summary,
        )
        from oceanpath.modules.linear_probing_sklearn import (
            setup_logging as setup_probe_logging,
        )

        setup_probe_logging(output_dir, verbose=False)

        try:
            # ── Build train DataFrame from cached store ───────────────────
            store = store_cache[
                (task.mmap_dir, task.max_instances, task.sampling_mode, task.sampling_seed)
            ]
            manifest_df = load_manifest(
                csv_path=task.manifest_csv,
                slide_id_column=task.slide_id_column,
                filename_column=task.filename_column,
                label_column=task.label_column,
                patient_column=task.patient_column,
                split_column=(task.split_column if task.protocol == "predefined_split" else None),
            )

            # The cached store may contain the *union* of slide IDs from
            # multiple tasks sharing the same mmap_dir.  Filter to the IDs
            # this task actually needs so that merge_manifest_and_embeddings
            # sees exactly the same set as the original per-task extraction.
            store = self._filter_store(store, manifest_df)

            train_df = merge_manifest_and_embeddings(
                manifest_df=manifest_df,
                store=store,
                allow_missing_mmap=task.allow_missing_mmap,
                allow_missing_manifest=task.allow_missing_manifest,
            ).reset_index(drop=True)

            logger.info(
                "[LP callback] task=%s | merged %d slides | %d patients | %d classes",
                task.name,
                len(train_df),
                train_df["patient_id"].nunique(),
                train_df["label"].nunique(),
            )

            args = self._build_probe_args(task, output_dir)

            if task.protocol == "grouped_cv":
                summary = run_grouped_cv(args, train_df)

            elif task.protocol == "predefined_split":
                summary = run_predefined_split(args, train_df)

            elif task.protocol == "external_test":
                if task.test_mmap_dir is None or task.test_manifest_csv is None:
                    raise ValueError(
                        f"Task '{task.name}' uses protocol=external_test, "
                        "but test_mmap_dir or test_manifest_csv is missing."
                    )

                test_store = store_cache[
                    (task.test_mmap_dir, task.max_instances, task.sampling_mode, task.sampling_seed)
                ]
                test_manifest_df = load_manifest(
                    csv_path=task.test_manifest_csv,
                    slide_id_column=(task.test_slide_id_column or task.slide_id_column),
                    filename_column=(task.test_filename_column or task.filename_column),
                    label_column=(task.test_label_column or task.label_column),
                    patient_column=(task.test_patient_column or task.patient_column),
                    split_column=None,
                )
                test_store = self._filter_store(test_store, test_manifest_df)

                test_df = merge_manifest_and_embeddings(
                    manifest_df=test_manifest_df,
                    store=test_store,
                    allow_missing_mmap=task.allow_missing_mmap,
                    allow_missing_manifest=task.allow_missing_manifest,
                ).reset_index(drop=True)

                summary = run_external_test(args, train_df, test_df)

            else:
                raise ValueError(
                    f"Unknown protocol='{task.protocol}' for task='{task.name}'. "
                    "Expected grouped_cv, predefined_split, or external_test."
                )

            save_summary(output_dir, summary)
            return summary

        finally:
            self._cleanup_probe_logging()
            # All per-task locals (manifest_df / train_df / test_df /
            # filtered EmbeddingStore) drop on function return; force GC so
            # the next task starts with a clean heap and the CUDA caching
            # allocator (filled if probing happened to touch GPU) is reset.
            _release_memory()

    @staticmethod
    def _filter_store(store: Any, manifest_df: pd.DataFrame) -> Any:
        """Filter an ``EmbeddingStore`` to slide IDs in *manifest_df*, preserving manifest order."""
        from oceanpath.modules.linear_probing_sklearn import EmbeddingStore

        needed = manifest_df["slide_id"].tolist()

        # Fast path: exact match in identical order
        if store.slide_ids == needed:
            return store

        # Dict lookup: O(1) per ID instead of O(N) set scan
        sid_to_idx = {sid: i for i, sid in enumerate(store.slide_ids)}
        indices = []
        kept_ids = []
        for sid in needed:
            idx = sid_to_idx.get(sid)
            if idx is not None:
                indices.append(idx)
                kept_ids.append(sid)

        if indices:
            return EmbeddingStore(
                slide_ids=kept_ids,
                embeddings=store.embeddings[indices],
            )
        embed_dim = store.embeddings.shape[1] if store.embeddings.ndim == 2 else 0
        return EmbeddingStore(
            slide_ids=[],
            embeddings=np.empty((0, embed_dim), dtype=np.float32),
        )

    def _build_probe_args(
        self,
        task: LinearProbeTask,
        output_dir: Path,
    ) -> argparse.Namespace:
        from oceanpath.modules.linear_probing_sklearn import make_probe_args

        # mmap_dir is None because the callback passes merged DataFrames
        # directly to the run_* functions (the loaders never touch mmap).
        # n_boot is forced to 0 here to keep the callback cheap; bootstrap
        # CIs are the standalone runner's job.
        # LP-during-pretraining always uses the FULL labeled set —
        # label-deficient evaluation is the standalone runner's job
        # (`scripts/linear_probing.py --label_fraction ...`). Composite
        # checkpoint selection is calibrated against full-label numbers.
        return make_probe_args(
            mmap_dir=None,
            manifest_csv=task.manifest_csv,
            output_dir=str(output_dir),
            test_mmap_dir=task.test_mmap_dir,
            test_manifest_csv=task.test_manifest_csv,
            test_slide_id_column=task.test_slide_id_column,
            test_filename_column=task.test_filename_column,
            test_label_column=task.test_label_column,
            test_patient_column=task.test_patient_column,
            slide_id_column=task.slide_id_column,
            filename_column=task.filename_column,
            label_column=task.label_column,
            patient_column=task.patient_column,
            split_column=task.split_column,
            protocol=task.protocol,
            n_splits=task.n_splits,
            inner_splits=task.inner_splits,
            primary_metric=task.primary_metric,
            c_grid=list(task.c_grid),
            class_weight=task.class_weight,
            max_iter=task.max_iter,
            multi_class_mode=task.multi_class_mode,
            non_singleton_policy="error",
            allow_missing_mmap=task.allow_missing_mmap,
            allow_missing_manifest=task.allow_missing_manifest,
            aggregate_to_patient=task.aggregate_to_patient,
            train_split_names=list(task.train_split_names),
            eval_split_names=list(task.eval_split_names),
            n_boot=0,
            seed=task.seed,
            save_model=False,
            verbose=False,
        )

    @classmethod
    def _flatten_summary(
        cls,
        task: LinearProbeTask,
        summary: dict[str, Any],
    ) -> dict[str, float]:
        """Convert the sklearn probe summary dict into a flat W&B metric dict.

        If the task declares ``wandb_log_metrics``, only the resolved keys
        are forwarded — every other numeric field is dropped before reaching
        W&B. The composite-source metrics (e.g. ``mean_auroc``) MUST appear
        in the allow-list of any task that contributes to the composite,
        otherwise ``LinearProbeEvalCallback`` will skip the composite
        because of a missing source.
        """
        if task.wandb_log_metrics is not None:
            allowed = cls._resolve_metric_keys(tuple(task.wandb_log_metrics))
        else:
            allowed = None

        metrics: dict[str, float] = {}
        for key, value in summary.items():
            if not isinstance(value, numbers.Number):
                continue
            if allowed is not None and key not in allowed:
                continue
            metrics[f"lp/{task.name}/{key}"] = float(value)
        return metrics

    @staticmethod
    def _cleanup_probe_logging() -> None:
        """
        setup_probe_logging attaches handlers to the named 'linear_probe' logger.
        Close/remove them after each task to avoid accumulating file handles.
        """
        probe_logger = logging.getLogger("linear_probe")

        for handler in probe_logger.handlers[:]:
            handler.close()
            probe_logger.removeHandler(handler)
