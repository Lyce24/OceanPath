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

import logging

import lightning as L
import torch

logger = logging.getLogger(__name__)


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
            return

        if not self._embeddings:
            return

        # Stack and cap
        Z = torch.cat(self._embeddings, dim=0)[: self.max_samples]
        self._embeddings.clear()

        if Z.shape[0] < 2:
            return

        rankme, D = self._compute_rankme(Z)
        ratio = rankme / D

        pl_module.log("ssl/rankme", rankme, prog_bar=True, sync_dist=True)
        pl_module.log("ssl/rankme_ratio", ratio, sync_dist=True)
        logger.info(f"[RankMe] epoch={epoch}: rankme={rankme:.1f}, ratio={ratio:.3f} (D={D})")

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
            return

        if not self._embeddings:
            return

        Z = torch.cat(self._embeddings, dim=0)[: self.max_samples]
        self._embeddings.clear()

        if Z.shape[0] < 2:
            return

        alpha = self._compute_alpha_req(Z)

        pl_module.log("ssl/alpha_req", alpha, prog_bar=True, sync_dist=True)
        logger.info(f"[alpha-ReQ] epoch={epoch}: alpha={alpha:.3f}")

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
        self.rankme = RankMeCallback(
            compute_every_n_epochs=compute_every_n_epochs,
            max_samples=max_samples,
        )
        self.alpha_req = AlphaReQCallback(
            compute_every_n_epochs=compute_every_n_epochs,
            max_samples=max_samples,
        )

    def on_validation_batch_end(self, *args, **kwargs):
        self.rankme.on_validation_batch_end(*args, **kwargs)
        self.alpha_req.on_validation_batch_end(*args, **kwargs)

    def on_validation_epoch_end(self, *args, **kwargs):
        self.rankme.on_validation_epoch_end(*args, **kwargs)
        self.alpha_req.on_validation_epoch_end(*args, **kwargs)
