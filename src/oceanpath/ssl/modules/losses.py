"""
SSL loss functions for slide-level pretraining.

Four methods (the only ones in research scope):

1. **VICReg** (Bardes et al., 2022)
   Three terms: invariance (MSE between views), variance (hinge on std per dim),
   covariance (off-diagonal penalty). No negatives, no momentum, no stop-grad on
   one branch.

2. **JEPA** (Assran et al., 2023)
   Predict target representations from context. The predictor maps context
   embeddings → target embeddings. Loss = smooth-L1 on the predicted vs actual
   target representations. Used symmetrically (each view predicts the EMA-target
   of the other) at the slide level.

3. **LeJEPA** (Balestriero & LeCun, 2025)
   Latent-Euclidean JEPA. Combines a predictive loss (each view's embedding is
   pushed toward the centroid of the global views' embeddings) with SIGReg
   (Sketched Isotropic Gaussian Regularization), which pushes the embedding
   distribution to N(0, I) — provably optimal for downstream linear/k-NN/kernel
   probing. No EMA, no stop-gradient, no teacher-student. Single trade-off knob.

4. **LeJEPA-MC** (Multi-Crop LeJEPA for WSI MIL)
   Generalises LeJEPA to V_g global + V_l local views. Centroid is computed over
   global views only; every view (global and local) participates in the
   predictive term and SIGReg, matching DINO/LeJEPA multi-crop conventions.

All losses operate on slide-level embeddings [B, D] produced by the aggregator,
then projected through SSL heads.

The ``EMANetwork`` helper for the slide-level JEPA target branch lives in
``oceanpath.ssl.modules.heads`` (it is not a loss). Re-exported there for any consumer
that previously imported it via this module.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _ddp_active() -> bool:
    """True only when a process group is initialized with world_size > 1."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


# Numerical floor used in ratio diagnostics so we never divide by exactly zero
# (e.g. at init, or on the rare batch where pred_loss is numerically tiny).
_RATIO_EPS = 1e-12


def _z_norm_stats(z_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row L2 norms of every z, concatenated across views; returns mean, std."""
    norms = torch.cat([z.detach().float().norm(p=2, dim=1) for z in z_list])
    if norms.numel() <= 1:
        zero = norms.new_zeros(())
        return norms.mean() if norms.numel() else zero, zero
    return norms.mean(), norms.std(correction=0)


def _grad_norm_wrt_zs(
    scalar: torch.Tensor,
    z_list: list[torch.Tensor],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    """L2 norm of d(scalar)/d(z) summed over every z in z_list.

    All grads are computed in a single ``torch.autograd.grad`` call so each
    z gets a tensor of the same shape to take the squared sum of.
    """
    grads = torch.autograd.grad(
        scalar,
        z_list,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )
    sq = sum(g.detach().float().pow(2).sum() for g in grads)
    return sq.sqrt()


class _FullGatherLayer(torch.autograd.Function):
    """Autograd-aware all_gather along dim=0.

    Forward concatenates the tensor from every rank. Backward mirrors the
    official VICReg gather semantics: each rank gets its own gradient slice
    multiplied by ``world_size`` so Lightning/DDP's later gradient averaging
    yields the same gradient scale as a single-process global batch.

    The forward path supports uneven local batch sizes by gathering padded
    tensors and trimming them back to the original per-rank sizes. This keeps
    validation robust when the final validation batch is uneven, while still
    producing the exact global batch statistics needed during training.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = x.device

        local_n = int(x.shape[0])
        local_n_t = torch.tensor([local_n], device=device, dtype=torch.long)
        all_n_t = [torch.zeros_like(local_n_t) for _ in range(world_size)]
        dist.all_gather(all_n_t, local_n_t)
        sizes = [int(n.item()) for n in all_n_t]
        max_n = max(sizes)

        ctx.rank = rank
        ctx.world_size = world_size
        ctx.local_n = local_n
        ctx.sizes = sizes
        ctx.max_n = max_n

        if max_n == 0:
            return x

        padded = x.new_zeros((max_n, *x.shape[1:]))
        if local_n > 0:
            padded[:local_n] = x.contiguous()

        gathered = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded)
        return torch.cat([g[:n] for g, n in zip(gathered, sizes) if n > 0], dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        if ctx.max_n == 0:
            return grad_output

        padded_grads = grad_output.new_zeros((ctx.world_size, ctx.max_n, *grad_output.shape[1:]))
        offset = 0
        for i, n in enumerate(ctx.sizes):
            if n > 0:
                padded_grads[i, :n] = grad_output[offset : offset + n]
                offset += n

        # Every rank computed the same global-stat loss, so each rank has
        # the same per-chunk grad_output. Summing here gives world_size *
        # local_grad, compensating for DDP's subsequent gradient averaging.
        dist.all_reduce(padded_grads, op=dist.ReduceOp.SUM)
        return padded_grads[ctx.rank, : ctx.local_n].contiguous()


def _gather_for_stats(x: torch.Tensor) -> torch.Tensor:
    """All-gather x across ranks for batch-statistic computation.

    No-op outside DDP. Supports uneven local batch sizes, which can occur
    during validation; training normally remains even because the train loader
    drops the last partial batch.
    """
    if not _ddp_active():
        return x
    return _FullGatherLayer.apply(x)


# ═════════════════════════════════════════════════════════════════════════════
# 1. VICReg
# ═════════════════════════════════════════════════════════════════════════════


class VICRegLoss(nn.Module):
    """
    Variance-Invariance-Covariance Regularization.

    Parameters
    ----------
    inv_weight : float
        Weight for invariance (MSE) term. λ in the paper.
    var_weight : float
        Weight for variance (hinge) term. μ in the paper.
    cov_weight : float
        Weight for covariance (off-diagonal) term. nu in the paper.
    var_target : float
        Target std for the variance hinge (gamma in paper, default 1.0).
    eps : float
        Numerical stability for std computation.
    """

    def __init__(
        self,
        inv_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        var_target: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.var_target = var_target
        self.eps = eps

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        z1, z2: [B, D] projected embeddings from two views.

        Returns dict with 'loss', 'inv_loss', 'var_loss', 'cov_loss'.
        """
        z1 = z1.float()
        z2 = z2.float()

        # ── Gather across DDP ranks for batch-statistic terms ────────────
        # VICReg is a batch-statistic loss. Under DDP every term should see
        # the same global batch that a single-process run would see.
        # Outside DDP this is a no-op.
        z1_g = _gather_for_stats(z1)
        z2_g = _gather_for_stats(z2)
        B, D = z1_g.shape

        # ── Invariance: MSE between paired embeddings ────────────────────
        inv_loss = F.mse_loss(z1_g, z2_g)

        # ── Variance: hinge loss on per-dimension std ────────────────────
        # Encourage each dimension to have std ≥ var_target
        # Use correction=0 (population variance) — this is a regularizer,
        # not a statistical estimate, and avoids NaN when B=1.
        z1_std = torch.sqrt(z1_g.var(dim=0, correction=0) + self.eps)
        z2_std = torch.sqrt(z2_g.var(dim=0, correction=0) + self.eps)
        var_loss = F.relu(self.var_target - z1_std).mean() + F.relu(self.var_target - z2_std).mean()

        # ── Covariance: penalize off-diagonal of covariance matrix ──────
        z1_c = z1_g - z1_g.mean(dim=0)
        z2_c = z2_g - z2_g.mean(dim=0)
        cov1 = (z1_c.T @ z1_c) / max(B - 1, 1)  # (D, D)
        cov2 = (z2_c.T @ z2_c) / max(B - 1, 1)

        def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        # Zero diagonal, sum squared off-diagonal / D
        cov_loss = (off_diagonal(cov1).pow(2).sum() + off_diagonal(cov2).pow(2).sum()) / D

        loss = self.inv_weight * inv_loss + self.var_weight * var_loss + self.cov_weight * cov_loss

        return {
            "loss": loss,
            "inv_loss": inv_loss.detach(),
            "var_loss": var_loss.detach(),
            "cov_loss": cov_loss.detach(),
        }


# ═════════════════════════════════════════════════════════════════════════════
# 2. JEPA (Joint Embedding Predictive Architecture)
# ═════════════════════════════════════════════════════════════════════════════


class JEPALoss(nn.Module):
    """
    Joint Embedding Predictive Architecture loss.

    Predicts target representations from context representations.
    Unlike contrastive methods, JEPA doesn't push negatives apart —
    it only tries to predict target from context in representation space.

    Uses smooth-L1 (Huber) loss for robustness to outlier embeddings.

    Parameters
    ----------
    loss_type : str
        'smooth_l1' (default, robust) or 'mse' (simpler).
    beta : float
        Smooth-L1 transition point (only for smooth_l1).
    """

    def __init__(
        self,
        loss_type: str = "smooth_l1",
        beta: float = 2.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.beta = beta

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        predicted: [B, D] predictor output (from context view).
        target: [B, D] target encoder output (detached, from target view).

        Returns dict with 'loss'.
        """
        target = target.detach()

        if self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(predicted, target, beta=self.beta)
        elif self.loss_type == "mse":
            loss = F.mse_loss(predicted, target)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return {"loss": loss}


# ═════════════════════════════════════════════════════════════════════════════
# 3. LeJEPA  (Balestriero & LeCun, 2025)
# ═════════════════════════════════════════════════════════════════════════════


def sigreg(
    x: torch.Tensor,
    global_step: int,
    num_slices: int = 1024,
    integration_points: int = 17,
    integration_range: float = 5.0,
) -> torch.Tensor:
    """Sketched Isotropic Gaussian Regularization (Algorithm 1 of LeJEPA paper).

    Drives the empirical distribution of `x` toward an isotropic standard
    normal by matching the empirical characteristic function (ECF) of random
    1-D projections against the CF of N(0, 1), under a Gaussian weight
    w(t) = exp(-t²/2) (Epps-Pulley statistic).

    Properties (proved in the paper):
      * Linear time/memory in batch size N (vs quadratic for kernel-MMD).
      * Gradient and curvature uniformly bounded (Theorem 4) — stable
        regardless of input distribution.
      * Multivariate-equivalent test via the hyperspherical Cramér-Wold
        theorem: matching all 1-D projections matches the joint distribution.

    DDP semantics
    -------------
    The direction set A is sampled deterministically from `global_step`, so
    every rank uses the same projections in any given step. The ECF is built
    via a *weighted sum* across ranks (sum of `e^{i t x·a}` over all ranks,
    divided by the global N). This is correct for *uneven* per-rank batch
    sizes — using `ReduceOp.AVG` would weight every rank equally regardless
    of how many samples it contributes, which is wrong on the last
    validation batch when `drop_last=False`.

    Parameters
    ----------
    x : (N, D) tensor
        Embeddings on this rank.
    global_step : int
        Lightning's `trainer.global_step`. Same value across ranks.
    num_slices : int
        |A|. Authors recommend 256-4096; 1024 was the default in their
        ImageNet-1K experiments. SGD compounds direction coverage across
        steps so even small |A| converges (paper Figure 7).
    integration_points : int
        Trapezoidal quadrature knots over [-integration_range, integration_range].
        Authors found 17 sufficient.
    integration_range : float
        Half-width of the integration domain.

    Returns
    -------
    Scalar tensor: SIGReg loss (mean over slices of ``T(slice) * N_global``).
    """
    device = x.device
    # Force float32 internally for numerical stability of complex arithmetic.
    x = x.float()
    D = x.shape[1]

    # ── Direction sampling (synchronized across DDP ranks) ────────────────
    g = torch.Generator(device=device)
    g.manual_seed(int(global_step))
    A = torch.randn(D, num_slices, generator=g, device=device, dtype=torch.float32)
    # `clamp_min` guards against the (vanishingly rare) all-zero column.
    A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)

    # ── Integration grid + N(0,1) characteristic function ────────────────
    t = torch.linspace(
        -integration_range,
        integration_range,
        integration_points,
        device=device,
        dtype=torch.float32,
    )
    # CF of N(0,1) is exp(-t²/2) (real). Doubles as the Gaussian weight w(t).
    target_cf = torch.exp(-0.5 * t.pow(2))

    # ── Empirical CF: e^{i t (x · a)} summed (NOT averaged) over batch ───
    # We sum here so we can do a weighted average over ranks below by
    # dividing by the global N.
    x_proj = x @ A  # (N, M)
    x_t = x_proj.unsqueeze(2) * t  # (N, M, T)
    ecf_sum = (1j * x_t).exp().sum(dim=0)  # (M, T) complex64

    # Sample count on this rank, kept as a float tensor so we can all-reduce
    # it alongside the ECF without an extra Python sync.
    n_eff = torch.tensor(float(x.shape[0]), device=device, dtype=torch.float32)

    if _ddp_active():
        # PyTorch's complex all_reduce support is uneven across backends —
        # reduce real / imag separately to be portable.
        ecf_real = ecf_sum.real.contiguous()
        ecf_imag = ecf_sum.imag.contiguous()
        dist.all_reduce(ecf_real, op=dist.ReduceOp.SUM)
        dist.all_reduce(ecf_imag, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_eff, op=dist.ReduceOp.SUM)
        ecf_sum = torch.complex(ecf_real, ecf_imag)

    # Global mean ECF: sum(...) / N_global. clamp_min(1) guards against the
    # degenerate empty-batch case.
    ecf = ecf_sum / n_eff.clamp_min(1.0)

    # ── Weighted L2 distance between ECF and target, per (slice, t) ──────
    err = (ecf - target_cf).abs().pow(2) * target_cf  # (M, T) real

    # Trapezoidal integration over t for each slice, then N-scaled mean
    # over slices (paper Algorithm 1, Definition 2).
    T_per_slice = torch.trapezoid(err, t, dim=1)  # (M,)
    return (T_per_slice * n_eff).mean()


class LeJEPALoss(nn.Module):
    """LeJEPA loss: predictive term + SIGReg.

    The predictive term (Eq. 7 in the paper) pushes each view's embedding
    toward the centroid μ of the *global* views' embeddings. With two views
    treated as both global (V_g = V = 2, no local crops) — the natural setup
    for the existing two-view DataModule — μ = (z1 + z2) / 2 and the
    predictive loss reduces to a symmetric L2 around the midpoint.

    SIGReg pushes the embedding distribution to N(0, I) — the paper's
    Theorem 1 proves this is the unique distribution minimizing worst-case
    downstream prediction risk for linear, k-NN, and kernel probing.

    No EMA, no stop-gradient, no teacher-student, no predictor. The single
    knob is `sigreg_weight` (λ in the paper). Authors recommend λ = 0.05
    as a robust default across architectures and datasets.

    Parameters
    ----------
    sigreg_weight : float
        λ in Eq. (LeJEPA). 0.05 is the recommended starting point.
    num_slices : int
        |A| in SIGReg. Paper default for ImageNet-1K runs is 1024; smaller
        values (e.g. 256) work for smoke tests but should not be used for
        production runs.
    integration_points : int
        Trapezoidal knots in the Epps-Pulley integral.
    integration_range : float
        Integration half-width over t.
    """

    def __init__(
        self,
        sigreg_weight: float = 0.05,
        num_slices: int = 1024,
        integration_points: int = 17,
        integration_range: float = 5.0,
    ):
        super().__init__()
        self.sigreg_weight = float(sigreg_weight)
        self.num_slices = int(num_slices)
        self.integration_points = int(integration_points)
        self.integration_range = float(integration_range)

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        global_step: int = 0,
    ) -> dict[str, torch.Tensor]:
        """
        z1, z2: [B, D] embeddings (or projected embeddings) from the two views.
        global_step: integer used to deterministically seed direction sampling
                     so every DDP rank shares the same A.

        Returns dict with 'loss', 'pred_loss', 'sigreg_loss'.
        """
        z1 = z1.float()
        z2 = z2.float()

        # ── Predictive loss: predict the centroid of the global views ────
        # Match LeJEPA Algorithm 2: mean over views, batch, AND embedding dim
        # (i.e. plain `.mean()` on the [V, B, D] residual stack). Using
        # `.sum(dim=-1).mean()` would scale the term by D, making SIGReg
        # ineffective relative to it for any D > 1.
        views = torch.stack([z1, z2], dim=0)  # [V=2, B, D]
        center = views.mean(dim=0, keepdim=True)  # [1,   B, D]
        pred_loss = (views - center).pow(2).mean()

        # ── SIGReg on each view's embedding ──────────────────────────────
        # Use the same global_step for both views to mirror Algorithm 2 of
        # the paper, which calls SIGReg(emb, global_step) for every view.
        sig1 = sigreg(
            z1,
            global_step,
            num_slices=self.num_slices,
            integration_points=self.integration_points,
            integration_range=self.integration_range,
        )
        sig2 = sigreg(
            z2,
            global_step,
            num_slices=self.num_slices,
            integration_points=self.integration_points,
            integration_range=self.integration_range,
        )
        sigreg_loss = (sig1 + sig2) * 0.5

        # ── DDP gradient-scale compensation for SIGReg ────────────────────
        # SIGReg is a global-batch statistic: every rank computes the same
        # scalar (because the ECF was all-reduced). DDP then averages parameter
        # gradients across ranks, which divides SIGReg's contribution by
        # world_size. Pre-multiply by world_size in the loss path so the
        # post-DDP gradient matches a single-process run with the same total
        # batch. Logged value stays unscaled to keep dashboards interpretable.
        # VICReg solves the same issue with `_FullGatherLayer`; SIGReg cannot
        # use that gather (it averages the ECF, not raw embeddings).
        if _ddp_active() and self.training:
            sigreg_loss_scaled = sigreg_loss * float(dist.get_world_size())
        else:
            sigreg_loss_scaled = sigreg_loss

        loss = (1.0 - self.sigreg_weight) * pred_loss + self.sigreg_weight * sigreg_loss_scaled

        loss_dict = {
            "loss": loss,
            "pred_loss": pred_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
        }

        # Training-only diagnostics (logged under ssl/* by the LightningModule).
        # `_diagnostics` is consumed and removed from the dict in the SSL
        # pretrain module before the rest is logged under train/*.
        if self.training and z1.requires_grad:
            loss_dict["_diagnostics"] = self._lejepa_diagnostics(
                z_list=[z1, z2],
                pred_loss=pred_loss,
                sigreg_loss=sigreg_loss,
            )

        return loss_dict

    def _lejepa_diagnostics(
        self,
        z_list: list[torch.Tensor],
        pred_loss: torch.Tensor,
        sigreg_loss: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """LeJEPA per-step diagnostics computed without DDP scaling so the
        logged values are interpretable as the contribution of each term to
        the actual loss object (the world-size factor we apply to compensate
        for DDP gradient averaging is a back-pass-only correction).

        Grad norms are taken w.r.t. z so they are cheap (no full backward)
        and stable in scale across architectures. ``retain_graph=True`` is
        required because Lightning's optimizer.step() will trigger a real
        ``loss.backward()`` later in the same iteration.
        """
        lam = self.sigreg_weight

        z_norm_mean, z_norm_std = _z_norm_stats(z_list)

        weighted_sigreg = lam * sigreg_loss
        weighted_pred = (1.0 - lam) * pred_loss
        loss_ratio = weighted_sigreg / (weighted_pred.abs() + _RATIO_EPS)

        grad_pred_raw = _grad_norm_wrt_zs(pred_loss, z_list, retain_graph=True)
        grad_sig_raw = _grad_norm_wrt_zs(sigreg_loss, z_list, retain_graph=True)
        grad_norm_pred = (1.0 - lam) * grad_pred_raw
        grad_norm_sigreg = lam * grad_sig_raw
        grad_ratio = grad_norm_sigreg / (grad_norm_pred + _RATIO_EPS)

        return {
            "z_norm_mean": z_norm_mean,
            "z_norm_std": z_norm_std,
            "weighted_sigreg_loss": weighted_sigreg.detach(),
            "loss_ratio_sigreg_to_pred": loss_ratio.detach(),
            "grad_norm_pred": grad_norm_pred.detach(),
            "grad_norm_sigreg": grad_norm_sigreg.detach(),
            "grad_ratio_sigreg_to_pred": grad_ratio.detach(),
        }


# ═════════════════════════════════════════════════════════════════════════════
# 4. LeJEPA-MC  (Multi-Crop LeJEPA for WSI MIL)
# ═════════════════════════════════════════════════════════════════════════════


class LeJEPAMCLoss(nn.Module):
    """Multi-crop LeJEPA loss for WSI MIL.

    Adapts the original LeJEPA global/local-view formulation (Balestriero &
    LeCun, 2025, Section "Multi-crop") to feature-bag inputs. The structural
    contract is preserved exactly:

        μ      = (1/V_g) Σ_{v ∈ V_g} z_v        (centroid over GLOBAL views only)
        L_pred = (1/V)   Σ_{v ∈ V}   mean_{B,D} (z_v - μ)²    (every view predicts μ)
        L_sig  = (1/V)   Σ_{v ∈ V}   SIGReg(z_v)
        L      = (1 - λ) L_pred + λ L_sig

    where V = V_g + V_l. The squared residual is averaged over batch AND
    embedding dim (Algorithm 2 of the paper), not summed over D — summing
    would scale the predictive term by D and break the SIGReg trade-off. Local views participate in the predictive loss but
    do NOT contribute to the centroid — only global views define the target,
    matching DINO/LeJEPA conventions.

    Notes
    -----
    * Gradients flow through global views into the centroid, exactly as in
      the existing 2-crop ``LeJEPALoss``. There is no stop-gradient.
    * SIGReg is applied to every view independently; the same ``global_step``
      is passed to every call so all views share the same direction set A
      within a step (and across DDP ranks) — this is the canonical
      Algorithm 2 behaviour from the paper.
    * ``z_views`` may be a list or a stacked tensor of shape ``[V, B, D]``.
      Allowing both keeps the call site flexible without forcing a particular
      collate format on the data loader.

    Parameters
    ----------
    sigreg_weight : float
        λ in the LeJEPA loss. 0.05 is the paper's recommended default.
    num_slices : int
        |A| in SIGReg. Paper default for ImageNet-1K runs is 1024; smaller
        values (e.g. 256) work for smoke tests but should not be used for
        production runs.
    integration_points : int
        Trapezoidal knots in the Epps-Pulley integral.
    integration_range : float
        Integration half-width.
    """

    def __init__(
        self,
        sigreg_weight: float = 0.05,
        num_slices: int = 1024,
        integration_points: int = 17,
        integration_range: float = 5.0,
    ):
        super().__init__()
        if not (0.0 <= sigreg_weight <= 1.0):
            raise ValueError(f"sigreg_weight must be in [0, 1], got {sigreg_weight}")
        self.sigreg_weight = float(sigreg_weight)
        self.num_slices = int(num_slices)
        self.integration_points = int(integration_points)
        self.integration_range = float(integration_range)

    def forward(
        self,
        z_views,  # list[Tensor[B, D]] | Tensor[V, B, D]
        num_global: int,
        global_step: int = 0,
    ) -> dict:
        """
        z_views: list of V tensors of shape [B, D], OR a single tensor of
                 shape [V, B, D]. The first ``num_global`` entries are the
                 global views. Local views may use a different fixed_n at the
                 input stage but must produce the same projector output dim D.
        num_global: V_g.
        global_step: integer used to deterministically seed SIGReg's
                     direction sampling (must match across DDP ranks).
        """
        # Normalise inputs ------------------------------------------------
        if isinstance(z_views, torch.Tensor):
            if z_views.ndim != 3:
                raise ValueError(
                    f"z_views tensor must have shape [V, B, D], got {tuple(z_views.shape)}"
                )
            z_list = [z_views[v] for v in range(z_views.shape[0])]
        else:
            z_list = list(z_views)

        V = len(z_list)
        if V < 1:
            raise ValueError("z_views is empty")
        if not (1 <= num_global <= V):
            raise ValueError(
                f"num_global must satisfy 1 <= num_global <= V, got num_global={num_global}, V={V}"
            )
        # Cast all to float32 for numerical stability of statistics
        # (matches the original LeJEPALoss).
        z_list = [z.float() for z in z_list]

        # Predictive term -------------------------------------------------
        # μ over GLOBAL views (V_g). Stacking the full V into a single
        # tensor is fine because every projector output has the same D
        # (shape per view is [B, D] regardless of input N).
        z_stack = torch.stack(z_list, dim=0)  # [V, B, D]
        z_global = z_stack[:num_global]  # [V_g, B, D]
        center = z_global.mean(dim=0, keepdim=True)  # [1, B, D]

        # Mean over (B, D) per view -> [V]. Algorithm 2 averages over views,
        # batch, AND embedding dimension. Using `.sum(dim=-1).mean()` here
        # would scale by D and make SIGReg ineffective for any D > 1.
        per_view_pred = (z_stack - center).pow(2).mean(dim=(1, 2))  # [V]
        pred_loss = per_view_pred.mean()

        # SIGReg term -----------------------------------------------------
        per_view_sig = torch.stack(
            [
                sigreg(
                    z,
                    global_step,
                    num_slices=self.num_slices,
                    integration_points=self.integration_points,
                    integration_range=self.integration_range,
                )
                for z in z_list
            ],
            dim=0,
        )  # [V]
        sigreg_loss = per_view_sig.mean()

        # ── DDP gradient-scale compensation (see LeJEPALoss for rationale).
        if _ddp_active() and self.training:
            sigreg_loss_scaled = sigreg_loss * float(dist.get_world_size())
        else:
            sigreg_loss_scaled = sigreg_loss

        # Combine ---------------------------------------------------------
        loss = (1.0 - self.sigreg_weight) * pred_loss + self.sigreg_weight * sigreg_loss_scaled

        loss_dict = {
            "loss": loss,
            "pred_loss": pred_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
            # Per-view diagnostics (detached). Useful to verify local views
            # are not driving pred_loss to a degenerate solution.
            "pred_loss_global_mean": per_view_pred[:num_global].mean().detach(),
            "pred_loss_local_mean": (
                per_view_pred[num_global:].mean().detach()
                if num_global < V
                else torch.zeros((), device=loss.device)
            ),
            "num_global": num_global,
            "num_local": V - num_global,
        }

        if self.training and z_list[0].requires_grad:
            lam = self.sigreg_weight
            z_norm_mean, z_norm_std = _z_norm_stats(z_list)
            weighted_sigreg = lam * sigreg_loss
            weighted_pred = (1.0 - lam) * pred_loss
            loss_ratio = weighted_sigreg / (weighted_pred.abs() + _RATIO_EPS)

            grad_pred_raw = _grad_norm_wrt_zs(pred_loss, z_list, retain_graph=True)
            grad_sig_raw = _grad_norm_wrt_zs(sigreg_loss, z_list, retain_graph=True)
            grad_norm_pred = (1.0 - lam) * grad_pred_raw
            grad_norm_sigreg = lam * grad_sig_raw
            grad_ratio = grad_norm_sigreg / (grad_norm_pred + _RATIO_EPS)

            loss_dict["_diagnostics"] = {
                "z_norm_mean": z_norm_mean,
                "z_norm_std": z_norm_std,
                "weighted_sigreg_loss": weighted_sigreg.detach(),
                "loss_ratio_sigreg_to_pred": loss_ratio.detach(),
                "grad_norm_pred": grad_norm_pred.detach(),
                "grad_norm_sigreg": grad_norm_sigreg.detach(),
                "grad_ratio_sigreg_to_pred": grad_ratio.detach(),
            }

        return loss_dict
