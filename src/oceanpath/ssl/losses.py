"""
SSL loss functions for slide-level pretraining.

Five methods, ordered by conceptual complexity:

1. **VICReg** (Bardes et al., 2022)
   Three terms: invariance (MSE between views), variance (hinge on std per dim),
   covariance (off-diagonal penalty). No negatives, no momentum, no stop-grad on
   one branch. Simplest to tune.

2. **SimCLR** (Chen et al., 2020)
   NT-Xent: normalized temperature-scaled cross-entropy. Treats other samples in
   the batch as negatives. Needs large batch sizes (≥256) for good negatives.

3. **BYOL** (Grill et al., 2020)
   Asymmetric: online branch has predictor, target branch is EMA-updated.
   No negatives needed. Loss = negative cosine similarity.

4. **DINO** (Caron et al., 2021; Oquab et al., 2024 for v2)
   Self-distillation: student/teacher with centering + sharpening. Teacher is
   EMA of student. Cross-entropy between teacher (sharp) and student (soft)
   distributions over a learned prototype space.

5. **JEPA** (Assran et al., 2023)
   Predict target representations from context. Unlike contrastive methods,
   JEPA operates in representation space (not pixel/feature space). The predictor
   maps context embeddings → target embeddings. Loss = smooth-L1 on the predicted
   vs actual target representations.

All losses operate on slide-level embeddings [B, D] produced by the aggregator,
then projected through SSL heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        B, D = z1.shape

        # ── Invariance: MSE between paired embeddings ────────────────────
        inv_loss = F.mse_loss(z1, z2)

        # ── Variance: hinge loss on per-dimension std ────────────────────
        # Encourage each dimension to have std ≥ var_target
        # Use correction=0 (population variance) — this is a regularizer,
        # not a statistical estimate, and avoids NaN when B=1.
        z1_std = torch.sqrt(z1.var(dim=0, correction=0) + self.eps)
        z2_std = torch.sqrt(z2.var(dim=0, correction=0) + self.eps)
        var_loss = F.relu(self.var_target - z1_std).mean() + F.relu(self.var_target - z2_std).mean()

        # ── Covariance: penalize off-diagonal of covariance matrix ──────
        z1_c = z1 - z1.mean(dim=0)
        z2_c = z2 - z2.mean(dim=0)
        cov1 = (z1_c.T @ z1_c) / max(B - 1, 1)  # (D, D)
        cov2 = (z2_c.T @ z2_c) / max(B - 1, 1)

        # Zero diagonal, sum squared off-diagonal / D
        cov1.fill_diagonal_(0)
        cov2.fill_diagonal_(0)
        cov_loss = (cov1.pow(2).sum() + cov2.pow(2).sum()) / D

        loss = self.inv_weight * inv_loss + self.var_weight * var_loss + self.cov_weight * cov_loss

        return {
            "loss": loss,
            "inv_loss": inv_loss.detach(),
            "var_loss": var_loss.detach(),
            "cov_loss": cov_loss.detach(),
        }


# ═════════════════════════════════════════════════════════════════════════════
# 2. SimCLR (NT-Xent)
# ═════════════════════════════════════════════════════════════════════════════


class SimCLRLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy (NT-Xent).

    For batch of B pairs, creates 2B-sample contrastive problem.
    Each sample's positive is its paired view; all other 2(B-1)
    samples are negatives.

    Parameters
    ----------
    temperature : float
        Softmax temperature (τ). Lower = harder negatives.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        z1, z2: [B, D] L2-normalized projected embeddings.

        Returns dict with 'loss'.
        """
        B = z1.shape[0]
        device = z1.device

        # L2 normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate: [2B, D]
        z = torch.cat([z1, z2], dim=0)

        # Full similarity matrix: [2B, 2B]
        sim = (z @ z.T) / self.temperature  # (2B, 2B)

        # Mask out self-similarity (diagonal)
        mask_self = torch.eye(2 * B, device=device, dtype=torch.bool)
        sim.masked_fill_(mask_self, float("-inf"))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat(
            [
                torch.arange(B, 2 * B, device=device),
                torch.arange(0, B, device=device),
            ]
        )  # (2B,)

        loss = F.cross_entropy(sim, labels)

        return {"loss": loss}


# ═════════════════════════════════════════════════════════════════════════════
# 3. BYOL
# ═════════════════════════════════════════════════════════════════════════════


class BYOLLoss(nn.Module):
    """
    Bootstrap Your Own Latent loss.

    Negative cosine similarity between online prediction and
    stop-gradient target projection. Symmetric: both views
    serve as online and target.

    No parameters — the asymmetry comes from the architecture
    (predictor on online branch, EMA on target branch).
    """

    def forward(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        z1_target: torch.Tensor,
        z2_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        p1, p2: [B, D] predictor outputs from online branch.
        z1_target, z2_target: [B, D] projector outputs from EMA target (detached).

        Symmetric loss: predict target-view2 from online-view1, and vice versa.
        """
        loss = (
            self._cosine_loss(p1, z2_target.detach()) + self._cosine_loss(p2, z1_target.detach())
        ) * 0.5

        return {"loss": loss}

    @staticmethod
    def _cosine_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Negative cosine similarity (minimizing = aligning)."""
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2.0 - 2.0 * (p * z).sum(dim=-1).mean()


# ═════════════════════════════════════════════════════════════════════════════
# 4. DINO
# ═════════════════════════════════════════════════════════════════════════════


class DINOLoss(nn.Module):
    """
    Self-distillation loss (DINO / DINOv2).

    Cross-entropy between sharpened teacher output and softened student
    output over a learned prototype space. Teacher is centered to prevent
    mode collapse.

    Parameters
    ----------
    out_dim : int
        Prototype dimension (number of prototypes / "classes").
    teacher_temp : float
        Temperature for teacher softmax (low = sharp).
    student_temp : float
        Temperature for student softmax (higher = softer).
    center_momentum : float
        EMA momentum for center update.
    """

    def __init__(
        self,
        out_dim: int = 4096,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        student_out: [B, out_dim] student logits (pre-softmax).
        teacher_out: [B, out_dim] teacher logits (pre-softmax, detached).

        Returns dict with 'loss'.
        """
        # Teacher: center + sharpen
        teacher_centered = teacher_out.detach() - self.center
        teacher_probs = F.softmax(teacher_centered / self.teacher_temp, dim=-1)

        # Student: soften
        student_log_probs = F.log_softmax(student_out / self.student_temp, dim=-1)

        # Cross-entropy: H(teacher, student)
        loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()

        # Update center only during training.
        # Keeping center fixed in eval makes validation deterministic.
        if self.training:
            self._update_center(teacher_out)

        return {"loss": loss}

    @torch.no_grad()
    def _update_center(self, teacher_out: torch.Tensor) -> None:
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        self.center = self.center_momentum * self.center + (1 - self.center_momentum) * batch_center


# ═════════════════════════════════════════════════════════════════════════════
# 5. JEPA (Joint Embedding Predictive Architecture)
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
