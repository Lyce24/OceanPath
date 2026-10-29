"""
GPU memory budget estimator.

Estimates peak GPU memory from the config BEFORE training starts.
Run during dry-run mode to catch OOM before Slurm queue wait.

Memory components for MIL training:
════════════════════════════════════
  1. Model parameters        — weights of aggregator + head
  2. Optimizer state          — AdamW stores 2 extra copies (momentum + variance)
  3. Gradients                — same size as parameters
  4. Activations (forward)    — [B, N, D] input bag + intermediate tensors
  5. Attention matrix          — [B, N, N] for TransMIL (quadratic!)
  6. Loss + backward buffer   — roughly same as forward activations

The critical variable is N (max_instances / bag size):
  - N=4000, D=1024: forward bag alone is 16 MB per sample
  - TransMIL attention at N=8000: N²x4 bytes = 256 MB per sample

Usage:
    from oceanpath.utils.memory import estimate_memory, print_memory_budget

    budget = estimate_memory(cfg)
    print_memory_budget(budget)
    if not budget.fits_gpu:
        raise RuntimeError(f"Estimated {budget.total_mb:.0f} MB > GPU capacity")
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Data structures
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class MemoryBudget:
    """Itemized GPU memory estimate."""

    # Component breakdown (MB)
    model_params_mb: float = 0.0
    optimizer_state_mb: float = 0.0
    gradient_mb: float = 0.0
    forward_activation_mb: float = 0.0
    attention_mb: float = 0.0
    backward_buffer_mb: float = 0.0
    cuda_overhead_mb: float = 300.0  # CUDA context + cuDNN workspace

    # Summary
    total_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    fits_gpu: bool = True
    headroom_pct: float = 0.0

    # Config context
    arch: str = ""
    max_instances: int = 0
    feature_dim: int = 0
    embed_dim: int = 0
    precision: str = "32"
    batch_size: int = 1

    # Warnings
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# ═════════════════════════════════════════════════════════════════════════════
# Estimator
# ═════════════════════════════════════════════════════════════════════════════


def estimate_memory(cfg, gpu_memory_mb: float | None = None) -> MemoryBudget:
    """
    Estimate peak GPU memory from config.

    Parameters
    ----------
    cfg : DictConfig
        Full resolved Hydra config with model, training, encoder, platform.
    gpu_memory_mb : float, optional
        Override GPU memory. Auto-detected if None.

    Returns
    -------
    MemoryBudget with itemized estimates and fits_gpu flag.
    """
    budget = MemoryBudget()

    # ── Extract config values ─────────────────────────────────────────────
    arch = _get(cfg, "model.arch", "abmil")
    in_dim = _get(cfg, "encoder.feature_dim", 1024)
    embed_dim = _get(cfg, "model.embed_dim", 512)
    max_inst = _get(cfg, "training.max_instances") or 8000
    batch_size = _get(cfg, "training.batch_size", 1)
    precision = str(_get(cfg, "platform.precision", "32"))
    num_fc_layers = _get(cfg, "model.num_fc_layers", 1)
    num_attn_layers = _get(cfg, "model.num_attention_layers", 2)
    num_heads = _get(cfg, "model.num_heads", 8)
    gradient_checkpointing = _get(cfg, "model.gradient_checkpointing", False)

    budget.arch = arch
    budget.max_instances = max_inst
    budget.feature_dim = in_dim
    budget.embed_dim = embed_dim
    budget.precision = precision
    budget.batch_size = batch_size

    # Bytes per parameter (training always uses float32 for master weights)
    bytes_per_param = 4  # float32 for optimizer state

    # Bytes per activation element (depends on AMP)
    if "16" in precision:
        bytes_per_activation = 2  # float16/bfloat16
    else:
        bytes_per_activation = 4  # float32

    # ── 1. Model parameters ───────────────────────────────────────────────
    n_params = _estimate_param_count(
        arch,
        in_dim,
        embed_dim,
        num_fc_layers,
        num_attn_layers,
        num_heads,
    )
    budget.model_params_mb = n_params * bytes_per_param / 1e6

    # ── 2. Optimizer state (AdamW: 2 extra copies) ────────────────────────
    # momentum (float32) + variance (float32) = 2x params
    budget.optimizer_state_mb = n_params * bytes_per_param * 2 / 1e6

    # ── 3. Gradients ──────────────────────────────────────────────────────
    budget.gradient_mb = n_params * bytes_per_param / 1e6

    # ── 4. Forward activations ────────────────────────────────────────────
    # Input bag: [B, N, D]
    input_bag_mb = batch_size * max_inst * in_dim * bytes_per_activation / 1e6

    # After patch embedding: [B, N, E]
    embed_bag_mb = batch_size * max_inst * embed_dim * bytes_per_activation / 1e6

    # MLP intermediates (rough: 2x embed bag for residuals)
    mlp_mb = embed_bag_mb * num_fc_layers

    budget.forward_activation_mb = input_bag_mb + embed_bag_mb + mlp_mb

    # ── 5. Attention matrix ───────────────────────────────────────────────
    if arch == "transmil":
        # TransMIL: N² attention per layer per head
        # NystromAttention approximates this but still allocates landmarks
        # Actual: ~O(N x num_landmarks) but we estimate worst-case full N²
        attn_per_layer = batch_size * num_heads * max_inst * max_inst * bytes_per_activation / 1e6
        budget.attention_mb = attn_per_layer * num_attn_layers

        if max_inst > 4000:
            budget.warnings.append(
                f"TransMIL with N={max_inst}: attention matrix is "
                f"{budget.attention_mb:.0f} MB. Consider gradient_checkpointing "
                f"or reducing max_instances."
            )
    elif arch == "abmil":
        # ABMIL: attention is [B, N, 1] — negligible
        budget.attention_mb = batch_size * max_inst * bytes_per_activation / 1e6
    else:
        # Static pooling: no attention
        budget.attention_mb = 0.0

    # ── 6. Backward buffer ────────────────────────────────────────────────
    # Backward roughly doubles the activation memory (stores intermediates
    # for chain rule). Gradient checkpointing reduces this substantially.
    if gradient_checkpointing:
        # Recomputes activations during backward → ~sqrt(layers) memory
        backward_multiplier = 0.5
        budget.warnings.append("gradient_checkpointing enabled: reduced backward buffer estimate")
    else:
        backward_multiplier = 1.0

    budget.backward_buffer_mb = (
        budget.forward_activation_mb + budget.attention_mb
    ) * backward_multiplier

    # ── Total ─────────────────────────────────────────────────────────────
    budget.total_mb = (
        budget.model_params_mb
        + budget.optimizer_state_mb
        + budget.gradient_mb
        + budget.forward_activation_mb
        + budget.attention_mb
        + budget.backward_buffer_mb
        + budget.cuda_overhead_mb
    )

    # ── GPU capacity ──────────────────────────────────────────────────────
    if gpu_memory_mb is not None:
        budget.gpu_memory_mb = gpu_memory_mb
    else:
        budget.gpu_memory_mb = _detect_gpu_memory()

    if budget.gpu_memory_mb > 0:
        # Use 85% threshold to leave room for fragmentation
        safe_limit = budget.gpu_memory_mb * 0.85
        budget.fits_gpu = budget.total_mb < safe_limit
        budget.headroom_pct = (safe_limit - budget.total_mb) / safe_limit * 100

        if not budget.fits_gpu:
            overshoot = budget.total_mb - safe_limit
            budget.warnings.append(
                f"Estimated {budget.total_mb:.0f} MB exceeds 85% of "
                f"{budget.gpu_memory_mb:.0f} MB GPU by {overshoot:.0f} MB. "
                f"Reduce max_instances or enable gradient_checkpointing."
            )

    return budget


def _estimate_param_count(
    arch: str,
    in_dim: int,
    embed_dim: int,
    num_fc_layers: int,
    num_attn_layers: int = 2,
    num_heads: int = 8,
) -> int:
    """
    Estimate parameter count from architecture config.

    These are approximations — the actual count depends on implementation
    details (bias terms, layer norms, etc.) but should be within 10%.
    """
    params = 0

    # Patch embedding MLP: in_dim → embed_dim (with intermediates)
    if num_fc_layers == 1:
        params += in_dim * embed_dim + embed_dim  # Linear + bias
    else:
        params += in_dim * embed_dim + embed_dim  # first layer
        for _ in range(num_fc_layers - 2):
            params += embed_dim * embed_dim + embed_dim  # hidden
        params += embed_dim * embed_dim + embed_dim  # last (if >1 layers)

    if arch == "abmil":
        # Gated attention: 2 parallel projections + combination
        attn_dim = int(embed_dim * 0.75)  # typical ratio
        # V branch: embed_dim → attn_dim
        params += embed_dim * attn_dim + attn_dim
        # U branch (gate): embed_dim → attn_dim
        params += embed_dim * attn_dim + attn_dim
        # attention head: attn_dim → 1
        params += attn_dim + 1

    elif arch == "transmil":
        # Per attention layer: Q, K, V projections + output + LayerNorm
        per_layer = (
            3 * embed_dim * embed_dim  # Q, K, V
            + embed_dim * embed_dim  # output projection
            + 4 * embed_dim  # 2x LayerNorm (scale + bias each)
        )
        params += per_layer * num_attn_layers

        # PPEG: 3 depthwise conv layers
        # Conv2d(E, E, k, groups=E): E x k x k params per conv
        params += embed_dim * (7 * 7 + 5 * 5 + 3 * 3)

        # CLS token
        params += embed_dim

    elif arch == "static":
        # No attention parameters
        pass

    # Classification head: embed_dim → num_classes (added by WSIClassifier)
    # Not included here — this is aggregator-only estimate

    return params


def _detect_gpu_memory() -> float:
    """Detect GPU total memory in MB."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_mem / 1e6
    except Exception:
        pass
    return 0.0


def _get(cfg, dotted_key: str, default=None):
    """Safely get a nested config value using dot notation."""
    try:
        from omegaconf import OmegaConf

        val = OmegaConf.select(cfg, dotted_key, default=default)
        return val
    except Exception:
        pass

    # Fallback for plain dicts
    keys = dotted_key.split(".")
    current = cfg
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
        elif hasattr(current, k):
            current = getattr(current, k)
        else:
            return default
        if current is None:
            return default
    return current


# ═════════════════════════════════════════════════════════════════════════════
# Display
# ═════════════════════════════════════════════════════════════════════════════


def print_memory_budget(budget: MemoryBudget) -> None:
    """Print formatted memory budget to console."""
    print(f"\n{'=' * 60}")
    print("  GPU Memory Budget")
    print(f"{'=' * 60}")
    print(f"  Architecture:    {budget.arch}")
    print(f"  Feature dim:     {budget.feature_dim}")
    print(f"  Embed dim:       {budget.embed_dim}")
    print(f"  Max instances:   {budget.max_instances:,}")
    print(f"  Batch size:      {budget.batch_size}")
    print(f"  Precision:       {budget.precision}")

    print(f"\n  {'Component':<30s} {'MB':>8s}")
    print(f"  {'─' * 40}")
    print(f"  {'Model parameters':<30s} {budget.model_params_mb:>8.1f}")
    print(f"  {'Optimizer state (AdamW)':<30s} {budget.optimizer_state_mb:>8.1f}")
    print(f"  {'Gradients':<30s} {budget.gradient_mb:>8.1f}")
    print(f"  {'Forward activations':<30s} {budget.forward_activation_mb:>8.1f}")
    print(f"  {'Attention matrix':<30s} {budget.attention_mb:>8.1f}")
    print(f"  {'Backward buffer':<30s} {budget.backward_buffer_mb:>8.1f}")
    print(f"  {'CUDA overhead':<30s} {budget.cuda_overhead_mb:>8.1f}")
    print(f"  {'─' * 40}")
    print(f"  {'TOTAL':<30s} {budget.total_mb:>8.1f}")

    if budget.gpu_memory_mb > 0:
        status = "✓ FITS" if budget.fits_gpu else "✗ EXCEEDS"
        print(f"\n  GPU:  {budget.gpu_memory_mb:.0f} MB ({status})")
        print(f"  Headroom: {budget.headroom_pct:+.0f}%")

    for w in budget.warnings:
        print(f"\n  ⚠  {w}")

    print(f"{'=' * 60}\n")
