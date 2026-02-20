"""
Ensemble model wrapper for Stage 7 interpretability.

Stage 6 inference already averages softmax probabilities across fold models.
But Stage 7 needs more than predictions — it needs:
  - Averaged attention weights (for heatmaps)
  - Averaged slide embeddings (for UMAP)

This module provides EnsembleWrapper, which loads K fold checkpoints and
runs inference that returns all three (predictions, attention, embeddings)
with proper fold-level aggregation.

Design decision: WHY average attention across folds?
════════════════════════════════════════════════════════════════════════════
Each fold model sees slightly different training data, so attention patterns
differ. Averaging attention weights produces a more stable, representative
heatmap than any single fold — it highlights patches that ALL models agree
are important, filtering out fold-specific noise.

For UMAP: averaging embeddings produces a consensus representation that
reflects shared structure across folds rather than fold-specific artifacts.

Alternative considered: concatenate fold embeddings [K*D] for UMAP.
Rejected because it inflates dimensionality and mixes fold variance with
class variance, making UMAP harder to interpret.

Attention normalisation: all aggregators (ABMIL, TransMIL) store
PRE-softmax logits in extras["attention_weights"]. We apply softmax
per-fold to normalise the scale, then average. Averaging raw logits
would let the fold with the largest magnitude dominate.
"""

import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Output container
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class EnsembleResult:
    """Container for ensemble inference results on one slide."""

    probs: np.ndarray  # [C]   averaged softmax probs
    logits: np.ndarray  # [C]   averaged logits
    attention: np.ndarray | None  # [N]   averaged softmax attention (or None)
    embedding: np.ndarray  # [D]   averaged slide embedding
    fold_details: list[dict]  # per-fold raw outputs
    n_models: int

    @property
    def pred_class(self) -> int:
        return int(np.argmax(self.probs))

    @property
    def pred_prob(self) -> float:
        return float(self.probs[self.pred_class])


# ═════════════════════════════════════════════════════════════════════════════
# Wrapper
# ═════════════════════════════════════════════════════════════════════════════


class EnsembleWrapper:
    """
    Loads K fold checkpoints and runs inference with attention + embedding
    collection, then averages across folds.

    Works for all three model strategies:
        - ensemble:   loads fold_0.ckpt … fold_{K-1}.ckpt
        - best_fold:  loads model.ckpt (1 model)
        - refit:      loads model.ckpt (1 model)

    Usage
    ─────
        ew = EnsembleWrapper(model_dir, device="cuda:0")
        result = ew.predict_slide(features_tensor)
        # result.probs       [C]     averaged softmax probs
        # result.attention    [N]     averaged softmax attention
        # result.embedding    [D]     averaged slide embedding
    """

    def __init__(self, model_dir: Path | str, device: str = "cuda:0"):
        """
        Parameters
        ----------
        model_dir : Path
            Path to final model directory. For ensemble: contains fold_*.ckpt.
            For best_fold/refit: contains model.ckpt.
        device : str
            Device to run inference on.
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.models: list = []
        self.strategy = "unknown"

        # Read info.json for strategy metadata
        info_path = self.model_dir / "info.json"
        if info_path.is_file():
            info = json.loads(info_path.read_text())
            self.strategy = info.get("strategy", "unknown")

        self._load_models()

    # ── Loading ───────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        """Load all checkpoints from model_dir into memory."""
        if self.strategy == "ensemble":
            ckpt_paths = sorted(self.model_dir.glob("fold_*.ckpt"))
            if not ckpt_paths:
                raise FileNotFoundError(f"No fold_*.ckpt files in {self.model_dir}")
            for path in ckpt_paths:
                self.models.append(self._load_one(path))
            logger.info(
                f"EnsembleWrapper: loaded {len(self.models)} fold models from {self.model_dir}"
            )
        else:
            # best_fold or refit — single model
            ckpt_path = self.model_dir / "model.ckpt"
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            self.models.append(self._load_one(ckpt_path))
            logger.info(f"EnsembleWrapper: loaded single model ({self.strategy}) from {ckpt_path}")

    def _load_one(self, path: Path):
        """Load a single MILTrainModule checkpoint."""
        from oceanpath.modules.train_module import MILTrainModule

        try:
            module = MILTrainModule.load_from_checkpoint(
                str(path),
                weights_only=False,
                map_location=self.device,
            )
        except TypeError:
            # Older Lightning versions don't have weights_only
            module = MILTrainModule.load_from_checkpoint(
                str(path),
                map_location=self.device,
            )
        module.to(self.device)
        module.eval()
        return module

    @property
    def n_models(self) -> int:
        return len(self.models)

    # ── Per-slide inference ───────────────────────────────────────────────

    @torch.no_grad()
    def predict_slide(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> EnsembleResult:
        """
        Run all fold models on a single slide, average results.

        Parameters
        ----------
        features : [N, D] or [1, N, D] patch features.
        mask : optional [N] or [1, N] mask.
        coords : optional [N, 2] or [1, N, 2] coordinates.

        Returns
        -------
        EnsembleResult with averaged probs, attention, embedding.
        """
        # Ensure batch dimension
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if mask is not None and mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if coords is not None and coords.ndim == 2:
            coords = coords.unsqueeze(0)

        features = features.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        if coords is not None:
            coords = coords.to(self.device)

        all_probs = []
        all_logits = []
        all_attention = []
        all_embeddings = []
        fold_details = []

        for i, module in enumerate(self.models):
            # WSIClassifier.forward → MILOutput(slide_embedding, logits, extras)
            output = module.model(
                features,
                mask=mask,
                coords=coords,
                return_attention=True,
            )

            logits = output.logits.float()  # [1, C]
            probs = F.softmax(logits, dim=-1)  # [1, C]
            embedding = output.slide_embedding  # [1, D]

            all_probs.append(probs.cpu())
            all_logits.append(logits.cpu())
            all_embeddings.append(embedding.cpu())

            # Attention: extras["attention_weights"] is pre-softmax [1, N]
            # Normalise to softmax per fold before averaging
            attn_raw = output.extras.get("attention_weights")
            has_attn = attn_raw is not None
            if has_attn:
                attn_soft = F.softmax(attn_raw.float(), dim=-1)  # [1, N]
                all_attention.append(attn_soft.cpu())

            fold_details.append(
                {
                    "fold": i,
                    "probs": probs.squeeze(0).cpu().numpy(),
                    "logits": logits.squeeze(0).cpu().numpy(),
                    "has_attention": has_attn,
                }
            )

        # Average across folds
        avg_probs = torch.stack(all_probs).mean(dim=0).squeeze(0)  # [C]
        avg_logits = torch.stack(all_logits).mean(dim=0).squeeze(0)  # [C]
        avg_embedding = torch.stack(all_embeddings).mean(dim=0).squeeze(0)  # [D]

        avg_attention = None
        if all_attention:
            avg_attention = torch.stack(all_attention).mean(dim=0).squeeze(0)  # [N]

        return EnsembleResult(
            probs=avg_probs.numpy(),
            logits=avg_logits.numpy(),
            attention=avg_attention.numpy() if avg_attention is not None else None,
            embedding=avg_embedding.numpy(),
            fold_details=fold_details,
            n_models=len(self.models),
        )

    # ── Batch embedding computation ───────────────────────────────────────

    @torch.no_grad()
    def get_all_embeddings(
        self,
        dataset,
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        """
        Compute averaged slide embeddings for all slides in a dataset.

        Parameters
        ----------
        dataset : MmapDataset
            Dataset to iterate over.

        Returns
        -------
        embeddings : [N_slides, D]
        slide_ids : list[str]
        labels : [N_slides]
        """
        embeddings = []
        slide_ids = []
        labels = []

        for idx in range(len(dataset)):
            item = dataset[idx]
            features = item["features"].unsqueeze(0).to(self.device)

            # Average embeddings across fold models
            fold_embs = []
            for module in self.models:
                output = module.model(features, return_attention=False)
                fold_embs.append(output.slide_embedding.cpu())

            avg_emb = torch.stack(fold_embs).mean(dim=0).squeeze(0)
            embeddings.append(avg_emb.numpy())
            slide_ids.append(item["slide_id"])
            labels.append(item["label"])

        return np.stack(embeddings), slide_ids, np.array(labels)

    # ── Cleanup ───────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Free GPU memory."""
        for m in self.models:
            del m
        self.models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
