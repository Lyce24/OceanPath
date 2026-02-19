"""
WSI Classifier: MIL aggregator + classification head.

Wraps any BaseMIL aggregator and adds a linear classification head.
This is the model that gets instantiated for training.

    WSIClassifier = aggregator(patches → slide_embedding) + head(slide_embedding → logits)

Config-driven: changing `model=abmil` to `model=transmil` swaps the
aggregator without touching training code.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from oceanpath.models.base import BaseMIL, MILOutput

logger = logging.getLogger(__name__)


class WSIClassifier(nn.Module):
    """
    Full WSI classification model.

    Parameters
    ----------
    aggregator : BaseMIL
        The MIL aggregator (ABMIL, TransMIL, StaticMIL, etc.).
    num_classes : int
        Number of output classes. 1 = binary with sigmoid, 2+ = multiclass.
    freeze_aggregator : bool
        If True, freeze all aggregator parameters (e.g., for pretrained encoders).
    """

    def __init__(
        self,
        aggregator: BaseMIL,
        num_classes: int = 2,
        freeze_aggregator: bool = False,
    ):
        super().__init__()
        self.aggregator = aggregator
        self.num_classes = num_classes
        self.freeze_aggregator = freeze_aggregator

        # Classification head
        self.head = nn.Linear(aggregator.embed_dim, num_classes)

        # Freeze aggregator if requested
        if freeze_aggregator:
            for param in self.aggregator.parameters():
                param.requires_grad = False
            logger.info(
                f"Aggregator '{type(aggregator).__name__}' frozen "
                f"({sum(p.numel() for p in aggregator.parameters()):,} params)"
            )

        # Initialize head
        self.head.reset_parameters()

        # Log parameter counts
        agg_params = sum(p.numel() for p in self.aggregator.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"WSIClassifier: aggregator={type(aggregator).__name__} "
            f"({agg_params:,} params), head ({head_params:,} params), "
            f"trainable={trainable:,}"
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> MILOutput:
        """
        Full forward: aggregate → classify.

        Parameters
        ----------
        features : torch.Tensor
            [B, N, D] or [N, D] patch embeddings.
        mask, coords, return_attention : see BaseMIL.forward.

        Returns
        -------
        MILOutput
            slide_embedding: [B, E]
            logits: [B, C] (or [B] if num_classes=1)
            extras: dict (attention_weights, etc.)
        """
        output = self.aggregator(
            features, mask=mask, coords=coords, return_attention=return_attention
        )

        logits = self.head(output.slide_embedding)  # [B, C]

        if self.num_classes == 1:
            logits = logits.squeeze(-1)  # [B]

        return MILOutput(
            slide_embedding=output.slide_embedding,
            logits=logits,
            extras=output.extras,
        )

    def load_aggregator_weights(
        self,
        path: str,
        prefix: str = "",
        strict: bool = False,
    ) -> None:
        """
        Load pretrained weights into the aggregator.

        Parameters
        ----------
        path : str
            Path to checkpoint file.
        prefix : str
            Key prefix to strip (e.g., "encoder.aggregator.").
        strict : bool
            If True, raise on missing/unexpected keys.
        """
        state_dict = torch.load(path, map_location="cpu", weights_only=True)

        # Handle nested checkpoint formats
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # Strip prefix
        if prefix:
            state_dict = {
                k.replace(prefix, ""): v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }

        missing, unexpected = self.aggregator.load_state_dict(state_dict, strict=strict)
        if missing:
            logger.warning(f"Missing keys when loading aggregator: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading aggregator: {unexpected}")
        if not missing and not unexpected:
            logger.info("Aggregator weights loaded successfully.")

    def initialize_weights(self) -> None:
        """Reinitialize all weights (aggregator + head)."""
        if not self.freeze_aggregator:
            self.aggregator.initialize_weights()
        self.head.reset_parameters()