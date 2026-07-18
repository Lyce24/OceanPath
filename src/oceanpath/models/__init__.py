"""
Model registry and factory.

    from oceanpath.models import build_classifier, build_aggregator

    model = build_aggregator("abmil", in_dim=1024, model_cfg=cfg.model)
    model = build_classifier("abmil", in_dim=1024, num_classes=2, model_cfg=cfg.model)
"""

from collections.abc import Callable
from typing import Any, TypeVar

from oceanpath.models.base import BaseMIL, MILOutput
from oceanpath.models.wsi_classifier import WSIClassifier

# ── Registry ──────────────────────────────────────────────────────────────────

_AGGREGATOR_REGISTRY: dict[str, type[BaseMIL]] = {}
_AggregatorT = TypeVar("_AggregatorT", bound=BaseMIL)


def register_aggregator(
    name: str,
) -> Callable[[type[_AggregatorT]], type[_AggregatorT]]:
    """Decorator to register a MIL aggregator class."""

    def decorator(cls: type[_AggregatorT]) -> type[_AggregatorT]:
        _AGGREGATOR_REGISTRY[name] = cls
        return cls

    return decorator


def _register_builtins() -> None:
    """Register built-in aggregators. Called on first import."""
    from oceanpath.models.abmil import ABMIL
    from oceanpath.models.mhabmil import MultiheadABMIL
    from oceanpath.models.static import StaticMIL
    from oceanpath.models.transmil import TransMIL

    _AGGREGATOR_REGISTRY["abmil"] = ABMIL
    _AGGREGATOR_REGISTRY["transmil"] = TransMIL
    _AGGREGATOR_REGISTRY["static"] = StaticMIL
    _AGGREGATOR_REGISTRY["mhabmil"] = MultiheadABMIL


_register_builtins()


def list_aggregators() -> list[str]:
    """Return names of all registered aggregators."""
    return sorted(_AGGREGATOR_REGISTRY.keys())


# ── Factory functions ─────────────────────────────────────────────────────────


def build_aggregator(
    arch: str,
    in_dim: int,
    model_cfg: Any = None,
    **kwargs: Any,
) -> BaseMIL:
    """
    Build a MIL aggregator from a config.

    Parameters
    ----------
    arch : str
        Architecture name ('abmil', 'transmil', 'static', or 'mhabmil').
    in_dim : int
        Input patch feature dimension (from encoder config).
    model_cfg : DictConfig or dict
        Model hyperparameters (embed_dim, dropout, etc.).
        Keys are passed as kwargs to the aggregator constructor.
    **kwargs
        Additional overrides.

    Returns
    -------
    BaseMIL
        Instantiated aggregator.
    """
    if arch not in _AGGREGATOR_REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. Available: {list_aggregators()}")

    cls = _AGGREGATOR_REGISTRY[arch]

    # Merge config dict with kwargs
    params: dict[str, Any] = {"in_dim": in_dim}
    if model_cfg is not None:
        cfg_dict = dict(model_cfg)

        import inspect

        valid_keys = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}

        for k, v in cfg_dict.items():
            if k in valid_keys:
                params[k] = v

    params.update(kwargs)

    return cls(**params)


def build_classifier(
    arch: str,
    in_dim: int,
    num_classes: int,
    model_cfg: Any = None,
    freeze_aggregator: bool = False,
    aggregator_weights_path: str | None = None,
    aggregator_weights_prefix: str = "",
    **kwargs: Any,
) -> WSIClassifier:
    """
    Build a full WSI classifier (aggregator + head) from config.

    Parameters
    ----------
    arch : str
        Aggregator architecture name.
    in_dim : int
        Input patch feature dimension.
    num_classes : int
        Number of output classes.
    model_cfg : DictConfig or dict
        Model hyperparameters.
    freeze_aggregator : bool
        Freeze aggregator weights.
    aggregator_weights_path : str, optional
        Path to pretrained aggregator weights.
    aggregator_weights_prefix : str
        Key prefix to strip from checkpoint.
    **kwargs
        Passed to aggregator constructor.

    Returns
    -------
    WSIClassifier
    """
    aggregator = build_aggregator(arch, in_dim, model_cfg, **kwargs)

    classifier = WSIClassifier(
        aggregator=aggregator,
        num_classes=num_classes,
        freeze_aggregator=freeze_aggregator,
    )

    if aggregator_weights_path:
        classifier.load_aggregator_weights(
            aggregator_weights_path,
            prefix=aggregator_weights_prefix,
        )

    return classifier


__all__ = [
    "BaseMIL",
    "MILOutput",
    "WSIClassifier",
    "build_aggregator",
    "build_classifier",
    "list_aggregators",
    "register_aggregator",
]
