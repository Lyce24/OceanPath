"""Small, dependency-light helpers for reading composed configuration."""

from __future__ import annotations

from typing import Any


def cfg_select(cfg: Any, key: str, default: Any = None) -> Any:
    """Select a dotted key from a DictConfig or plain mapping."""
    try:
        from omegaconf import OmegaConf

        return OmegaConf.select(cfg, key, default=default)
    except Exception:
        current = cfg
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current


def cfg_list(cfg: Any, key: str, default: list[Any] | None = None) -> list[Any]:
    """Resolve a config list into an ordinary Python list."""
    value = cfg_select(cfg, key, default if default is not None else [])
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return list(value or [])
