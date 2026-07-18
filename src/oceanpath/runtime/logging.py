"""One logging configuration for all OceanPath workflows."""

from __future__ import annotations

import logging
from typing import Any

from oceanpath.config.access import cfg_select


def setup_logging(cfg: Any, *, stage: str) -> None:
    """Configure process logging with stable stage and dataset context."""
    level = logging.DEBUG if bool(cfg_select(cfg, "verbose", False)) else logging.INFO
    dataset = cfg_select(cfg, "data.name", "unknown")
    experiment = cfg_select(cfg, "exp_name", "none")
    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s | %(levelname)-7s | "
            f"stage={stage} | data={dataset} | exp={experiment} | %(message)s"
        ),
        force=True,
    )
