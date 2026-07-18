"""Configuration boundary for OceanPath workflows.

Only this package and :mod:`oceanpath.workflows` translate Hydra objects into
typed domain inputs. Domain packages receive dataclasses and paths.
"""

from oceanpath.config.access import cfg_select
from oceanpath.config.paths import (
    FoundationPaths,
    artifact_config_fingerprint,
    training_config_fingerprint,
)

__all__ = [
    "FoundationPaths",
    "artifact_config_fingerprint",
    "cfg_select",
    "training_config_fingerprint",
]
