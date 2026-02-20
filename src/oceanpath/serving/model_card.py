"""
Model card generator for exported MIL models.

A model card is a structured JSON document that travels with the exported
artifact. It answers: *what was this model trained on, how well does it
perform, and can I trust it for my use case?*

Why model cards matter for pathology:
═════════════════════════════════════
Medical ML models are safety-critical. A model card captures:
  - Exact training config (reproducibility)
  - Validation metrics (expected performance)
  - Data provenance (cohort, encoder, patching)
  - Version hashes (detect config/data drift)

If the model card is missing or its hashes don't match, the model
should not be used in a clinical setting without re-validation.

Schema version:
  v1 — initial schema (2025)
  Increment when fields are added/removed. Old readers ignore new fields.
"""

import json
import hashlib
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


# ═════════════════════════════════════════════════════════════════════════════
# ModelCard
# ═════════════════════════════════════════════════════════════════════════════


class ModelCard:
    """
    Build and write a model_card.json for an exported artifact.

    Usage
    ─────
        card = ModelCard(cfg_dict, export_report)
        card.add_metrics(oof_metrics_dict)
        card.add_thresholds({"optimal_f1": 0.42, "high_sensitivity": 0.25})
        card.write("artifacts/exp_abc123/model_card.json")
    """

    def __init__(
        self,
        cfg: dict,
        export_report: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        cfg : resolved Hydra config as dict.
        export_report : output from Exporter.run() (optional).
        """
        self._cfg = cfg
        self._export = export_report or {}
        self._metrics: dict = {}
        self._thresholds: dict = {}
        self._custom: dict = {}

    # ── Builder methods ───────────────────────────────────────────────────

    def add_metrics(self, metrics: dict) -> "ModelCard":
        """Add training/validation metrics (e.g., from OOF evaluation)."""
        self._metrics.update(metrics)
        return self

    def add_thresholds(self, thresholds: dict) -> "ModelCard":
        """Add decision thresholds (e.g., optimal F1, high-sensitivity)."""
        self._thresholds.update(thresholds)
        return self

    def add_custom(self, key: str, value: Any) -> "ModelCard":
        """Add arbitrary metadata."""
        self._custom[key] = value
        return self

    # ── Build ─────────────────────────────────────────────────────────────

    def build(self) -> dict:
        """Build the complete model card dictionary."""
        cfg = self._cfg

        card = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_sha": _get_git_sha(),

            # ── Identity ──────────────────────────────────────────────
            "experiment_name": cfg.get("exp_name", "unknown"),
            "config_fingerprint": _config_fingerprint(cfg),

            # ── Architecture ──────────────────────────────────────────
            "model": {
                "architecture": _nested_get(cfg, "model.arch", "unknown"),
                "aggregator": _nested_get(cfg, "model.name", "unknown"),
                "embed_dim": _nested_get(cfg, "model.embed_dim"),
                "num_classes": _nested_get(cfg, "num_classes"),
                "in_dim": _nested_get(cfg, "encoder.feature_dim"),
                "total_params": self._export.get("model_params"),
            },

            # ── Encoder ───────────────────────────────────────────────
            "encoder": {
                "name": _nested_get(cfg, "encoder.name"),
                "feature_dim": _nested_get(cfg, "encoder.feature_dim"),
                "family": _nested_get(cfg, "encoder.family"),
                "source": _nested_get(cfg, "encoder.source"),
            },

            # ── Patching ──────────────────────────────────────────────
            "patching": {
                "patch_size": _nested_get(cfg, "extraction.patch_size"),
                "magnification": _nested_get(cfg, "extraction.mag"),
                "overlap": _nested_get(cfg, "extraction.overlap"),
            },

            # ── Data ──────────────────────────────────────────────────
            "data": {
                "dataset": _nested_get(cfg, "data.name"),
                "csv_path": _nested_get(cfg, "data.csv_path"),
                "label_columns": _nested_get(cfg, "data.label_columns"),
                "wsi_extensions": _nested_get(cfg, "data.wsi_extensions"),
            },

            # ── Training ──────────────────────────────────────────────
            "training": {
                "splits": _nested_get(cfg, "splits.name"),
                "n_folds": _nested_get(cfg, "splits.n_folds",
                           _nested_get(cfg, "splits.k")),
                "lr": _nested_get(cfg, "training.lr"),
                "weight_decay": _nested_get(cfg, "training.weight_decay"),
                "max_epochs": _nested_get(cfg, "training.max_epochs"),
                "loss_type": _nested_get(cfg, "training.loss_type"),
                "seed": _nested_get(cfg, "training.seed"),
            },

            # ── Export ────────────────────────────────────────────────
            "export": {
                "checkpoint_hash": self._export.get("checkpoint_hash"),
                "formats": list(self._export.get("exports", {}).keys()),
                "validation_passed": self._export.get("success"),
            },

            # ── Performance ───────────────────────────────────────────
            "metrics": self._metrics,

            # ── Thresholds ────────────────────────────────────────────
            "thresholds": self._thresholds,

            # ── Serving contract ──────────────────────────────────────
            "serving": {
                "input_schema": {
                    "features": {
                        "dtype": "float32",
                        "shape": ["batch", "n_patches", self._export.get(
                            "in_dim",
                            _nested_get(cfg, "encoder.feature_dim", "D"),
                        )],
                        "description": (
                            "Precomputed patch embeddings from the encoder. "
                            "Variable n_patches per slide."
                        ),
                    },
                },
                "output_schema": {
                    "logits": {"dtype": "float32", "shape": ["batch", "num_classes"]},
                    "probabilities": {"dtype": "float32", "shape": ["batch", "num_classes"]},
                    "embedding": {"dtype": "float32", "shape": ["batch", "embed_dim"]},
                },
                "contract": (
                    "Default: serves precomputed features → predictions. "
                    "Raw WSI serving is a separate heavier pipeline requiring "
                    "the encoder + TRIDENT feature extraction."
                ),
            },
        }

        if self._custom:
            card["custom"] = self._custom

        return card

    def write(self, path: str) -> Path:
        """Build the card and write to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        card = self.build()
        path.write_text(json.dumps(card, indent=2, default=str))
        logger.info(f"Model card written → {path}")
        return path


# ═════════════════════════════════════════════════════════════════════════════
# Metric loading helpers
# ═════════════════════════════════════════════════════════════════════════════


def load_oof_metrics(train_dir: str) -> dict:
    """
    Load OOF (out-of-fold) validation metrics from training outputs.

    Reads from: {train_dir}/oof_metrics.json (written by train.py Phase 1).
    Falls back to: {train_dir}/eval/evaluation_report.json (from Stage 6).
    """
    train_dir = Path(train_dir)

    # Try OOF metrics (from training)
    oof_path = train_dir / "oof_metrics.json"
    if oof_path.is_file():
        try:
            return json.loads(oof_path.read_text())
        except Exception as e:
            logger.warning(f"Cannot parse {oof_path}: {e}")

    # Fallback: evaluation report from Stage 6
    eval_path = train_dir / "eval" / "evaluation_report.json"
    if eval_path.is_file():
        try:
            full_report = json.loads(eval_path.read_text())
            return full_report.get("metrics", full_report)
        except Exception as e:
            logger.warning(f"Cannot parse {eval_path}: {e}")

    return {}


def load_threshold_info(eval_dir: str) -> dict:
    """
    Load threshold calibration from Stage 6.

    Reads from: {eval_dir}/threshold_analysis.json.
    """
    path = Path(eval_dir) / "threshold_analysis.json"
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except Exception as e:
            logger.warning(f"Cannot parse {path}: {e}")
    return {}


# ═════════════════════════════════════════════════════════════════════════════
# Utilities
# ═════════════════════════════════════════════════════════════════════════════


def _get_git_sha() -> Optional[str]:
    """Get current git commit SHA, or None if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _config_fingerprint(cfg: dict) -> str:
    """Short hash of the resolved config for artifact naming."""
    canonical = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def _nested_get(d: dict, dotted_key: str, default=None):
    """Get a value from a nested dict using dot notation."""
    keys = dotted_key.split(".")
    current = d
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
        else:
            return default
        if current is None:
            return default
    return current