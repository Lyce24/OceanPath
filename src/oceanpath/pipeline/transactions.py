"""
Atomic stage outputs and stage transaction metadata.

This module has two responsibilities:
1. Atomic writes for stage outputs (commit-on-success, rollback-on-failure).
2. Per-stage transaction metadata (fingerprint + provenance) used by DAG
   freshness checks to determine whether existing outputs are still valid.
"""

from __future__ import annotations

import json
import logging
import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_TRANSACTION_FILE = ".oceanpath_stage_transaction.json"


def _transaction_sidecar_path(output_path: str | Path) -> Path:
    """
    Resolve transaction metadata path for an output artifact.

    Directories:
      {output_dir}/.oceanpath_stage_transaction.json
    Files:
      {output_file.parent}/.{output_file.name}.transaction.json
    """
    output_path = Path(output_path)
    if output_path.exists() and output_path.is_file():
        return output_path.parent / f".{output_path.name}.transaction.json"

    # If suffix is present and path doesn't exist yet, treat as file output.
    if output_path.suffix and not output_path.exists():
        return output_path.parent / f".{output_path.name}.transaction.json"

    return output_path / _TRANSACTION_FILE


def write_stage_transaction(
    output_path: str | Path,
    *,
    stage_name: str,
    stage_fingerprint: str,
    inputs: list[str] | None = None,
    config_keys: list[str] | None = None,
    extra: dict | None = None,
) -> Path:
    """Write per-stage transaction metadata next to the output artifact."""
    meta_path = _transaction_sidecar_path(output_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "stage_name": stage_name,
        "stage_fingerprint": stage_fingerprint,
        "inputs": inputs or [],
        "config_keys": config_keys or [],
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "extra": extra or {},
    }
    meta_path.write_text(json.dumps(payload, sort_keys=True, indent=2))
    return meta_path


def read_stage_transaction(output_path: str | Path) -> dict | None:
    """Read per-stage transaction metadata. Returns None if missing/invalid."""
    meta_path = _transaction_sidecar_path(output_path)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        logger.warning(f"Invalid transaction metadata: {meta_path}")
        return None


def transaction_matches(
    output_path: str | Path,
    *,
    stage_name: str,
    stage_fingerprint: str,
) -> bool:
    """Check whether output metadata matches expected stage + fingerprint."""
    meta = read_stage_transaction(output_path)
    if meta is None:
        return False
    return (
        meta.get("stage_name") == stage_name and meta.get("stage_fingerprint") == stage_fingerprint
    )


@contextmanager
def atomic_output(
    final_dir: str | Path,
    validator: Callable[[Path], None] | None = None,
    cleanup_stale: bool = True,
):
    """
    Context manager for atomic directory output.

    Stage writes to `{final_dir}.tmp`, validator runs on tmp, and on success
    the tmp directory is atomically moved to final_dir.
    """
    final_dir = Path(final_dir)
    tmp_dir = final_dir.with_suffix(".tmp")

    if cleanup_stale and tmp_dir.exists():
        logger.warning(f"Removing stale tmp dir from previous crash: {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"atomic_output: writing to {tmp_dir}")

    try:
        yield tmp_dir

        if validator is not None:
            logger.debug(f"atomic_output: validating {tmp_dir}")
            validator(tmp_dir)

        if final_dir.exists():
            backup = final_dir.with_suffix(".backup")
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            final_dir.rename(backup)
            try:
                tmp_dir.rename(final_dir)
                shutil.rmtree(backup, ignore_errors=True)
            except Exception:
                if backup.exists() and not final_dir.exists():
                    backup.rename(final_dir)
                raise
        else:
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            tmp_dir.rename(final_dir)

        logger.info(f"atomic_output: committed -> {final_dir}")

    except Exception:
        logger.warning(f"atomic_output: rolling back {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def validate_files_exist(*required_names: str) -> Callable[[Path], None]:
    """Factory: validator that checks required output files exist."""

    def _validate(tmp_dir: Path) -> None:
        missing = [name for name in required_names if not (tmp_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing required outputs: {missing} in {tmp_dir}")

    return _validate


def validate_parquet_not_empty(*parquet_names: str) -> Callable[[Path], None]:
    """Factory: validator that checks parquet files exist and have rows."""

    def _validate(tmp_dir: Path) -> None:
        import pandas as pd

        for name in parquet_names:
            path = tmp_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Missing: {path}")
            df = pd.read_parquet(str(path))
            if len(df) == 0:
                raise ValueError(f"Empty parquet file: {path}")

    return _validate


def validate_no_nans(
    *parquet_names: str, columns: list[str] | None = None
) -> Callable[[Path], None]:
    """Factory: validator that checks parquet columns contain no NaNs."""

    def _validate(tmp_dir: Path) -> None:
        import pandas as pd

        for name in parquet_names:
            path = tmp_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Missing: {path}")
            df = pd.read_parquet(str(path))
            check_cols = columns or [c for c in df.columns if c.startswith("prob_")]
            for col in check_cols:
                if col in df.columns and df[col].isna().any():
                    n_nan = int(df[col].isna().sum())
                    raise ValueError(
                        f"NaN detected in column {col} for {path} ({n_nan}/{len(df)} rows)"
                    )

    return _validate


def compose_validators(*validators: Callable[[Path], None]) -> Callable[[Path], None]:
    """Compose multiple validators into a single validator."""

    def _validate(tmp_dir: Path) -> None:
        for validator in validators:
            validator(tmp_dir)

    return _validate


__all__ = [
    "atomic_output",
    "compose_validators",
    "read_stage_transaction",
    "transaction_matches",
    "validate_files_exist",
    "validate_no_nans",
    "validate_parquet_not_empty",
    "write_stage_transaction",
]
