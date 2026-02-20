"""
Atomic stage outputs — no partial results from crashed stages.

Every stage writes to a .tmp directory, validates outputs, then atomically
renames to the final location. If the stage crashes, only .tmp remains
(cleaned up on next run). This prevents downstream stages from reading
half-written parquet files or incomplete checkpoints.

Usage:
    from oceanpath.pipeline.transactions import atomic_output

    with atomic_output(final_dir, validator=validate_training_output) as tmp:
        # Write all outputs to tmp/
        save_model(tmp / "model.pt")
        save_predictions(tmp / "preds.parquet")
    # On exit: validator runs, tmp/ renamed to final_dir/

    # If an exception occurs inside the with block:
    # tmp/ is cleaned up, final_dir/ is untouched
"""

import logging
import shutil
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def atomic_output(
    final_dir: str | Path,
    validator: Callable[[Path], None] | None = None,
    cleanup_stale: bool = True,
):
    """
    Context manager for atomic stage output.

    Parameters
    ----------
    final_dir : Path
        Where validated outputs should end up.
    validator : callable(Path) → None
        Validation function called on tmp_dir before commit.
        Should raise on invalid outputs (missing files, NaN rows, etc.).
    cleanup_stale : bool
        If True, remove any leftover .tmp dir from a previous crash.

    Yields
    ------
    tmp_dir : Path
        Temporary directory to write into.

    Raises
    ------
    Whatever the validator raises (output is NOT committed).
    Whatever the stage code raises (output is NOT committed).
    """
    final_dir = Path(final_dir)
    tmp_dir = final_dir.with_suffix(".tmp")

    # Clean up stale tmp from previous crash
    if cleanup_stale and tmp_dir.exists():
        logger.warning(f"Removing stale tmp dir from previous crash: {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"atomic_output: writing to {tmp_dir}")

    try:
        yield tmp_dir

        # ── Validate before commit ────────────────────────────────────
        if validator is not None:
            logger.debug(f"atomic_output: validating {tmp_dir}")
            validator(tmp_dir)

        # ── Commit: atomic rename ─────────────────────────────────────
        if final_dir.exists():
            # Backup existing output (in case rename fails partway)
            backup = final_dir.with_suffix(".backup")
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            final_dir.rename(backup)
            try:
                tmp_dir.rename(final_dir)
                shutil.rmtree(backup, ignore_errors=True)
            except Exception:
                # Restore backup if rename failed
                if backup.exists() and not final_dir.exists():
                    backup.rename(final_dir)
                raise
        else:
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            tmp_dir.rename(final_dir)

        logger.info(f"atomic_output: committed → {final_dir}")

    except Exception:
        # ── Rollback: clean up tmp ────────────────────────────────────
        logger.warning(f"atomic_output: rolling back {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


# ═════════════════════════════════════════════════════════════════════════════
# Built-in validators
# ═════════════════════════════════════════════════════════════════════════════


def validate_files_exist(*required_names: str) -> Callable[[Path], None]:
    """
    Factory: returns a validator that checks required files exist.

    Usage:
        with atomic_output(out, validator=validate_files_exist("model.pt", "metrics.json")):
            ...
    """

    def _validate(tmp_dir: Path) -> None:
        missing = [name for name in required_names if not (tmp_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing required outputs: {missing} in {tmp_dir}")

    return _validate


def validate_parquet_not_empty(*parquet_names: str) -> Callable[[Path], None]:
    """
    Factory: validator that checks parquet files exist and have > 0 rows.
    """

    def _validate(tmp_dir: Path) -> None:
        import pandas as pd

        for name in parquet_names:
            path = tmp_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Missing: {path}")
            df = pd.read_parquet(str(path))
            if len(df) == 0:
                raise ValueError(f"Empty parquet: {path}")

    return _validate


def validate_no_nans(
    *parquet_names: str, columns: list[str] | None = None
) -> Callable[[Path], None]:
    """
    Factory: validator that checks parquet files have no NaN in critical columns.
    """

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
                    n_nan = df[col].isna().sum()
                    raise ValueError(f"NaN in {path}:{col} ({n_nan}/{len(df)} rows)")

    return _validate


def compose_validators(*validators: Callable[[Path], None]) -> Callable[[Path], None]:
    """Combine multiple validators into one."""

    def _validate(tmp_dir: Path) -> None:
        for v in validators:
            v(tmp_dir)

    return _validate
