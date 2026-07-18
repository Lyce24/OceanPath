"""Run lifecycle, provenance, and completion metadata."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oceanpath.config.access import cfg_select
from oceanpath.utils.repro import capture_provenance, save_provenance, verify_environment


@dataclass(frozen=True)
class RunContext:
    """Metadata made available to a workflow during one stage run."""

    stage: str
    output_dir: Path
    started_at: str
    provenance: dict[str, Any]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    temporary.replace(path)


def _save_config(cfg: Any, path: Path) -> None:
    from omegaconf import OmegaConf

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True))
    temporary.replace(path)


@contextmanager
def run_context(
    cfg: Any,
    *,
    stage: str,
    output_dir: str | Path,
    persist: bool = True,
) -> Iterator[RunContext]:
    """Capture provenance and write a success/failure completion signal."""
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    output_dir = Path(output_dir)
    metadata_dir = output_dir / "_run"

    if bool(cfg_select(cfg, "runtime.verify_environment", False)):
        verify_environment(strict=bool(cfg_select(cfg, "runtime.strict_environment", False)))

    manifest_path = cfg_select(cfg, "data.csv_path", None)
    provenance = capture_provenance(cfg=cfg, csv_path=manifest_path)
    context = RunContext(
        stage=stage,
        output_dir=output_dir,
        started_at=started_at,
        provenance=provenance,
    )

    if persist:
        save_provenance(provenance, metadata_dir / "provenance.json")
        _save_config(cfg, metadata_dir / "config.yaml")

    try:
        yield context
    except BaseException as error:
        if persist:
            _write_json(
                metadata_dir / "status.json",
                {
                    "stage": stage,
                    "status": "failed",
                    "started_at": started_at,
                    "duration_seconds": time.monotonic() - started,
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                },
            )
        raise
    else:
        if persist:
            _write_json(
                metadata_dir / "status.json",
                {
                    "stage": stage,
                    "status": "completed",
                    "started_at": started_at,
                    "duration_seconds": time.monotonic() - started,
                    "config_fingerprint": provenance.get("config_fingerprint"),
                    "git_sha": provenance.get("git", {}).get("sha"),
                },
            )
