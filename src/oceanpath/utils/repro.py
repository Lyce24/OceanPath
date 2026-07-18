"""
Reproducibility utilities — environment verification and provenance capture.

Every experiment should answer: "What exact code, data, packages, and config
produced these results?" This module captures that provenance and verifies
the current environment matches the lockfile.

Usage at script entrypoint:
    from oceanpath.utils.repro import verify_environment, capture_provenance

    verify_environment()  # warns if packages diverge from uv.lock
    prov = capture_provenance(cfg)
    save_provenance(prov, output_dir / "provenance.json")
"""

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Project root: 2 levels up from src/oceanpath/utils/repro.py
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ═════════════════════════════════════════════════════════════════════════════
# Config fingerprinting
# ═════════════════════════════════════════════════════════════════════════════


def config_fingerprint(cfg: Any) -> str:
    """
    Deterministic hash of the resolved Hydra config.

    Same config → same fingerprint → outputs are comparable or skippable.
    Different learning rate → different fingerprint → new output directory.

    Parameters
    ----------
    cfg : DictConfig or dict
        The fully resolved config (after Hydra overrides).

    Returns
    -------
    12-char hex string (sha256 prefix).
    """
    from omegaconf import OmegaConf

    if hasattr(cfg, "_metadata"):  # DictConfig
        canonical = OmegaConf.to_yaml(cfg, sort_keys=True)
    elif isinstance(cfg, dict):
        canonical = json.dumps(cfg, sort_keys=True, default=str)
    else:
        canonical = str(cfg)

    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ═════════════════════════════════════════════════════════════════════════════
# Data versioning
# ═════════════════════════════════════════════════════════════════════════════


def manifest_hash(csv_path: str | Path) -> str:
    """
    Hash a manifest CSV to identify the exact slide set.

    Uses SHA-256 of the sorted file contents so the hash is stable
    regardless of row ordering in the CSV.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return "missing"

    lines = csv_path.read_text().strip().splitlines()
    if len(lines) < 2:
        return "empty"

    header = lines[0]
    body = sorted(lines[1:])  # sort data rows for order-independence
    canonical = header + "\n" + "\n".join(body)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ═════════════════════════════════════════════════════════════════════════════
# Environment verification
# ═════════════════════════════════════════════════════════════════════════════


def verify_environment(
    lockfile: str | Path | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """
    Verify that the active project environment matches uv.lock.

    For the project lockfile, delegates marker/extras handling to
    ``uv sync --check --frozen --inexact`` so documented optional extras can
    coexist with the base environment. Custom lockfiles and systems without uv
    use the small parser below as a fallback.

    Parameters
    ----------
    lockfile : path to uv.lock (auto-detected if None).
    strict : if True, raise RuntimeError on mismatch.

    Returns
    -------
    dict with: status, mismatches, installed_count, lock_count.
    """
    if lockfile is None:
        lockfile = _PROJECT_ROOT / "uv.lock"
    lockfile = Path(lockfile)

    if not lockfile.is_file():
        msg = f"Lockfile not found: {lockfile}"
        logger.warning(msg)
        return {"status": "missing", "message": msg}

    project_lockfile = (_PROJECT_ROOT / "uv.lock").resolve()
    if lockfile.resolve() == project_lockfile:
        uv_result = _verify_project_environment_with_uv(lockfile, strict=strict)
        if uv_result is not None:
            return uv_result

    # Parse installed and locked packages. A uv lock can contain multiple
    # versions of one dependency for different Python/platform markers, so a
    # fallback audit must accept any locked variant rather than whichever entry
    # happened to appear last in the file.
    installed = _get_installed_packages()
    locked_versions = _parse_uv_lock_versions(lockfile)

    if not locked_versions:
        return {"status": "parse_error", "message": "Could not parse uv.lock"}

    mismatches: list[dict[str, Any]] = []
    for pkg, versions in locked_versions.items():
        inst_ver = installed.get(pkg)
        # A lock may contain optional extras that are intentionally absent.
        # Without uv's resolver context, only compare packages that are active
        # in the current environment.
        if inst_ver is not None and inst_ver not in versions:
            mismatches.append(
                {
                    "package": pkg,
                    "locked": " | ".join(sorted(versions)),
                    "installed": inst_ver,
                    "reason": "version_mismatch",
                }
            )

    # Unlike optional/transitive entries, direct dependencies of the editable
    # project are required in every usable OceanPath environment. The old
    # fallback silently treated a missing required package as a valid absent
    # extra, which could incorrectly report an incomplete environment as clean.
    for pkg in sorted(_parse_required_lock_packages(lockfile)):
        if pkg not in installed:
            mismatches.append(
                {
                    "package": pkg,
                    "locked": " | ".join(sorted(locked_versions.get(pkg, {"required"}))),
                    "installed": None,
                    "reason": "missing_required",
                }
            )

    result = {
        "status": "ok" if not mismatches else "mismatch",
        "mismatches": mismatches,
        "installed_count": len(installed),
        "lock_count": len(locked_versions),
        "lockfile": str(lockfile),
    }

    if mismatches:
        mismatch_strs = [
            f"  {m['package']}: locked={m['locked']}, installed={m['installed']}"
            for m in mismatches[:10]
        ]
        msg = (
            f"Environment diverges from {lockfile.name} "
            f"({len(mismatches)} packages):\n" + "\n".join(mismatch_strs)
        )
        if len(mismatches) > 10:
            msg += f"\n  ... and {len(mismatches) - 10} more"

        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
    else:
        logger.info(f"Environment matches {lockfile.name} ({len(locked_versions)} packages)")

    return result


def _verify_project_environment_with_uv(
    lockfile: Path,
    *,
    strict: bool,
) -> dict[str, Any] | None:
    """Use uv's resolver-aware audit when uv is available."""
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        return None

    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = sys.prefix
    completed = subprocess.run(
        [
            uv_executable,
            "sync",
            "--check",
            "--frozen",
            # Optional extract/serve/track packages are deliberately allowed in
            # the same environment. Exact sync would flag those documented
            # extras as removals when checking the base project.
            "--inexact",
            "--project",
            str(_PROJECT_ROOT),
        ],
        cwd=_PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    installed = _get_installed_packages()
    locked = _parse_uv_lock(lockfile)
    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    status = "ok" if completed.returncode == 0 else "mismatch"
    result: dict[str, Any] = {
        "status": status,
        "mismatches": [] if status == "ok" else [{"message": output}],
        "installed_count": len(installed),
        "lock_count": len(locked),
        "lockfile": str(lockfile),
        "method": "uv_sync_check",
    }

    if status == "ok":
        logger.info("Environment matches %s (verified by uv)", lockfile.name)
        return result

    message = f"Environment diverges from {lockfile.name}: {output}"
    if strict:
        raise RuntimeError(message)
    logger.warning(message)
    return result


def _get_installed_packages() -> dict[str, str]:
    """Get currently installed packages as {name: version}."""
    packages = {}
    try:
        # Use importlib.metadata (stdlib, fast, no subprocess)
        from importlib.metadata import distributions

        for dist in distributions():
            name = dist.metadata["Name"]
            if name:
                packages[name.lower().replace("-", "_")] = dist.version
    except Exception:
        # Fallback to pip list
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                for pkg in json.loads(result.stdout):
                    packages[pkg["name"].lower().replace("-", "_")] = pkg["version"]
        except Exception:
            pass

    return packages


def _parse_uv_lock_versions(lockfile: Path) -> dict[str, set[str]]:
    """Parse every locked version for each normalized package name."""
    packages: dict[str, set[str]] = {}
    current_name: str | None = None
    current_version: str | None = None

    def save_current() -> None:
        if current_name and current_version:
            normalized = current_name.lower().replace("-", "_")
            packages.setdefault(normalized, set()).add(current_version)

    for raw_line in lockfile.read_text().splitlines():
        line = raw_line.strip()
        if line == "[[package]]":
            save_current()
            current_name = None
            current_version = None
        elif line.startswith("name = "):
            current_name = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("version = ") and current_name:
            current_version = line.split("=", 1)[1].strip().strip('"')

    save_current()
    return packages


def _parse_required_lock_packages(lockfile: Path) -> set[str]:
    """Return direct dependencies of the editable project in a uv lockfile."""
    required: set[str] = set()
    for block in lockfile.read_text().split("[[package]]")[1:]:
        if not re.search(r'^source\s*=\s*\{\s*(?:editable|virtual)\s*=\s*"\."\s*\}', block, re.M):
            continue

        in_dependencies = False
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if line == "dependencies = [":
                in_dependencies = True
                continue
            if in_dependencies and line == "]":
                break
            if not in_dependencies:
                continue
            match = re.search(r'\bname\s*=\s*"([^"]+)"', line)
            if match:
                required.add(match.group(1).lower().replace("-", "_"))
    return required


def _parse_uv_lock(lockfile: Path) -> dict[str, str]:
    """
    Parse uv.lock to extract package versions.

    uv.lock is TOML-like with [[package]] sections.
    We extract name + version from each.
    """
    return {
        name: sorted(versions)[-1] for name, versions in _parse_uv_lock_versions(lockfile).items()
    }


# ═════════════════════════════════════════════════════════════════════════════
# Git info
# ═════════════════════════════════════════════════════════════════════════════


def _git_info(repo_dir: Path | None = None) -> dict[str, Any]:
    """Capture git SHA, branch, and dirty status."""
    repo_dir = repo_dir or _PROJECT_ROOT
    info: dict[str, Any] = {"sha": None, "branch": None, "dirty": None}

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=5,
        )
        if sha.returncode == 0:
            info["sha"] = sha.stdout.strip()

        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=5,
        )
        if branch.returncode == 0:
            info["branch"] = branch.stdout.strip()

        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=5,
        )
        if dirty.returncode == 0:
            info["dirty"] = len(dirty.stdout.strip()) > 0
    except Exception:
        pass

    return info


# ═════════════════════════════════════════════════════════════════════════════
# Provenance capture
# ═════════════════════════════════════════════════════════════════════════════


def capture_provenance(
    cfg: Any = None,
    csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Capture full experiment provenance.

    Returns a dict suitable for saving alongside experiment outputs.
    Includes: git info, config fingerprint, manifest hash, CUDA info,
    Python version, platform, timestamp.
    """
    import torch

    prov: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git": _git_info(),
    }

    # Config fingerprint
    if cfg is not None:
        prov["config_fingerprint"] = config_fingerprint(cfg)

    # Data version
    if csv_path is not None:
        prov["manifest_hash"] = manifest_hash(csv_path)

    # PyTorch / CUDA
    prov["torch_version"] = torch.__version__
    prov["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        prov["cuda_version"] = torch.version.cuda
        prov["gpu_name"] = torch.cuda.get_device_name(0)
        prov["gpu_count"] = torch.cuda.device_count()
    else:
        prov["cuda_version"] = None
        prov["gpu_name"] = None

    return prov


def save_provenance(provenance: dict[str, Any], path: str | Path) -> None:
    """Save provenance dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2, default=str))
    logger.info(f"Provenance → {path}")
