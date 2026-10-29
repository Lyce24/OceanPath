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
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Project root: 2 levels up from src/oceanpath/utils/repro.py
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ═════════════════════════════════════════════════════════════════════════════
# Config fingerprinting
# ═════════════════════════════════════════════════════════════════════════════


def config_fingerprint(cfg) -> str:
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
) -> dict:
    """
    Compare installed packages against uv.lock.

    Warns on divergence so you know before training starts that your
    environment doesn't match the lockfile. This catches "it works on
    my machine" problems when moving to a different server.

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

    # Parse installed packages
    installed = _get_installed_packages()

    # Parse lockfile packages
    locked = _parse_uv_lock(lockfile)

    if not locked:
        return {"status": "parse_error", "message": "Could not parse uv.lock"}

    # Compare
    mismatches = []
    for pkg, lock_ver in locked.items():
        inst_ver = installed.get(pkg)
        if inst_ver is None:
            mismatches.append({"package": pkg, "locked": lock_ver, "installed": None})
        elif inst_ver != lock_ver:
            mismatches.append({"package": pkg, "locked": lock_ver, "installed": inst_ver})

    result = {
        "status": "ok" if not mismatches else "mismatch",
        "mismatches": mismatches,
        "installed_count": len(installed),
        "lock_count": len(locked),
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
        logger.info(f"Environment matches {lockfile.name} ({len(locked)} packages)")

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


def _parse_uv_lock(lockfile: Path) -> dict[str, str]:
    """
    Parse uv.lock to extract package versions.

    uv.lock is TOML-like with [[package]] sections.
    We extract name + version from each.
    """
    packages = {}
    text = lockfile.read_text()

    current_name = None
    current_version = None

    for line in text.splitlines():
        line = line.strip()

        if line == "[[package]]":
            # Save previous
            if current_name and current_version:
                packages[current_name.lower().replace("-", "_")] = current_version
            current_name = None
            current_version = None
        elif line.startswith("name = "):
            current_name = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("version = ") and current_name:
            current_version = line.split("=", 1)[1].strip().strip('"')

    # Don't forget the last package
    if current_name and current_version:
        packages[current_name.lower().replace("-", "_")] = current_version

    return packages


# ═════════════════════════════════════════════════════════════════════════════
# Git info
# ═════════════════════════════════════════════════════════════════════════════


def _git_info(repo_dir: Path | None = None) -> dict:
    """Capture git SHA, branch, and dirty status."""
    repo_dir = repo_dir or _PROJECT_ROOT
    info = {"sha": None, "branch": None, "dirty": None}

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
    cfg=None,
    csv_path: str | Path | None = None,
) -> dict:
    """
    Capture full experiment provenance.

    Returns a dict suitable for saving alongside experiment outputs.
    Includes: git info, config fingerprint, manifest hash, CUDA info,
    Python version, platform, timestamp.
    """
    import torch

    prov = {
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


def save_provenance(provenance: dict, path: str | Path) -> None:
    """Save provenance dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2, default=str))
    logger.info(f"Provenance → {path}")
