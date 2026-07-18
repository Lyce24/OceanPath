"""Canonical logical slide identifiers shared by every pipeline stage."""

from __future__ import annotations

from pathlib import Path

KNOWN_SLIDE_EXTENSIONS = frozenset(
    {
        ".h5",
        ".pt",
        ".npy",
        ".npz",
        ".parquet",
        ".svs",
        ".tiff",
        ".tif",
        ".ndpi",
        ".mrxs",
        ".vsi",
        ".scn",
        ".sdpc",
        ".bif",
    }
)


def normalize_slide_id(value: object, *, root: str | Path | None = None) -> str:
    """Return the extension-free, POSIX-style logical ID for a slide.

    Known WSI and feature-artifact suffixes are removed case-insensitively;
    arbitrary dots (for example in TCGA identifiers) remain intact. When a
    root is supplied, absolute or root-prefixed values are made relative to it.
    """
    raw = str(value).strip().replace("\\", "/")
    if raw:
        candidate = Path(raw).expanduser()
        if root is not None and candidate.is_absolute():
            root_path = Path(str(root)).expanduser().resolve(strict=False)
            try:
                raw = candidate.resolve(strict=False).relative_to(root_path).as_posix()
            except ValueError:
                raw = candidate.name
        elif candidate.is_absolute():
            raw = candidate.name
    raw = raw.removeprefix("./")
    for suffix in KNOWN_SLIDE_EXTENSIONS:
        if raw.lower().endswith(suffix):
            return raw[: -len(suffix)]
    return raw


__all__ = ["KNOWN_SLIDE_EXTENSIONS", "normalize_slide_id"]
