"""Feature-storage domain."""

from oceanpath.storage.mmap import (
    MMAP_SCHEMA_VERSION,
    BuildResult,
    MmapBuildConfig,
    build_mmap,
    validate_mmap_dir,
)

__all__ = [
    "MMAP_SCHEMA_VERSION",
    "BuildResult",
    "MmapBuildConfig",
    "build_mmap",
    "validate_mmap_dir",
]
