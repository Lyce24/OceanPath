"""CLI entry point for the mmap-build workflow."""

import hydra
from omegaconf import DictConfig

from oceanpath.workflows.mmap import run_mmap_build


@hydra.main(config_path="../configs", config_name="build_mmap", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(run_mmap_build(cfg).to_json())


if __name__ == "__main__":
    main()
