"""CLI entry point for the feature-extraction workflow."""

import hydra
from omegaconf import DictConfig

from oceanpath.workflows.extraction import run_extraction


@hydra.main(config_path="../configs", config_name="extract", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(run_extraction(cfg).to_json())


if __name__ == "__main__":
    main()
