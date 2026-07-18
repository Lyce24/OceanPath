"""CLI entry point for the split-generation workflow."""

import hydra
from omegaconf import DictConfig

from oceanpath.workflows.splitting import run_split_generation


@hydra.main(config_path="../configs", config_name="make_splits", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(run_split_generation(cfg).to_json())


if __name__ == "__main__":
    main()
