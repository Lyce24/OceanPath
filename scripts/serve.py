"""CLI entry point for optional HTTP model serving."""

import hydra
from omegaconf import DictConfig

from oceanpath.workflows.serving import run_server


@hydra.main(config_path="../configs", config_name="serve", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_server(cfg)


if __name__ == "__main__":
    main()
