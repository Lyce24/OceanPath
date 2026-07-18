"""CLI entry point for supervised MIL training."""

import hydra
from omegaconf import DictConfig

from oceanpath.workflows.training import run_training


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(run_training(cfg).to_json())


if __name__ == "__main__":
    main()
