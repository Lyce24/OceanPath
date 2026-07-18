"""CLI entry point for basic model evaluation."""

import hydra
from omegaconf import DictConfig

from oceanpath.workflows.evaluation import run_evaluation


@hydra.main(config_path="../configs", config_name="evaluate", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(run_evaluation(cfg).to_json())


if __name__ == "__main__":
    main()
