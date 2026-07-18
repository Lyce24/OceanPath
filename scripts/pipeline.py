"""CLI entry point for the essential foundation pipeline."""

import hydra
from omegaconf import DictConfig

from oceanpath.workflows.pipeline import run_pipeline_workflow


@hydra.main(config_path="../configs", config_name="pipeline", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_pipeline_workflow(cfg)


if __name__ == "__main__":
    main()
