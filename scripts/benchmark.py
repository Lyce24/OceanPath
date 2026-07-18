"""CLI entry point for pipeline throughput benchmarks."""

import hydra
from omegaconf import DictConfig

from oceanpath.workflows.benchmark import run_benchmark


@hydra.main(config_path="../configs", config_name="benchmark", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
