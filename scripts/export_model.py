"""CLI entry point for portable model export."""

import hydra
from omegaconf import DictConfig

from oceanpath.workflows.export import run_export


@hydra.main(config_path="../configs", config_name="export", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(run_export(cfg).to_json())


if __name__ == "__main__":
    main()
