import argparse
import time

import torch
import lightning.pytorch as pl

import wandb

def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "Missing dependency: pyyaml. Install with: pip install pyyaml"
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def _resolve_device(device_cfg: str) -> torch.device:
    device_cfg = (device_cfg or "auto").lower()
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_cfg == "cuda":
        return torch.device("cuda")
    if device_cfg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device setting: {device_cfg} (use auto|cuda|cpu)")


def _parse_scalar(v: str):
    """Best-effort parse: bool/int/float/null/str."""
    s = v.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    # int?
    try:
        if s.startswith("0") and len(s) > 1 and s[1].isdigit():
            # keep as string to avoid octal-ish surprises
            return s
        return int(s)
    except ValueError:
        pass
    # float?
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _set_nested(d: dict, keypath: str, value):
    """Set d[a][b][c] = value for keypath 'a.b.c' (create dicts as needed)."""
    keys = keypath.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    overrides items like:
      - experiment.batch_size=125
      - wandb.enabled=false
      - device=cuda
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Bad override '{item}'. Use key=value (e.g., experiment.lr=1e-4)")
        k, v = item.split("=", 1)
        _set_nested(cfg, k.strip(), _parse_scalar(v))
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--override", type=str, nargs="*", default=[], help="Override YAML config values")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    cfg = _apply_overrides(cfg, args.override)

    seed = int(cfg.get("seed", 42))
    pl.seed_everything(seed, workers=True)

    device = _resolve_device(cfg.get("device", "auto"))

    # Build ExperimentConfig from YAML
    exp_cfg_dict = cfg.get("experiment", {})
    if not exp_cfg_dict:
        raise ValueError("YAML must contain an 'experiment:' section with ExperimentConfig fields.")

    cv_mode = exp_cfg_dict.get("cv_mode", "k-fold").lower()
    if cv_mode == "k-fold":
        from modules.k_folds_exp import ExperimentLIT, ExperimentConfig
    elif cv_mode == "oof":
        from modules.oof_exp import ExperimentLIT, ExperimentConfig
    else:
        raise ValueError(f"Unknown cv_mode: {cv_mode} (use k-fold|oof)")

    print(f"Using CV mode: {cv_mode}")

    config = ExperimentConfig(**exp_cfg_dict)

    # W&B
    wandb_cfg = cfg.get("wandb", {})
    use_wandb = bool(wandb_cfg.get("enabled", True)) and (not args.no_wandb)

    wandb_run = None
    if use_wandb:
        current_time = time.strftime("%Y%m%d-%H%M%S")
        project = wandb_cfg.get("project", "SSL-Path")
        tags = wandb_cfg.get("tags", [config.mil, "k-fold-cv"])

        if "name" in wandb_cfg:
            run_name = wandb_cfg["name"]
        else:
            run_name = f"{config.mil}_{config.target_col}_seed{config.seed}_{current_time}"

        wandb_run = wandb.init(
            project=project,
            name=run_name,
            config=config.to_dict(),
            tags=tags,
        )

    # Run experiment
    exp = ExperimentLIT(
        config=config,
        device=device,
        wandb_run=wandb_run,
    )

    out = exp.run()
    return out


if __name__ == "__main__":
    main()
