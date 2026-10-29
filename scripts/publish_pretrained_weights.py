#!/usr/bin/env python
"""
Stage, and optionally upload, the SSL-pretrained Mamba-2 MIL aggregators to the
Hugging Face Hub with a model card.

The aggregators were pretrained on UNI2-h patch features. MahmoodLab releases UNI2-h
and its features under CC-BY-NC-ND 4.0 and counts models trained on its outputs as
derivatives that may not be used commercially without approval, so a release has to
carry the same non-commercial terms, be gated, and be cleared with the UNI2-h authors
first. This script therefore stages everything locally by default; it uploads only
with --upload, to a private repository, with manual access approval.

Usage:
    python scripts/publish_pretrained_weights.py --out build/hf_release           # stage only
    python scripts/publish_pretrained_weights.py --out build/hf_release \
        --repo-id <user>/oceanpath-mamba2mil-uni2h --upload                       # needs HF login
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parent.parent
METHODS = {
    "vicreg": "VICReg (variance-invariance-covariance)",
    "jepa": "JEPA (predictive, EMA target)",
    "lejepa": "LeJEPA (predict-to-centroid + SIGReg, two views)",
    "lejepa_mc": "LeJEPA-MC (multi-crop LeJEPA)",
}
# GPU-hours of the pretraining run behind each released checkpoint (wandb run records).
COMPUTE = {
    "vicreg": "26.0 GPU-h, 1x RTX A5000",
    "jepa": "20.1 GPU-h, 2x RTX A5000",
    "lejepa": "18.3 GPU-h, 2x RTX A5500",
    "lejepa_mc": "21.1 GPU-h, 2x RTX 3090",
}

CARD_HEADER = """---
license: cc-by-nc-nd-4.0
library_name: pytorch
tags:
- histopathology
- computational-pathology
- multiple-instance-learning
- self-supervised
- mamba2
extra_gated_prompt: >-
  These slide-level aggregators were trained on UNI2-h patch features, which MahmoodLab
  releases under CC-BY-NC-ND 4.0. They may only be used for non-commercial, academic
  research with attribution to UNI2-h and OceanPath. Commercial use requires prior
  approval from the UNI2-h authors.
extra_gated_fields:
  Full name: text
  Affiliation: text
  I agree to use these weights for non-commercial academic research only: checkbox
---
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _results_table() -> str:
    """Fine-tuning results from the committed results files, if present."""
    path = REPO / "results" / "supervised" / "summary.csv"
    if not path.is_file():
        return "See the OceanPath README for downstream results.\n"
    s = pd.read_csv(path)
    keep = s[(s.metric.isin(["auroc", "qwk"])) & (s.init.str.startswith("mamba2mil"))]
    lines = [
        "| Initialisation | Task | Split | Metric | Mean ± SD over seeds |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in keep.itertuples(index=False):
        lines.append(
            f"| {r.init} | {r.data} | {r.split} | {r.metric} | {r.mean:.3f} ± {r.sd:.3f} |"
        )
    return "\n".join(lines) + "\n"


def stage(ckpt_dir: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    model_cfg = OmegaConf.to_container(
        OmegaConf.load(REPO / "configs" / "model" / "mamba2mil.yaml")
    )
    manifest = {}
    for method in METHODS:
        src = ckpt_dir / f"aggregator_weights_{method}.pt"
        state = torch.load(src, map_location="cpu", weights_only=True)
        dst = out / f"mamba2mil_uni2h_{method}.safetensors"
        save_file({k: v.contiguous() for k, v in state.items()}, str(dst))
        manifest[method] = {"file": dst.name, "sha256": _sha256(dst), "source_sha256": _sha256(src)}
    config = {
        "architecture": "mamba2mil",
        "in_dim": 1536,
        "patch_encoder": "UNI2-h (MahmoodLab), 256 px patches at 20x",
        "model": model_cfg,
        "checkpoints": manifest,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    rows = "\n".join(
        f"| `{m}` | {desc} | {COMPUTE[m]} | `{manifest[m]['file']}` |"
        for m, desc in METHODS.items()
    )
    card = (
        CARD_HEADER
        + f"""
# OceanPath Mamba-2 MIL aggregators, pretrained on UNI2-h features

Four slide-level aggregators (Mamba-2 MIL, 2 SSD blocks, d_state 64, embed dim 384) pretrained
without labels on UNI2-h patch features of 14,529 TCGA and CPTAC slides (142.7M patches,
slides with at least 2,048 patches), one per self-supervised objective. Code, training
configs and the full evaluation are in [OceanPath](https://github.com/Lyce24/OceanPath).

| Objective | Method | Pretraining compute (100 epochs) | File |
| --- | --- | --- | --- |
{rows}

## Use

```python
from safetensors.torch import load_file
from oceanpath.models import build_classifier

model = build_classifier("mamba2mil", in_dim=1536, num_classes=2, model_cfg=...)  # configs/model/mamba2mil.yaml
model.aggregator.load_state_dict(load_file("mamba2mil_uni2h_lejepa.safetensors"))
```

Inputs are bags of UNI2-h patch features (`[N, 1536]`, 256 px patches at 20x). The Mamba-2
kernels need a CUDA GPU (`pip install "oceanpath[mamba]"`).

## Downstream results (fine-tuning, patient level)

{_results_table()}
## Limitations

- Pretrained only on public TCGA and CPTAC slides; behaviour on other scanners, stains and
  sites is not characterised beyond the CPTAC external test.
- In the OceanPath evaluations these initialisations do not consistently beat a randomly
  initialised Mamba-2 MIL or ABMIL trained from scratch; see the README before relying on them.
- Not a medical device and not validated for clinical use.

## License

CC-BY-NC-ND 4.0, inherited from UNI2-h. Cite UNI (Chen et al., Nature Medicine 2024) and OceanPath.
"""
    )
    (out / "README.md").write_text(card)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path(os.environ.get("OCEANPATH_ROOT", ".")) / "checkpoints",
        help="folder with aggregator_weights_<method>.pt (default: $OCEANPATH_ROOT/checkpoints)",
    )
    parser.add_argument("--out", type=Path, default=Path("build/hf_release"))
    parser.add_argument("--repo-id", help="e.g. <user>/oceanpath-mamba2mil-uni2h")
    parser.add_argument("--upload", action="store_true", help="push to a private, gated repo")
    args = parser.parse_args()

    config = stage(args.ckpt_dir, args.out)
    print(json.dumps(config["checkpoints"], indent=2))
    print(f"staged in {args.out}")
    if not args.upload:
        print("not uploaded (pass --upload with --repo-id after the UNI2-h terms are cleared)")
        return
    if not args.repo_id:
        parser.error("--upload needs --repo-id")
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, private=True, exist_ok=True)
    api.update_repo_settings(args.repo_id, gated="manual")
    api.upload_folder(folder_path=str(args.out), repo_id=args.repo_id)
    print(f"uploaded to https://huggingface.co/{args.repo_id} (private, manual approval)")


if __name__ == "__main__":
    main()
