#!/usr/bin/env python
"""
Export a randomly initialised MIL classifier as a serving artifact.

The weights are random, so the predictions mean nothing. The artifact exercises
the export → serve path end to end (ONNX + TorchScript export with numerical
validation, model card, FastAPI server) without data or a trained checkpoint.
CI builds one for the serving-image smoke test, and scripts/benchmark_serving.py
builds one per architecture.

Usage:
    python scripts/make_demo_artifact.py --out artifacts/demo
    python scripts/make_demo_artifact.py --arch transmil --in-dim 1536 --out artifacts/transmil
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from oceanpath.serving.demo import export_random_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("artifacts/demo"))
    parser.add_argument("--arch", default="abmil", help="a configs/model/<arch>.yaml name")
    parser.add_argument("--in-dim", type=int, default=1536, help="patch feature dim (UNI2-h: 1536)")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    report = export_random_model(args.out, args.arch, args.in_dim, args.num_classes, args.seed)
    print(json.dumps({"output_dir": str(args.out), "success": report["success"]}, indent=2))
    if not report["success"]:
        raise SystemExit("export validation failed; see export_report.json")


if __name__ == "__main__":
    main()
