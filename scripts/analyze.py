from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oceanpath.eval.significance import paired_permutation_auc_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two model prediction CSVs")
    parser.add_argument("--pred-a", required=True, help="CSV with y_true,y_prob for model A")
    parser.add_argument("--pred-b", required=True, help="CSV with y_true,y_prob for model B")
    parser.add_argument("--output", required=True, help="Path to save analysis json")
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    a = pd.read_csv(args.pred_a)
    b = pd.read_csv(args.pred_b)

    if not {"y_true", "y_prob"}.issubset(a.columns) or not {"y_true", "y_prob"}.issubset(b.columns):
        raise ValueError("Both prediction files must contain y_true and y_prob columns")

    if len(a) != len(b):
        raise ValueError("Prediction files must have the same number of rows")

    if not (a["y_true"].to_numpy() == b["y_true"].to_numpy()).all():
        raise ValueError("y_true mismatch between files")

    result = paired_permutation_auc_test(
        y_true=a["y_true"].to_numpy(),
        y_prob_a=a["y_prob"].to_numpy(),
        y_prob_b=b["y_prob"].to_numpy(),
        n_permutations=args.n_permutations,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
