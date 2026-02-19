from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from oceanpath.modules.wsi_module import WSIClassificationModule


class ExportWrapper(torch.nn.Module):
	def __init__(self, model: torch.nn.Module):
		super().__init__()
		self.model = model

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		logits, _ = self.model(x)
		return logits


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Export OceanPath model")
	parser.add_argument("--checkpoint", required=True)
	parser.add_argument("--output-dir", required=True)
	parser.add_argument("--bag-size", type=int, default=128)
	parser.add_argument("--feature-dim", type=int, required=True)
	parser.add_argument("--opset", type=int, default=17)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	module = WSIClassificationModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
	model = module.model.eval().cpu()
	export_model = ExportWrapper(model).eval()

	dummy = torch.randn(1, args.bag_size, args.feature_dim, dtype=torch.float32)

	ts_path = output_dir / "wsi_model.ts"
	traced = torch.jit.trace(export_model, (dummy,))
	traced.save(str(ts_path))

	onnx_path = output_dir / "wsi_model.onnx"
	torch.onnx.export(
		export_model,
		args=(dummy,),
		f=str(onnx_path),
		input_names=["features"],
		output_names=["logits"],
		dynamic_axes={
			"features": {0: "batch", 1: "tiles"},
			"logits": {0: "batch"},
		},
		opset_version=args.opset,
	)

	print(f"TorchScript: {ts_path}")
	print(f"ONNX: {onnx_path}")


if __name__ == "__main__":
	main()

