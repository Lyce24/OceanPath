"""
Serve an exported artifact directory without Hydra.

    python -m oceanpath.serving --artifact-dir artifacts/my_experiment --port 8000

This is the CPU serving image's entry point. The artifact directory is what
scripts/export_model.py writes (model.onnx, model.pt, model_card.json). The ONNX
backend needs only numpy and onnxruntime; the TorchScript and PyTorch backends
also need torch.
"""

import argparse
import logging
import os


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m oceanpath.serving", description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        default=os.environ.get("OCEANPATH_ARTIFACT_DIR"),
        help="exported artifact directory (default: $OCEANPATH_ARTIFACT_DIR)",
    )
    parser.add_argument(
        "--backend", default="auto", choices=["auto", "onnx", "torchscript", "pytorch"]
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ["OCEANPATH_THREADS"])
        if os.environ.get("OCEANPATH_THREADS")
        else None,
        help="intra-op CPU threads (default: $OCEANPATH_THREADS, else the runtime's choice)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if not args.artifact_dir:
        parser.error("--artifact-dir is required (or set OCEANPATH_ARTIFACT_DIR)")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

    import uvicorn

    from oceanpath.serving.server import create_app

    app = create_app(
        artifact_dir=args.artifact_dir,
        backend=args.backend,
        device=args.device,
        num_threads=args.threads,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
