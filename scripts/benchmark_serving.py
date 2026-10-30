#!/usr/bin/env python
"""
CPU serving latency: p50/p95 per slide for PyTorch, TorchScript and ONNX Runtime
across bag sizes, measured through the server's own InferenceBackend, plus the
HTTP round trip of POST /predict/npy and POST /predict.

Each architecture is exported once with random weights (oceanpath.serving.demo);
latency does not depend on the weights. One "slide" is one bag of N patch
features of dimension D (UNI2-h: 1536), batch size 1, as the server handles it.

Usage:
    python scripts/benchmark_serving.py --out results/serving_latency
    python scripts/benchmark_serving.py --archs abmil --bag-sizes 1000 8000 --threads 4

Writes latency.csv (one row per arch x backend x path x threads x bag size),
environment.json and latency.md under --out.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import socket
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from oceanpath.serving.demo import export_random_model
from oceanpath.serving.server import InferenceBackend, create_app

BACKENDS = ("pytorch", "torchscript", "onnx")


def _time_calls(fn, warmup: int, repeats: int, budget_s: float) -> list[float]:
    """Milliseconds per call; stops early once the time budget is spent."""
    for _ in range(warmup):
        fn()
    times: list[float] = []
    deadline = time.perf_counter() + budget_s
    while len(times) < repeats and (len(times) < 5 or time.perf_counter() < deadline):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def _summary(times: list[float]) -> dict:
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": statistics.fmean(times),
        "n_calls": len(times),
    }


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    """uvicorn in a background thread, so HTTP timings include real sockets."""

    def __init__(self, artifact_dir: Path, threads: int):
        import uvicorn

        self.port = _free_port()
        app = create_app(str(artifact_dir), backend="onnx", num_threads=threads)
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        while not self.server.started:
            time.sleep(0.05)
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=10)


def _environment(threads: list[int]) -> dict:
    import onnxruntime
    import torch

    cpu = platform.processor() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "logical_cores": os.cpu_count(),
        "threads_tested": threads,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
        "platform": platform.platform(),
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    import httpx

    rng = np.random.default_rng(0)
    rows, skipped = [], []
    for arch in args.archs:
        artifact = args.out / "artifacts" / arch
        report = export_random_model(
            artifact, arch=arch, in_dim=args.in_dim, validation_n_patches=[10, 100]
        )
        # Benchmark only formats whose export and parity check passed; PyTorch always runs.
        usable = ["pytorch"] + [
            fmt
            for fmt, check in (("torchscript", "torchscript_numerical"), ("onnx", "onnx_numerical"))
            if report["validation"].get(check, {}).get("status") == "pass"
        ]
        for fmt in sorted(set(BACKENDS) - set(usable)):
            print(f"{arch}: {fmt} export did not pass its parity check; not benchmarked")
            skipped.append({"arch": arch, "backend": fmt, "reason": "parity check failed"})

        for threads in args.threads:
            for backend in [b for b in BACKENDS if b in usable]:
                engine = InferenceBackend(str(artifact), backend=backend, num_threads=threads)
                for n in args.bag_sizes:
                    x = rng.standard_normal((n, args.in_dim)).astype(np.float32)
                    times = _time_calls(
                        lambda e=engine, x=x: e.predict(x), args.warmup, args.repeats, args.budget
                    )
                    rows.append(
                        {
                            "arch": arch,
                            "path": "model",
                            "backend": backend,
                            "threads": threads,
                            "n_patches": n,
                            **_summary(times),
                        }
                    )
                    print(
                        f"{arch:10s} {backend:12s} t={threads} N={n:6d} "
                        f"p50={rows[-1]['p50_ms']:8.1f} ms p95={rows[-1]['p95_ms']:8.1f} ms"
                    )

            if "onnx" not in usable:
                continue
            # HTTP round trip through the ONNX server: .npy body for every size,
            # JSON nested lists only for the sizes where that is still practical.
            with _Server(artifact, threads) as base, httpx.Client(timeout=300) as client:
                for n in args.bag_sizes:
                    x = rng.standard_normal((n, args.in_dim)).astype(np.float32)
                    buf = io.BytesIO()
                    np.save(buf, x)
                    body = buf.getvalue()

                    def post_npy(body=body):
                        client.post(f"{base}/predict/npy", content=body).raise_for_status()

                    times = _time_calls(post_npy, args.warmup, args.repeats, args.budget)
                    rows.append(
                        {
                            "arch": arch,
                            "path": "http_npy",
                            "backend": "onnx",
                            "threads": threads,
                            "n_patches": n,
                            **_summary(times),
                        }
                    )
                    if n <= args.max_json_patches:
                        payload = json.dumps({"features": x.tolist()})

                        def post_json(payload=payload):
                            client.post(
                                f"{base}/predict",
                                content=payload,
                                headers={"Content-Type": "application/json"},
                            ).raise_for_status()

                        times = _time_calls(post_json, 1, max(5, args.repeats // 3), args.budget)
                        rows.append(
                            {
                                "arch": arch,
                                "path": "http_json",
                                "backend": "onnx",
                                "threads": threads,
                                "n_patches": n,
                                **_summary(times),
                            }
                        )
    df = pd.DataFrame(rows)
    df.attrs["skipped"] = skipped
    return df


def _md_table(df: pd.DataFrame) -> str:
    """A pipe table without pandas' optional tabulate dependency."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(" --- " for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _markdown(df: pd.DataFrame) -> str:
    """One table per thread count: rows = arch x path/backend, columns = bag size."""
    df = df.assign(
        cell=df.apply(lambda r: f"{r.p50_ms:,.1f} / {r.p95_ms:,.1f}", axis=1),
        label=df.apply(
            lambda r: r.backend if r.path == "model" else f"HTTP {r.path[5:]} (onnx)", axis=1
        ),
    )
    order = {"pytorch": 0, "torchscript": 1, "onnx": 2, "HTTP npy (onnx)": 3, "HTTP json (onnx)": 4}
    out = []
    for threads, sub in df.groupby("threads"):
        table = sub.pivot_table(
            index=["arch", "label"], columns="n_patches", values="cell", aggfunc="first"
        ).fillna("—")
        table = table.reset_index()
        table["_o"] = table["label"].map(order)
        table = table.sort_values(["arch", "_o"]).drop(columns="_o")
        table.columns = ["Aggregator", "Path"] + [f"N = {c:,}" for c in table.columns[2:]]
        out.append(f"#### {threads} CPU thread{'s' if threads > 1 else ''} (p50 / p95 ms)\n")
        out.append(_md_table(table))
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("results/serving_latency"))
    parser.add_argument("--archs", nargs="+", default=["abmil", "transmil", "perceiver"])
    parser.add_argument("--bag-sizes", nargs="+", type=int, default=[500, 2000, 8000, 32000])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--in-dim", type=int, default=1536)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--budget", type=float, default=30.0, help="seconds per cell, at least 5 calls"
    )
    parser.add_argument("--max-json-patches", type=int, default=2000)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = run(args)
    df.to_csv(args.out / "latency.csv", index=False)
    env = _environment(args.threads) | {"not_benchmarked": df.attrs.get("skipped", [])}
    (args.out / "environment.json").write_text(json.dumps(env, indent=2))
    (args.out / "latency.md").write_text(_markdown(df) + "\n")
    print(_markdown(df))


if __name__ == "__main__":
    main()
