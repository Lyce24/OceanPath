"""
Stage 9: Throughput benchmarking with bottleneck identification.

Profiles the training pipeline in isolation to answer:
  - How many slides/second can you process?
  - What's the bottleneck: data loading, forward pass, backward pass, or optimizer?
  - How does throughput change with max_instances (bag size)?
  - Does AMP actually help on your hardware?

Method:
  Uses synthetic data (random tensors of correct shape) to isolate compute
  from disk I/O when profiling the model, and real memmap data when
  profiling the data loader. This separation is critical — a "training step"
  benchmark that includes disk reads conflates two independent bottlenecks.

Phases:
  1. Data loading   — real memmap reads, no GPU
  2. Forward only   — synthetic data, model forward pass
  3. Forward+backward — synthetic data, full gradient computation
  4. Full train step — synthetic data, forward+backward+optimizer
  5. Inference       — synthetic data, model eval mode
  6. Bag size scaling — forward at varying patch counts

Run after any architecture or data pipeline change:
    python scripts/benchmark.py platform=local data=gej encoder=univ1 \\
           model=abmil training=gej

Output:
    outputs/benchmark/{exp_name}/benchmark_report.json
"""

import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Timing utilities
# ═════════════════════════════════════════════════════════════════════════════


def _sync_cuda():
    """Synchronize CUDA for accurate timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_fn(fn, n_iters: int, warmup: int = 5) -> dict:
    """
    Time a function over n_iters, returning percentile statistics.

    Includes warmup iterations (excluded from timing) to let JIT, CUDA
    caching, and memory allocators stabilize.
    """
    for _ in range(warmup):
        fn()
    _sync_cuda()

    times = []
    for _ in range(n_iters):
        _sync_cuda()
        t0 = time.perf_counter()
        fn()
        _sync_cuda()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p10_ms": float(np.percentile(times, 10)),
        "p90_ms": float(np.percentile(times, 90)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "n_iters": n_iters,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Benchmark phases
# ═════════════════════════════════════════════════════════════════════════════


def benchmark_data_loading(
    cfg: DictConfig, n_batches: int = 100,
) -> dict:
    """
    Phase 1: Profile real data loading from memmap.

    Measures: disk read → subsample → augment → tensor conversion.
    Does NOT touch the GPU.
    """
    from oceanpath.data.dataset import MmapDataset
    from torch.utils.data import DataLoader

    t = cfg.training

    # Build label map from CSV
    try:
        import pandas as pd
        csv_df = pd.read_csv(cfg.data.csv_path)
        filename_col = cfg.data.filename_column
        label_col = cfg.data.label_columns[0] if cfg.data.label_columns else "label"
        label_map = {
            Path(str(row[filename_col])).stem: int(row[label_col])
            for _, row in csv_df.iterrows()
        }
    except Exception as e:
        logger.warning(f"Cannot load labels for data benchmark: {e}")
        return {"status": "skipped", "reason": str(e)}

    dataset = MmapDataset(
        mmap_dir=str(cfg.data.mmap_dir),
        slide_ids=list(label_map.keys()),
        labels=label_map,
        max_instances=t.get("max_instances") or None,  # None = no cap (real data)
        is_train=True,
        return_coords=False,
    )

    if len(dataset) == 0:
        return {"status": "skipped", "reason": "empty dataset"}

    loader = DataLoader(
        dataset,
        batch_size=t.get("batch_size", 1),
        shuffle=True,
        num_workers=min(cfg.platform.num_workers, 4),
        pin_memory=True,
        drop_last=True,
    )

    times = []
    loader_iter = iter(loader)
    for _ in range(min(n_batches, len(loader))):
        t0 = time.perf_counter()
        try:
            _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            _ = next(loader_iter)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    if not times:
        return {"status": "skipped", "reason": "no batches loaded"}

    times = np.array(times)
    return {
        "status": "ok",
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p90_ms": float(np.percentile(times, 90)),
        "std_ms": float(np.std(times)),
        "n_batches": len(times),
        "slides_per_sec": float(1000.0 / np.mean(times)) if np.mean(times) > 0 else 0,
        "dataset_size": len(dataset),
        "num_workers": min(cfg.platform.num_workers, 4),
    }


def benchmark_forward(
    cfg: DictConfig, n_iters: int = 100, device: str = "cuda",
) -> dict:
    """Phase 2: Profile forward pass only (synthetic data)."""
    model, in_dim, max_inst = _build_model(cfg, device)
    model.eval()

    x = torch.randn(1, max_inst, in_dim, device=device)
    mask = torch.ones(1, max_inst, dtype=torch.float32, device=device)

    def _forward():
        with torch.no_grad():
            model(x, mask=mask, return_attention=False)

    result = _time_fn(_forward, n_iters)
    result["slides_per_sec"] = float(1000.0 / result["mean_ms"]) if result["mean_ms"] > 0 else 0
    result["max_instances"] = max_inst
    result["device"] = device

    del model, x, mask
    _cleanup()
    return result


def benchmark_train_step(
    cfg: DictConfig, n_iters: int = 100, device: str = "cuda",
) -> dict:
    """Phase 3: Profile forward + backward + optimizer step (synthetic data)."""
    model, in_dim, max_inst = _build_model(cfg, device)
    model.train()

    t = cfg.training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t.get("lr", 1e-4),
        weight_decay=t.get("weight_decay", 5e-3),
    )

    num_classes = cfg.get("num_classes", 2)
    loss_fn = torch.nn.CrossEntropyLoss()

    use_amp = "16" in str(cfg.platform.get("precision", "32"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device != "cpu" else None

    x = torch.randn(1, max_inst, in_dim, device=device)
    mask = torch.ones(1, max_inst, dtype=torch.float32, device=device)
    labels = torch.randint(0, num_classes, (1,), device=device)

    def _train_step():
        optimizer.zero_grad(set_to_none=True)
        if use_amp and device != "cpu":
            with torch.amp.autocast("cuda"):
                output = model(x, mask=mask, return_attention=False)
                loss = loss_fn(output.logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(x, mask=mask, return_attention=False)
            loss = loss_fn(output.logits, labels)
            loss.backward()
            optimizer.step()

    result = _time_fn(_train_step, n_iters)
    result["slides_per_sec"] = float(1000.0 / result["mean_ms"]) if result["mean_ms"] > 0 else 0
    result["max_instances"] = max_inst
    result["device"] = device
    result["amp"] = use_amp

    del model, optimizer, x, mask, labels
    _cleanup()
    return result


def benchmark_inference(
    cfg: DictConfig, n_iters: int = 100, device: str = "cuda",
) -> dict:
    """Phase 4: Profile inference with attention (synthetic data)."""
    model, in_dim, max_inst = _build_model(cfg, device)
    model.eval()

    x = torch.randn(1, max_inst, in_dim, device=device)
    mask = torch.ones(1, max_inst, dtype=torch.float32, device=device)

    def _infer():
        with torch.no_grad():
            model(x, mask=mask, return_attention=True)

    result = _time_fn(_infer, n_iters)
    result["slides_per_sec"] = float(1000.0 / result["mean_ms"]) if result["mean_ms"] > 0 else 0
    result["max_instances"] = max_inst
    result["device"] = device

    del model, x, mask
    _cleanup()
    return result


def benchmark_bag_sizes(
    cfg: DictConfig,
    sizes: list[int] | None = None,
    n_iters: int = 50,
    device: str = "cuda",
) -> list[dict]:
    """
    Phase 5: Profile forward pass at different bag sizes.

    Shows how throughput scales with n_patches — critical for choosing
    max_instances.
    """
    if sizes is None:
        sizes = [100, 500, 1000, 2000, 4000, 8000]

    model, in_dim, _ = _build_model(cfg, device)
    model.eval()

    results = []
    for n_patches in sizes:
        x = torch.randn(1, n_patches, in_dim, device=device)
        mask = torch.ones(1, n_patches, dtype=torch.float32, device=device)

        def _fwd(x=x, mask=mask):
            with torch.no_grad():
                model(x, mask=mask, return_attention=False)

        try:
            r = _time_fn(_fwd, n_iters, warmup=3)
            r["n_patches"] = n_patches
            r["slides_per_sec"] = float(1000.0 / r["mean_ms"]) if r["mean_ms"] > 0 else 0
            results.append(r)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM at n_patches={n_patches}")
                results.append({"n_patches": n_patches, "status": "OOM"})
                _cleanup()
            else:
                raise

    del model
    _cleanup()
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Bottleneck identification
# ═════════════════════════════════════════════════════════════════════════════


def identify_bottleneck(results: dict) -> dict:
    """
    Analyze benchmark results to identify the training bottleneck.

    Compares data loading time vs forward vs backward+optimizer.
    """
    data_ms = results.get("data_loading", {}).get("mean_ms", 0)
    forward_ms = results.get("forward", {}).get("mean_ms", 0)
    train_ms = results.get("train_step", {}).get("mean_ms", 0)

    backward_optim_ms = max(0, train_ms - forward_ms) if train_ms > 0 else 0

    analysis = {
        "data_loading_ms": round(data_ms, 1),
        "forward_ms": round(forward_ms, 1),
        "backward_optim_ms": round(backward_optim_ms, 1),
        "train_step_ms": round(train_ms, 1),
    }

    if data_ms == 0 and train_ms == 0:
        analysis["bottleneck"] = "unknown"
        analysis["advice"] = "Benchmark did not produce valid timings"
        return analysis

    components = {
        "data_loading": data_ms,
        "forward_pass": forward_ms,
        "backward_optimizer": backward_optim_ms,
    }

    bottleneck = max(components, key=components.get)
    analysis["bottleneck"] = bottleneck

    advice_map = {
        "data_loading": (
            "Data loading is the bottleneck. Consider: "
            "increase num_workers, enable LRU cache (cache_size_mb), "
            "move memmap to SSD, or reduce max_instances."
        ),
        "forward_pass": (
            "Forward pass is the bottleneck. Consider: "
            "enable AMP (precision: 16-mixed), reduce max_instances, "
            "use a simpler aggregator, or enable gradient_checkpointing."
        ),
        "backward_optimizer": (
            "Backward + optimizer is the bottleneck. Consider: "
            "enable AMP, use gradient accumulation, reduce model size, "
            "or enable gradient_checkpointing."
        ),
    }
    analysis["advice"] = advice_map.get(bottleneck, "")

    effective_ms = max(data_ms, train_ms) if data_ms > 0 else train_ms
    if effective_ms > 0:
        analysis["effective_slides_per_sec"] = round(1000.0 / effective_ms, 1)

    return analysis


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _build_model(cfg: DictConfig, device: str):
    """Build model from config (same as train.py)."""
    from oceanpath.models import build_classifier

    in_dim = cfg.encoder.feature_dim
    num_classes = cfg.get("num_classes", 2)

    # max_instances can be None (no cap) — use a realistic default for
    # synthetic benchmarking. 4000 is typical for GEJ/BLCA cohorts.
    max_inst = cfg.training.get("max_instances") or 4000

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = build_classifier(
        arch=cfg.model.arch,
        in_dim=in_dim,
        num_classes=num_classes,
        model_cfg=model_cfg,
    )
    model.to(device)
    return model, in_dim, max_inst


def _cleanup():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_gpu_info() -> dict:
    """Collect GPU info for the report."""
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "cuda_version": torch.version.cuda,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


@hydra.main(config_path="../configs", config_name="benchmark", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Stage 9: Throughput benchmarking."""

    level = logging.DEBUG if cfg.get("verbose", False) else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(message)s", force=True)

    b = cfg.benchmark
    n_iters = b.get("n_iters", 100)

    device = "cpu"
    if torch.cuda.is_available() and cfg.platform.get("num_gpus", 0) > 0:
        device = f"cuda:{cfg.platform.get('gpu', 0)}"

    logger.info("=" * 60)
    logger.info("  Stage 9: Throughput Benchmark")
    logger.info("=" * 60)
    effective_max_inst = cfg.training.get("max_instances") or 4000
    logger.info(f"Device:        {device}")
    logger.info(f"Architecture:  {cfg.model.arch}")
    logger.info(f"Encoder dim:   {cfg.encoder.feature_dim}")
    logger.info(f"max_instances: {cfg.training.get('max_instances')} (bench: {effective_max_inst})")
    logger.info(f"Iterations:    {n_iters}")

    output_dir = Path(b.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.dry_run:
        print(f"\n{'=' * 60}")
        print(f"  DRY RUN — benchmark.py")
        print(f"{'=' * 60}")
        print(f"  Device:      {device}")
        print(f"  Model:       {cfg.model.arch} (D={cfg.encoder.feature_dim})")
        print(f"  Iterations:  {n_iters}")
        print(f"  Output:      {output_dir}")
        print(f"{'=' * 60}\n")
        return

    results = {
        "config": {
            "arch": cfg.model.arch,
            "encoder": cfg.encoder.name,
            "feature_dim": cfg.encoder.feature_dim,
            "max_instances": cfg.training.get("max_instances"),
            "max_instances_bench": cfg.training.get("max_instances") or 4000,
            "precision": str(cfg.platform.get("precision", "32")),
            "device": device,
        },
        "gpu": _get_gpu_info(),
    }

    phases = [
        ("data_loading",  b.get("run_data_loading", True),
         lambda: benchmark_data_loading(cfg, n_batches=n_iters)),
        ("forward",       b.get("run_forward", True),
         lambda: benchmark_forward(cfg, n_iters=n_iters, device=device)),
        ("train_step",    b.get("run_train_step", True),
         lambda: benchmark_train_step(cfg, n_iters=n_iters, device=device)),
        ("inference",     b.get("run_inference", True),
         lambda: benchmark_inference(cfg, n_iters=n_iters, device=device)),
    ]

    for name, enabled, fn in phases:
        if enabled:
            logger.info(f"Phase: {name}...")
            results[name] = fn()
            r = results[name]
            if r.get("status") == "skipped":
                logger.info(f"  {name}: skipped ({r.get('reason', '?')})")
            elif "mean_ms" in r:
                logger.info(
                    f"  {name}: {r['mean_ms']:.1f}ms "
                    f"({r.get('slides_per_sec', 0):.0f} slides/s)"
                )

    # Bag size scaling
    if b.get("run_bag_sizes", True):
        logger.info("Phase: bag_sizes...")
        sizes = list(b.get("bag_sizes", [100, 500, 1000, 2000, 4000, 8000]))
        results["bag_sizes"] = benchmark_bag_sizes(
            cfg, sizes=sizes, n_iters=min(n_iters, 50), device=device,
        )
        for r in results["bag_sizes"]:
            if r.get("status") == "OOM":
                logger.info(f"  n_patches={r['n_patches']}: OOM")
            elif "mean_ms" in r:
                logger.info(
                    f"  n_patches={r['n_patches']}: {r['mean_ms']:.1f}ms "
                    f"({r.get('slides_per_sec', 0):.0f} slides/s)"
                )

    # Bottleneck analysis
    results["bottleneck"] = identify_bottleneck(results)

    # Save
    report_path = output_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Report → {report_path}")

    # Summary
    _print_summary(results)


def _print_summary(results: dict) -> None:
    """Print structured benchmark summary."""
    print(f"\n{'=' * 60}")
    print(f"  Benchmark Results")
    print(f"{'=' * 60}")

    cfg = results.get("config", {})
    print(f"  Model:  {cfg.get('arch', '?')} | D={cfg.get('feature_dim', '?')}")
    print(f"  Device: {cfg.get('device', '?')} | AMP: {cfg.get('precision', '?')}")

    gpu = results.get("gpu", {})
    if gpu.get("available"):
        print(f"  GPU:    {gpu.get('name')} ({gpu.get('memory_gb')} GB)")

    print(f"\n  {'Phase':<25s} {'Mean (ms)':>10s} {'Slides/s':>10s}")
    print(f"  {'─' * 47}")

    for phase in ["data_loading", "forward", "train_step", "inference"]:
        r = results.get(phase, {})
        if r.get("status") == "skipped":
            print(f"  {phase:<25s} {'skipped':>10s}")
        elif "mean_ms" in r:
            print(f"  {phase:<25s} {r['mean_ms']:>10.1f} {r.get('slides_per_sec', 0):>10.0f}")

    bn = results.get("bottleneck", {})
    if bn.get("bottleneck") and bn["bottleneck"] != "unknown":
        print(f"\n  Bottleneck: {bn['bottleneck']}")
        print(f"  Advice:     {bn.get('advice', '')}")

    bag = results.get("bag_sizes", [])
    if bag:
        print(f"\n  Bag size scaling:")
        for r in bag:
            if r.get("status") == "OOM":
                print(f"    {r['n_patches']:>6d} patches: OOM")
            elif "mean_ms" in r:
                print(
                    f"    {r['n_patches']:>6d} patches: "
                    f"{r['mean_ms']:>8.1f}ms "
                    f"({r.get('slides_per_sec', 0):>6.0f} slides/s)"
                )

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()