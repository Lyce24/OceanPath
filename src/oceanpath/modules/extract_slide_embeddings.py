#!/usr/bin/env python3
"""
Extract slide-level embeddings from patch-level OceanPath mmap features.

Why this script exists
======================
Your downstream linear probe should operate on ONE vector per slide, not on
raw patch bags. In this codebase, the patch-level mmap stores one slide as a
variable-length bag of patch features [N, D]. This script converts each slide
into a single slide embedding either by:

  1) mean-pooling raw patch features (baseline), or
  2) forwarding the patch bag through the fixed MIL aggregator and taking
     `out.slide_embedding`.

The script can write:
  - a flat `.npz` file: embeddings, slide_ids
  - an OceanPath-style embedding mmap directory, with one feature row per slide,
    so it plugs directly into `linear_probing_sklearn.py`

Recommended workflow
====================
1) Extract slide embeddings ONCE.
2) Reuse those cached slide embeddings for all downstream linear-probe runs.
3) Keep extraction separate from sklearn probing.

Examples
========
# Mean-pool baseline -> embedding mmap
python scripts/extract_slide_embeddings.py \
    --mode mean_pool \
    --mmap_dir mmap/tcga_nsclc_patch_features \
    --manifest_csv data/tcga_nsclc_manifest.csv \
    --output_path mmap/tcga_nsclc_meanpool_embeddings \
    --output_format mmap

# Pretrained aggregator -> embedding mmap
python scripts/extract_slide_embeddings.py \
    --mode aggregator \
    --mmap_dir mmap/tcga_nsclc_patch_features \
    --manifest_csv data/tcga_nsclc_manifest.csv \
    --ckpt outputs/pretrain/vicreg/model.ckpt \
    --output_path mmap/tcga_nsclc_vicreg_embeddings \
    --output_format mmap \
    --batch_size 1 \
    --num_workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger("extract_slide_embeddings")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract slide embeddings from patch-level mmap features."
    )

    p.add_argument(
        "--mode", choices=["mean_pool", "max_pool", "mean_max_pool", "aggregator"], required=True
    )
    p.add_argument("--mmap_dir", type=str, required=True, help="Input PATCH-LEVEL mmap directory.")
    p.add_argument(
        "--manifest_csv",
        type=str,
        required=True,
        help="Manifest CSV used to define which slides to extract.",
    )
    p.add_argument(
        "--output_path", type=str, required=True, help="Output .npz path or output mmap directory."
    )
    p.add_argument("--output_format", choices=["npz", "mmap"], default="mmap")

    p.add_argument(
        "--slide_id_column", type=str, default=None, help="Explicit slide_id column in manifest."
    )
    p.add_argument(
        "--filename_column",
        type=str,
        default="filename",
        help="Fallback filename column if slide_id_column is absent.",
    )

    p.add_argument(
        "--arch",
        type=str,
        default="mamba2mil",
        help="Aggregator architecture name (e.g. mamba2mil, abmil, transmil).",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint for aggregator mode. Omit for random-init baseline.",
    )
    p.add_argument("--device", type=str, default=None, help="cuda / cuda:0 / cpu. Default: auto.")
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for embedding extraction. Use 1 if bags are large.",
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help=(
            "Optional eval-time cap on patches per slide. Default None = use full bag. "
            "Only set this if you hit OOM; lowering batch_size is preferred."
        ),
    )
    p.add_argument(
        "--sampling_mode",
        type=str,
        default="contiguous",
        choices=("contiguous", "random", "spatial_stratified"),
        help=(
            "How to subsample when bag size > max_instances. "
            "contiguous: first-N (matches pretraining contiguous pre-cap). "
            "random: uniform-without-replacement, deterministic per slide. "
            "spatial_stratified: grid-stratified across bbox cells (best coverage)."
        ),
    )
    p.add_argument(
        "--sampling_seed",
        type=int,
        default=42,
        help="Base seed for per-slide subsampling determinism.",
    )

    # Aggregator hyperparameters: defaults match the proposal
    p.add_argument("--in_dim", type=int, default=1536)
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--d_state", type=int, default=64)
    p.add_argument("--d_conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--headdim", type=int, default=64)
    p.add_argument("--attn_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--use_hilbert", action="store_true", default=True)
    p.add_argument("--no_use_hilbert", dest="use_hilbert", action="store_false")
    p.add_argument("--hilbert_order", type=int, default=16)
    p.add_argument("--use_cobra_inference", action="store_true", default=False)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)

    # Output mmap precision / chunking
    p.add_argument(
        "--feat_precision", type=int, choices=[16, 32], default=32, help="Embedding mmap precision."
    )
    p.add_argument("--coord_dtype", type=str, default="int32")
    p.add_argument("--max_chunk_gb", type=float, default=9.5)

    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def setup_logging(verbose: bool) -> None:
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)


_KNOWN_EXTENSIONS = frozenset(
    {
        # Feature / data files
        ".h5",
        ".pt",
        ".npy",
        ".npz",
        ".parquet",
        # WSI formats
        ".svs",
        ".tiff",
        ".tif",
        ".ndpi",
        ".mrxs",
        ".vsi",
        ".scn",
        ".sdpc",
        ".bif",
    }
)


def _normalize_slide_id(x: str) -> str:
    """Normalize a slide identifier.

    - Converts backslashes to forward slashes
    - Strips known file extensions (.h5, .svs, .tiff, etc.)
    - Does NOT strip arbitrary suffixes (e.g. TCGA UUIDs like .8923A151-...)
    """
    x = str(x).strip().replace("\\", "/")
    for suffix in _KNOWN_EXTENSIONS:
        if x.lower().endswith(suffix):
            x = x[: -len(suffix)]
            break
    return x


def _resolve_slide_ids(
    df: pd.DataFrame, slide_id_column: str | None, filename_column: str
) -> pd.Series:
    if slide_id_column is not None:
        if slide_id_column not in df.columns:
            raise ValueError(
                f"slide_id_column='{slide_id_column}' not found. Columns: {list(df.columns)}"
            )
        return df[slide_id_column].astype(str).map(_normalize_slide_id)

    if filename_column not in df.columns:
        raise ValueError(
            f"Neither slide_id_column nor filename_column='{filename_column}' found. Columns: {list(df.columns)}"
        )

    # Normalize separators and strip known file extensions only.
    # Preserves directory structure and non-extension dots (e.g. TCGA UUIDs).
    return df[filename_column].astype(str).map(_normalize_slide_id)


def load_slide_ids(
    manifest_csv: str | Path, slide_id_column: str | None, filename_column: str
) -> list[str]:
    df = pd.read_csv(manifest_csv)
    slide_ids = _resolve_slide_ids(df, slide_id_column, filename_column)
    if slide_ids.duplicated().any():
        dup = slide_ids[slide_ids.duplicated(keep=False)].head(10).tolist()
        raise ValueError(f"Manifest has duplicate slide_ids. First examples: {dup}")
    return slide_ids.tolist()


@torch.inference_mode()
def extract_mean_pool(loader: DataLoader, device: torch.device) -> tuple[np.ndarray, list[str]]:
    all_embeddings: list[np.ndarray] = []
    all_slide_ids: list[str] = []

    for batch in loader:
        feats = batch["features"].to(device, non_blocking=True)  # [B, N, D]
        mask = batch["mask"].to(device, non_blocking=True)  # [B, N]
        summed = (feats * mask.unsqueeze(-1)).sum(dim=1)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = summed / counts
        all_embeddings.append(pooled.cpu().numpy().astype(np.float32, copy=False))
        all_slide_ids.extend(batch["slide_ids"])

    if not all_embeddings:
        raise RuntimeError(
            "No slides were extracted. The DataLoader produced zero batches. "
            "This usually means no manifest slide_ids matched the mmap index. "
            "Check that data.mmap_dir, data.csv_path, and filename_column are consistent."
        )

    return np.concatenate(all_embeddings, axis=0), all_slide_ids


@torch.inference_mode()
def extract_pooling(
    loader: DataLoader, device: torch.device, mode: str
) -> tuple[np.ndarray, list[str]]:
    """Extract slide embeddings via mean_pool, max_pool, or mean_max_pool."""
    all_embeddings: list[np.ndarray] = []
    all_slide_ids: list[str] = []

    for batch in loader:
        feats = batch["features"].to(device, non_blocking=True)  # [B, N, D]
        mask = batch["mask"].to(device, non_blocking=True).bool()  # [B, N]

        if mode == "mean_pool":
            summed = (feats * mask.unsqueeze(-1)).sum(dim=1)
            counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = summed / counts

        elif mode == "max_pool":
            masked_feats = feats.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            pooled = masked_feats.max(dim=1).values
            pooled = torch.nan_to_num(pooled, neginf=0.0)

        elif mode == "mean_max_pool":
            summed = (feats * mask.unsqueeze(-1)).sum(dim=1)
            counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
            mean_pooled = summed / counts

            masked_feats = feats.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            max_pooled = masked_feats.max(dim=1).values
            max_pooled = torch.nan_to_num(max_pooled, neginf=0.0)

            pooled = torch.cat([mean_pooled, max_pooled], dim=-1)

        else:
            raise ValueError(f"Unknown pooling mode: {mode}")

        all_embeddings.append(pooled.cpu().numpy().astype(np.float32))
        all_slide_ids.extend(batch["slide_ids"])

    if not all_embeddings:
        raise RuntimeError(
            "No slides were extracted. The DataLoader produced zero batches. "
            "This usually means no manifest slide_ids matched the mmap index. "
            "Check that data.mmap_dir, data.csv_path, and filename_column are consistent."
        )

    return np.concatenate(all_embeddings, axis=0), all_slide_ids


def _extract_slide_embedding_from_output(out: Any) -> torch.Tensor:
    if hasattr(out, "slide_embedding"):
        return out.slide_embedding
    if isinstance(out, dict) and "slide_embedding" in out:
        return out["slide_embedding"]
    raise ValueError(
        "Model output does not contain `slide_embedding`. "
        "Check your aggregator forward() return structure."
    )


def _clean_checkpoint_state(raw_state: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "aggregator_state_dict" in raw_state and isinstance(
        raw_state["aggregator_state_dict"], dict
    ):
        return raw_state["aggregator_state_dict"]

    if "state_dict" in raw_state and isinstance(raw_state["state_dict"], dict):
        raw_state = raw_state["state_dict"]

    skip_prefixes = (
        "projector.",
        "predictor.",
        "teacher.",
        "classification_head.",
        "classifier.",
        "loss_fn.",
        "target_network.",
        "_online_for_ema.",
    )
    prefix_candidates = (
        "trunk.",
        "online_encoder.",
        "model.",
        "aggregator.",
        "encoder.",
        "backbone.",
    )

    cleaned: dict[str, torch.Tensor] = {}
    for k, v in raw_state.items():
        if any(k.startswith(prefix) for prefix in skip_prefixes):
            continue
        new_k = k
        for prefix in prefix_candidates:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix) :]
        cleaned[new_k] = v
    return cleaned


def build_aggregator_from_args(args: argparse.Namespace) -> torch.nn.Module:
    from oceanpath.models import build_aggregator

    # Pass as model_cfg dict so build_aggregator introspects the constructor
    # and filters to only valid params for the given architecture.
    model_cfg = {
        "embed_dim": args.embed_dim,
        "num_layers": args.num_layers,
        "d_state": args.d_state,
        "d_conv": args.d_conv,
        "expand": args.expand,
        "headdim": args.headdim,
        "attn_dim": args.attn_dim,
        "dropout": args.dropout,
        "use_hilbert": args.use_hilbert,
        "hilbert_order": args.hilbert_order,
        "use_cobra_inference": args.use_cobra_inference,
        "gradient_checkpointing": args.gradient_checkpointing,
    }

    model = build_aggregator(arch=args.arch, in_dim=args.in_dim, model_cfg=model_cfg)

    if args.ckpt:
        raw_state = torch.load(args.ckpt, map_location="cpu")
        state = _clean_checkpoint_state(raw_state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning(
                "Missing checkpoint keys (first 10): %s | total=%d", missing[:10], len(missing)
            )
        if unexpected:
            logger.warning(
                "Unexpected checkpoint keys (first 10): %s | total=%d",
                unexpected[:10],
                len(unexpected),
            )

    return model


@torch.inference_mode()
def extract_aggregator(
    loader: DataLoader, model: torch.nn.Module, device: torch.device
) -> tuple[np.ndarray, list[str]]:
    model.eval().to(device)
    all_embeddings: list[np.ndarray] = []
    all_slide_ids: list[str] = []

    for batch in loader:
        kwargs: dict[str, Any] = {
            "features": batch["features"].to(device, non_blocking=True),
            "mask": batch["mask"].to(device, non_blocking=True),
        }
        if "coords" in batch:
            kwargs["coords"] = batch["coords"].to(device, non_blocking=True)

        try:
            out = model(**kwargs)
        except TypeError:
            # Some forward() methods do not accept named args.
            if "coords" in kwargs:
                out = model(kwargs["features"], mask=kwargs["mask"], coords=kwargs["coords"])
            else:
                out = model(kwargs["features"], mask=kwargs["mask"])

        emb = _extract_slide_embedding_from_output(out)
        all_embeddings.append(emb.detach().cpu().numpy().astype(np.float32, copy=False))
        all_slide_ids.extend(batch["slide_ids"])

    if not all_embeddings:
        raise RuntimeError(
            "No slides were extracted. The DataLoader produced zero batches. "
            "This usually means no manifest slide_ids matched the mmap index. "
            "Check that data.mmap_dir, data.csv_path, and filename_column are consistent."
        )

    return np.concatenate(all_embeddings, axis=0), all_slide_ids


@dataclass
class ChunkedBinaryWriter:
    output_dir: Path
    prefix: str
    max_bytes: int
    current_chunk_id: int = 0
    current_file: Any = None
    current_bytes: int = 0
    chunk_files: list[str] = None

    def __post_init__(self):
        self.chunk_files = []
        self._open_new_chunk()

    def _chunk_path(self, chunk_id: int) -> Path:
        return self.output_dir / f"{self.prefix}_{chunk_id:03d}.bin"

    def _open_new_chunk(self) -> None:
        if self.current_file is not None:
            self.current_file.flush()
            self.current_file.close()
        path = self._chunk_path(self.current_chunk_id)
        self.current_file = open(path, "wb")  # noqa: SIM115  # closed in close()/_open_new_chunk
        self.chunk_files.append(str(path))
        self.current_bytes = 0

    def write(self, arr: np.ndarray) -> tuple[int, int]:
        data_bytes = arr.nbytes
        if self.current_bytes > 0 and self.current_bytes + data_bytes > self.max_bytes:
            self.current_chunk_id += 1
            self._open_new_chunk()
        chunk_id = self.current_chunk_id
        offset = self.current_bytes
        arr.tofile(self.current_file)
        self.current_bytes += data_bytes
        return chunk_id, offset

    def close(self) -> None:
        if self.current_file is not None:
            self.current_file.flush()
            self.current_file.close()
            self.current_file = None

    @property
    def num_chunks(self) -> int:
        return self.current_chunk_id + 1


def compute_source_hash(
    slide_ids: list[str], embeddings: np.ndarray, args: argparse.Namespace
) -> str:
    h = hashlib.sha256()
    for sid in slide_ids:
        h.update(sid.encode())
    h.update(str(embeddings.shape).encode())
    h.update(f"mode={args.mode}".encode())
    h.update(f"ckpt={args.ckpt}".encode())
    h.update(f"feat_precision={args.feat_precision}".encode())
    return h.hexdigest()[:20]


def write_embedding_mmap(
    output_dir: str | Path,
    slide_ids: list[str],
    embeddings: np.ndarray,
    args: argparse.Namespace,
) -> None:
    from oceanpath.data.mmap_builder import MMAP_SCHEMA_VERSION

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for p in output_dir.glob("features_*.bin"):
        p.unlink()
    for p in output_dir.glob("coords_*.bin"):
        p.unlink()

    feat_dtype = np.float16 if args.feat_precision == 16 else np.float32
    coord_dtype = np.dtype(args.coord_dtype)
    max_chunk_bytes = int(args.max_chunk_gb * 1024**3)

    feat_writer = ChunkedBinaryWriter(output_dir, "features", max_chunk_bytes)
    coord_writer = ChunkedBinaryWriter(output_dir, "coords", max_chunk_bytes)

    n_slides, feat_dim = embeddings.shape
    feat_chunk_ids = np.full(n_slides, -1, dtype=np.int16)
    feat_offsets = np.full(n_slides, -1, dtype=np.int64)
    coord_chunk_ids = np.full(n_slides, -1, dtype=np.int16)
    coord_offsets = np.full(n_slides, -1, dtype=np.int64)
    lengths = np.ones(n_slides, dtype=np.int32)
    raw_lengths = np.ones(n_slides, dtype=np.int32)

    dummy_coord = np.zeros((1, 2), dtype=coord_dtype)

    for i, _sid in enumerate(slide_ids):
        feat_arr = np.ascontiguousarray(embeddings[i : i + 1].astype(feat_dtype, copy=False))
        f_chunk, f_offset = feat_writer.write(feat_arr)
        c_chunk, c_offset = coord_writer.write(dummy_coord)
        feat_chunk_ids[i] = f_chunk
        feat_offsets[i] = f_offset
        coord_chunk_ids[i] = c_chunk
        coord_offsets[i] = c_offset

    feat_writer.close()
    coord_writer.close()

    np.savez(
        str(output_dir / "index_arrays.npz"),
        schema_version=np.int32(MMAP_SCHEMA_VERSION),
        slide_ids=np.array(slide_ids, dtype=object),
        lengths=lengths,
        raw_lengths=raw_lengths,
        feat_chunk_ids=feat_chunk_ids,
        feat_offsets=feat_offsets,
        coord_chunk_ids=coord_chunk_ids,
        coord_offsets=coord_offsets,
        feat_dim=np.int32(feat_dim),
        feat_dtype=np.array([np.dtype(feat_dtype).name], dtype=object),
        coord_dtype=np.array([coord_dtype.name], dtype=object),
        coord_dim=np.int32(2),
        bytes_per_feat=np.int32(np.dtype(feat_dtype).itemsize),
        num_feat_chunks=np.int32(feat_writer.num_chunks),
        num_coord_chunks=np.int32(coord_writer.num_chunks),
        total_patches=np.int64(n_slides),
        n_slides=np.int32(n_slides),
    )
    (output_dir / ".schema_version").write_text(str(MMAP_SCHEMA_VERSION))
    (output_dir / ".source_hash").write_text(compute_source_hash(slide_ids, embeddings, args))

    meta = {
        "mode": args.mode,
        "input_mmap_dir": args.mmap_dir,
        "ckpt": args.ckpt,
        "n_slides": int(n_slides),
        "feat_dim": int(feat_dim),
        "feat_precision": int(args.feat_precision),
        "note": "One embedding row per slide; dummy coords=[0,0] written for mmap schema compatibility.",
    }
    with open(output_dir / "embedding_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def build_loader(
    mmap_dir: str,
    slide_ids: list[str],
    mode: str,
    batch_size: int,
    num_workers: int,
    max_instances: int | None,
    sampling_mode: str = "contiguous",
    sampling_seed: int = 42,
) -> DataLoader:
    from oceanpath.data.datamodule import SimpleMILCollator
    from oceanpath.data.dataset import MmapDataset

    dummy_labels = dict.fromkeys(slide_ids, 0)
    # spatial_stratified needs coords even when mode != "aggregator".
    need_coords = (mode == "aggregator") or (sampling_mode == "spatial_stratified")
    dataset = MmapDataset(
        mmap_dir=mmap_dir,
        slide_ids=slide_ids,
        labels=dummy_labels,
        max_instances=None,
        is_train=False,
        cache_size_mb=0,
        return_coords=need_coords,
        force_float32=True,
    )

    if len(dataset) == 0:
        # Load mmap index to show diagnostic info
        idx = np.load(str(Path(mmap_dir) / "index_arrays.npz"), allow_pickle=True)
        mmap_ids = idx["slide_ids"].tolist()
        raise RuntimeError(
            f"No slide_ids matched between manifest ({len(slide_ids)} IDs) "
            f"and mmap index ({len(mmap_ids)} IDs). Cannot extract embeddings.\n"
            f"  Manifest examples: {slide_ids[:3]}\n"
            f"  Mmap examples:     {mmap_ids[:3]}\n"
            f"Check that filename_column / slide_id_column in the manifest "
            f"produces IDs matching the mmap's slide_id convention "
            f"(relative path from h5_dir, without .h5 extension)."
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=SimpleMILCollator(
            max_instances=max_instances,
            sampling_mode=sampling_mode,
            seed=sampling_seed,
        ),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Using device: %s", device)

    slide_ids = load_slide_ids(args.manifest_csv, args.slide_id_column, args.filename_column)
    logger.info("Resolved %d slide IDs from %s", len(slide_ids), args.manifest_csv)

    loader = build_loader(
        mmap_dir=args.mmap_dir,
        slide_ids=slide_ids,
        mode=args.mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_instances=args.max_instances,
        sampling_mode=getattr(args, "sampling_mode", "contiguous"),
        sampling_seed=int(getattr(args, "sampling_seed", 42)),
    )

    logger.info(
        "Extraction config | mode=%s | batch_size=%d | max_instances=%s | sampling_mode=%s",
        args.mode,
        args.batch_size,
        args.max_instances,
        getattr(args, "sampling_mode", "contiguous"),
    )

    if args.mode in {"mean_pool", "max_pool", "mean_max_pool"}:
        embeddings, kept_slide_ids = extract_pooling(loader, device, args.mode)
    else:
        torch.manual_seed(42)
        model = build_aggregator_from_args(args)
        embeddings, kept_slide_ids = extract_aggregator(loader, model, device)

    logger.info(
        "Extracted %d slide embeddings with dim=%d", embeddings.shape[0], embeddings.shape[1]
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.output_format == "npz":
        np.savez_compressed(
            str(output_path),
            embeddings=embeddings,
            slide_ids=np.array(kept_slide_ids, dtype=object),
        )
        logger.info("Saved npz embeddings -> %s", output_path)
    else:
        write_embedding_mmap(output_path, kept_slide_ids, embeddings, args)
        logger.info("Saved embedding mmap -> %s", output_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
