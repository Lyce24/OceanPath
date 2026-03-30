"""Check coordinate scale for ALiBi config."""

import sys
from pathlib import Path

import numpy as np


def check_mmap(mmap_dir: str):
    mmap_dir = Path(mmap_dir)
    idx = np.load(str(mmap_dir / "index_arrays.npz"), allow_pickle=True)

    slide_ids = idx["slide_ids"]
    lengths = idx["lengths"]
    coord_offsets = idx["coord_offsets"]
    coord_chunk_ids = idx["coord_chunk_ids"]
    coord_dtype = str(idx["coord_dtype"][0])
    coord_dim = int(idx["coord_dim"])

    # Load coord chunk files
    coord_files = sorted(mmap_dir.glob("coords_*.bin"))
    coord_mms = {
        int(f.stem.split("_")[1]): np.memmap(str(f), dtype=coord_dtype, mode="r")
        for f in coord_files
    }

    bytes_per_coord = np.dtype(coord_dtype).itemsize * coord_dim

    print(f"{len(slide_ids)} slides, coord_dtype={coord_dtype}, coord_dim={coord_dim}\n")

    # Sample 3 slides: first, middle, last
    for i in [0, len(slide_ids) // 2, len(slide_ids) - 1]:
        sid = slide_ids[i]
        n = lengths[i]
        chunk = coord_chunk_ids[i]
        byte_off = coord_offsets[i]
        patch_off = byte_off // bytes_per_coord

        mm = coord_mms[chunk]
        c = mm[patch_off * coord_dim : (patch_off + n) * coord_dim].reshape(n, coord_dim)

        sample = c[: min(n, 200)].astype(np.float64)
        diff = sample[:, None, :] - sample[None, :, :]
        dist = np.sqrt((diff**2).sum(-1))
        d_pos = dist[dist > 0.5]
        min_d = d_pos.min() if d_pos.size > 0 else 0.0

        print(f"  {sid} ({n} patches)")
        print(f"    x=[{c[:, 0].min()}, {c[:, 0].max()}]  y=[{c[:, 1].min()}, {c[:, 1].max()}]")
        print(f"    first 5: {c[:5].tolist()}")
        print(f"    min adjacent dist: {min_d:.1f}")
        print()

    # Recommendation from first slide
    n0 = lengths[0]
    mm0 = coord_mms[coord_chunk_ids[0]]
    p0 = coord_offsets[0] // bytes_per_coord
    c0 = (
        mm0[p0 * coord_dim : (p0 + min(n0, 200)) * coord_dim]
        .reshape(-1, coord_dim)
        .astype(np.float64)
    )
    diff0 = c0[:, None, :] - c0[None, :, :]
    dist0 = np.sqrt((diff0**2).sum(-1))
    d0 = dist0[dist0 > 0.5]
    min_dist = d0.min() if d0.size > 0 else 1.0

    print("=" * 50)
    if min_dist > 100:
        print(f"Pixel offsets (adjacent ≈ {min_dist:.0f})")
        print(f"→ coord_scale: {min_dist:.1f}")
    elif 0.5 < min_dist < 2.0:
        print(f"Already grid units (adjacent ≈ {min_dist:.1f})")
        print("→ coord_scale: null")
    else:
        print(f"Adjacent dist = {min_dist:.1f}")
        print(f"→ coord_scale: {min_dist:.1f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_coord_scale.py /path/to/mmap_dir")
        sys.exit(1)
    check_mmap(sys.argv[1])
