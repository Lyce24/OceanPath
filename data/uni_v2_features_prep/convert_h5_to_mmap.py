import h5py
import numpy as np
import pandas as pd
import argparse
import os
import gc
from tqdm import tqdm

def process_single_slide(slide_id, file_path, feat_key, coord_key, precision, max_instances):
    """Reads and processes a single slide (Features + Coords)."""
    try:
        if not os.path.isfile(file_path):
            return None, None, f"File not found: {file_path}"
        
        with h5py.File(file_path, "r") as f:
            if feat_key not in f:
                return None, None, f"Key '{feat_key}' not found"
            if coord_key not in f:
                return None, None, f"Key '{coord_key}' not found"
            
            # Read Data
            x = f[feat_key][:]
            coords = f[coord_key][:]
        
        # --- Handle Dimensions (Features) ---
        # Fix [1, N, D] -> [N, D]
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 2:
            return None, None, f"Feat expected 2D, got {x.shape}"
            
        # --- Handle Dimensions (Coords) ---
        # Fix [1, N, 2] -> [N, 2]
        if coords.ndim == 3 and coords.shape[0] == 1:
            coords = coords[0]
        if coords.ndim != 2:
            return None, None, f"Coords expected 2D, got {coords.shape}"

        # --- Consistency Check ---
        if x.shape[0] != coords.shape[0]:
            return None, None, f"Mismatch: Feat {x.shape[0]} vs Coords {coords.shape[0]}"

        # --- Subsampling (Synced) ---
        # We must apply the SAME random indices to both arrays
        if max_instances and x.shape[0] > max_instances:
            idx_sample = np.random.permutation(x.shape[0])[:max_instances]
            x = x[idx_sample]
            coords = coords[idx_sample]
        
        # --- Type Conversion ---
        # Features -> float16/32
        if precision == 16:
            x = x.astype(np.float16, copy=False)
        else:
            x = x.astype(np.float32, copy=False)

        # Coords -> int32 (Standard for pixel coords, saves space vs int64)
        coords = coords.astype(np.int32, copy=False)
        
        # --- Memory Layout ---
        x = np.ascontiguousarray(x)
        coords = np.ascontiguousarray(coords)
        
        return x, coords, None

    except Exception as e:
        return None, None, repr(e)

def main():
    parser = argparse.ArgumentParser(description="Safe Sequential H5 to Memmap (Feat + Coords)")
    parser.add_argument("--csv_path", type=str, default="./data/data_prep/ssl_tcga_cptac.csv")
    parser.add_argument("--output_dir", type=str, default="./src/univ2_mmap_v2/")
    parser.add_argument("--path_col", type=str, default="path")
    parser.add_argument("--id_col", type=str, default="slide_id")
    
    # Keys in H5 file
    parser.add_argument("--h5_feat_key", type=str, default="features")
    parser.add_argument("--h5_coord_key", type=str, default="coords")
    
    parser.add_argument("--precision", type=int, default=16, choices=[16, 32])
    parser.add_argument("--max_instances", type=int, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load CSV
    df = pd.read_csv(args.csv_path)
    paths = df[args.path_col].astype(str).tolist()
    slide_ids = df[args.id_col].astype(str).tolist()
    n = len(df)
    
    print(f"Processing {n} slides (Features + Coords)...")
    
    # Initialize index arrays
    offsets = np.full(n, -1, dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int32)
    slide_ids_out = np.asarray(slide_ids, dtype=object)
    
    # Output paths
    feat_bin = os.path.join(args.output_dir, "features.bin")
    coord_bin = os.path.join(args.output_dir, "coords.bin")  # <--- NEW FILE
    index_npz = os.path.join(args.output_dir, "index_arrays.npz")
    
    feat_dim = None
    total_rows = 0
    n_success = 0
    n_err = 0
    errors = []
    
    # Open BOTH files for writing
    with open(feat_bin, "wb") as f_feat, open(coord_bin, "wb") as f_coord:
        
        for i in tqdm(range(n), desc="Converting"):
            sid = slide_ids[i]
            path = paths[i]
            
            # 1. Process
            x, coords, err = process_single_slide(
                sid, path, args.h5_feat_key, args.h5_coord_key, 
                args.precision, args.max_instances
            )
            
            # 2. Handle Errors
            if err is not None:
                n_err += 1
                errors.append((sid, err))
                continue
            
            # 3. Check Dimensions
            N, D = x.shape
            if feat_dim is None:
                feat_dim = int(D)
            elif int(D) != feat_dim:
                n_err += 1
                errors.append((sid, f"Dim mismatch: expected {feat_dim}, got {D}"))
                del x, coords
                continue
            
            # 4. Write to Disk
            offsets[i] = total_rows
            lengths[i] = int(N)
            
            x.tofile(f_feat)      # Write features
            coords.tofile(f_coord) # Write coords
            
            total_rows += int(N)
            n_success += 1
            
            # 5. FORCE FREE RAM
            del x
            del coords
            
            if i % 100 == 0:
                gc.collect()

    # Save Index
    dtype_str = "float16" if args.precision == 16 else "float32"
    np.savez(
        index_npz,
        slide_ids=slide_ids_out,
        offsets=offsets,
        lengths=lengths,
        feat_dim=np.int32(feat_dim if feat_dim else 0),
        dtype=np.asarray([dtype_str], dtype=object),
        coord_dtype=np.asarray(["int32"], dtype=object), # Record that coords are int32
        total_patches=np.int64(total_rows),
    )
    
    # Verify file sizes
    expected_feat_bytes = total_rows * feat_dim * (2 if args.precision == 16 else 4)
    expected_coord_bytes = total_rows * 2 * 4 # Coords are Nx2 int32 (4 bytes)
    
    actual_feat = os.path.getsize(feat_bin)
    actual_coord = os.path.getsize(coord_bin)
    
    print(f"\nDone.")
    print(f"Success: {n_success}")
    print(f"Errors:  {n_err}")
    print(f"Total Patches: {total_rows:,}")
    
    if expected_feat_bytes != actual_feat:
        print(f"⚠️  WARNING: Feat size mismatch! Expected {expected_feat_bytes}, got {actual_feat}")
    if expected_coord_bytes != actual_coord:
        print(f"⚠️  WARNING: Coord size mismatch! Expected {expected_coord_bytes}, got {actual_coord}")

    if errors:
        err_path = os.path.join(args.output_dir, "errors.txt")
        with open(err_path, "w") as f:
            for s, e in errors:
                f.write(f"{s}: {e}\n")
        print(f"Error log saved to {err_path}")

if __name__ == "__main__":
    main()