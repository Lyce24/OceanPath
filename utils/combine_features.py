import os
import h5py
import numpy as np
import json
from tqdm import tqdm
import torch

def manage_feature(project, major_dir, aggregation, encoder: str, precision: int = 16, method: str = "mmap"):
    """
    Manages the memory-mapped feature store. If a combined memory-mapped store
    does not exist, it converts all H5 files in the feature_dir into a new
    memory-mapped store. If it exists, it updates the store with any new H5 files.

    Args:
        feature_dir (str): The directory containing the .h5 files and where
                           the 'combined_mmap' directory will be located.
    """
    for encoder in [encoder]:
        if encoder in ["uni_v1", "uni_v2", "gigapath", "resnet50", "kaiko-vitb16", "mSTAR", "vit-large_patch16_224"]:
            res, mag = 256, 20
            feature_dir = f"../{project}/{major_dir}/{mag}x_{res}px_0px_overlap/features_{encoder}"
        elif encoder in ["hoptimus1", "virchow2", "phikon_v2"]:
            res, mag = 224, 20
            feature_dir = f"../{project}/{major_dir}/{mag}x_{res}px_0px_overlap/features_{encoder}"
        elif encoder in ["conch_v15"]:
            res, mag = 512, 20
            feature_dir = f"../{project}/{major_dir}/{mag}x_{res}px_0px_overlap/features_{encoder}"
        elif encoder in ["titan", "feather"]:
            res, mag = 512, 20
            feature_dir = f"../{project}/{major_dir}/{mag}x_{res}px_0px_overlap/slide_features_{encoder}"

    combined_feature_dir = f"../{project}/combined_features/{major_dir}/features_{encoder}"

    if aggregation in ["mean", "max"]:
        static_aggregation_calculation(aggregation, feature_dir)
        combined_feature_dir = os.path.join(combined_feature_dir, aggregation)
        feature_dir = os.path.join(feature_dir, aggregation)

    if method == "mmap":
        combined_mmap_dir = os.path.join(combined_feature_dir, f"combined_mmap_{precision}")
        metadata_path = os.path.join(combined_mmap_dir, "metadata.json")

        print(f"Creating mmap for {encoder} with aggregation {aggregation} at precision {precision}...")

        if not os.path.exists(metadata_path):
            print(f"Metadata file not found at {metadata_path}. Starting fresh conversion to mmap.")
            _combine_h5_to_mmap(feature_dir, combined_mmap_dir, precision=precision)
        else:
            print(f"Metadata file found at {metadata_path}. Starting incremental update.")
            _update_mmap_with_new_files(combined_mmap_dir, feature_dir, precision=precision)
            
    elif method == "pt":
        combined_pt_path = os.path.join(combined_feature_dir, f"combined_pt_{precision}.pt")
        print(f"Combining features into .pt file at {combined_pt_path}...")
        _combine_h5_to_pt(
            feature_dir=feature_dir,
            out_path=combined_pt_path,
            precision=precision,
            skip_empty=True
        )

def static_aggregation_calculation(aggregation, feature_dir):
    target_dir = os.path.join(feature_dir, aggregation)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    processed_id = os.listdir(feature_dir)
    processed_id = [i for i in processed_id if i.endswith(".h5")]

    os.makedirs(target_dir, exist_ok=True)

    target_id = os.listdir(target_dir)
    target_id = [i for i in target_id if i.endswith(".h5")]

    not_targeted_id = set(processed_id) - set(target_id)
    if not not_targeted_id:
        print(f"All files already processed for {aggregation} aggregation in {target_dir}.")
        return

    patch_feature_dim = None # Will be determined from the first H5 file

    for wsi_id in tqdm(not_targeted_id, desc=f"Processing {encoder} with {aggregation} aggregation"):
        h5_path = os.path.join(feature_dir, f"{wsi_id}")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Feature file not found: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            arr = f["features"][:]  # numpy array of shape (n_patches, feature_dim)
            
            if patch_feature_dim is None:
                patch_feature_dim = arr.shape[1]
                # print(f"Determined patch feature dimension: {patch_feature_dim}")
            elif patch_feature_dim != arr.shape[1]:
                raise ValueError(
                    f"Feature dimension mismatch. Expected {patch_feature_dim}, "
                    f"got {arr.shape[1]} for {wsi_id}"
                )
            
            if arr.shape[0] == 0:
                print(f"Warning: WSI {wsi_id} has 0 patches. Skipping or handle as needed.")
                # Decide how to handle WSIs with no patches, e.g., skip or use a zero tensor.
                # For simplicity, let's create a tensor with one zero-feature if empty.
                # This might need more sophisticated handling in a real scenario.
                preloaded_features = torch.zeros(1, patch_feature_dim).float()
            else:
                preloaded_features = torch.from_numpy(arr).float()
                
            # calculate mean
            if aggregation == "mean":
                mean_feature = preloaded_features.mean(dim=0, keepdim=True)
                mean_h5_path = os.path.join(target_dir, f"{wsi_id}")
                os.makedirs(target_dir, exist_ok=True)
                with h5py.File(mean_h5_path, "w") as mean_file:
                    mean_file.create_dataset("features", data=mean_feature.numpy())
                # print(f"Processed WSI {wsi_id} for {encoder}: mean feature saved to {mean_h5_path}")
                
            elif aggregation == "max":
                max_feature = preloaded_features.max(dim=0, keepdim=True)[0]
                max_h5_path = os.path.join(target_dir, f"{wsi_id}")
                os.makedirs(target_dir, exist_ok=True)
                with h5py.File(max_h5_path, "w") as max_file:
                    max_file.create_dataset("features", data=max_feature.numpy())
                # print(f"Processed WSI {wsi_id} for {encoder}: max feature saved to {max_h5_path}")
            else:
                raise ValueError(f"Unsupported aggregation method: {aggregation}")

    print(f"Processed all WSIs for encoder {encoder}. Features saved in {target_dir} with aggregation {aggregation}.")

def _update_mmap_with_new_files(mmap_dir, new_h5_dir, precision=16):
    """
    Incrementally updates an existing memory-mapped feature store with new H5 files.

    Args:
        mmap_dir (str): The directory containing the existing 'features.npy', and 'metadata.json'.
        new_h5_dir (str): The directory where the new .h5 files are located.
    """
    # --- 1. Define Paths and Load Existing Metadata ---
    print("--- 1. Loading existing metadata and feature files ---")
    metadata_path = os.path.join(mmap_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Existing metadata.json not found. Cannot perform an update.")

    with open(metadata_path, 'r') as f:
        old_metadata = json.load(f)
        # keys: 'slides', 'total_patches', 'precision', 'feature_dim'

    old_features_path = os.path.join(mmap_dir, "features.npy")
    
    # --- 2. Identify Only the New H5 Files ---
    print("\n--- 2. Identifying new H5 files to add ---")
    existing_slide_ids = set(old_metadata['slides'].keys())
    all_new_h5_files = [f for f in os.listdir(new_h5_dir) if f.endswith(".h5")]
    
    files_to_add = []
    for h5_file in all_new_h5_files:
        wsi_id = h5_file.replace(".h5", "")
        if wsi_id not in existing_slide_ids:
            files_to_add.append(h5_file)
            
    if not files_to_add:
        print("No new files to add. The feature store is already up-to-date.")
        return
        
    print(f"Found {len(files_to_add)} new H5 files to process.")
    
    # --- 3. Scan NEW Files to Determine Additional Size ---
    print("\n--- 3. Scanning new H5 files for additional patches ---")
    additional_patches = 0
    feature_dim = old_metadata['feature_dim']

    for h5_file in tqdm(files_to_add, desc="Scanning new H5"):
        h5_path = os.path.join(new_h5_dir, h5_file)
        with h5py.File(h5_path, 'r') as f:
            features = f["features"]
            if features.shape[0] > 0 and features.shape[1] != feature_dim:
                raise ValueError(f"Feature dimension mismatch in {h5_file}!")
            additional_patches += features.shape[0]

    # --- 4. Calculate New Total Size and Create New MMP Files ---
    print("\n--- 4. Preparing new memory-mapped files ---")
    old_total_patches = old_metadata.get('total_patches', 0)
    
    # sanity check
    if old_metadata['slides']:
        # Find the last actual slide, not a dummy one
        last_slide_id = None
        for slide_id in reversed(list(old_metadata['slides'].keys())):
            if not old_metadata['slides'][slide_id].get('dummy', False):
                last_slide_id = slide_id
                break
        
        if last_slide_id is not None:
            if old_total_patches != old_metadata['slides'][last_slide_id]['end_idx']:
                print(f"Warning: Total patches in metadata ({old_total_patches}) does not match last slide's end index ({old_metadata['slides'][last_slide_id]['end_idx']}).")
                print(f"Adding from last slide: {last_slide_id} with end index {old_metadata['slides'][last_slide_id]['end_idx']}")
                old_total_patches = old_metadata['slides'][last_slide_id]['end_idx']
    else:
        print("No existing slides found in metadata. Starting fresh.")
        old_total_patches = 0
        
    new_total_patches = old_total_patches + additional_patches
    
    print(f"Old patches: {old_total_patches}, New patches: {additional_patches}, Total: {new_total_patches}")
    
    bytes_per = 2 if precision == 16 else 4
    est_gb = new_total_patches * feature_dim * bytes_per / (1024**3)
    print(f"Feature dim: {feature_dim}, total patches: {new_total_patches:,}")
    print(f"Estimated RAM for tensors: ~{est_gb:.2f} GB (precision={precision})")

    # Create new files with temporary names
    new_features_path = os.path.join(mmap_dir, "features_new.npy")
    new_metadata_path = os.path.join(mmap_dir, "metadata_new.json")

    # sanity check
    if old_metadata['precision'] != precision:
        raise ValueError(f"Precision mismatch: existing {old_metadata['precision']}, requested {precision}.")

    if precision == 16:
        new_features_mmap = np.memmap(new_features_path, dtype=np.float16, mode='w+', shape=(new_total_patches, feature_dim))
    elif precision == 32:
        new_features_mmap = np.memmap(new_features_path, dtype=np.float32, mode='w+', shape=(new_total_patches, feature_dim))
    else:
        raise ValueError("Unsupported precision. Use 16 or 32.")

    # --- 5. Fast-Copy Old Data ---
    print("\n--- 5. Fast-copying existing data to new mmap file ---")
    if old_total_patches > 0:
        if precision == 16:
            old_features_mmap = np.memmap(old_features_path, dtype=np.float16, mode='r', shape=(old_total_patches, feature_dim))
        elif precision == 32:
            old_features_mmap = np.memmap(old_features_path, dtype=np.float32, mode='r', shape=(old_total_patches, feature_dim))
        new_features_mmap[:old_total_patches] = old_features_mmap
        
    else:
        print("No old data to copy.")
        
    print("Old data copied to new memory-mapped file.")

    # --- 6. Append New Data from H5 Files ---
    print("\n--- 6. Appending new data from H5 files ---")
    new_metadata = old_metadata.copy()
    current_pos = old_total_patches

    for h5_file in tqdm(files_to_add, desc="Appending new H5"):
        wsi_id = h5_file.replace(".h5", "")
        h5_path = os.path.join(new_h5_dir, h5_file)
        
        with h5py.File(h5_path, 'r') as f:
            features = f["features"][:]
            
            if features.ndim == 1:
                features = features[np.newaxis, :]  # Ensure it's 2D
                
            num_patches = features.shape[0]

            if num_patches == 0:
                new_metadata["slides"][wsi_id] = {"start_idx": -1, "end_idx": -1, "dummy": True}
                continue

            end_pos = current_pos + num_patches
            new_features_mmap[current_pos:end_pos] = features.astype(np.float16) if precision == 16 else features.astype(np.float32)
            
            new_metadata["slides"][wsi_id] = {"start_idx": current_pos, "end_idx": end_pos, "dummy": False}
            current_pos = end_pos

    # --- 7. Save and Finalize ---
    print("\n--- 7. Saving new metadata and flushing memory-mapped files ---")
    if current_pos != new_total_patches:
        print(f"Warning: Expected {new_total_patches} patches, but wrote {current_pos}. Check data.")
        
    new_metadata["total_patches"] = new_total_patches
        
    new_features_mmap.flush()
        
    with open(new_metadata_path, 'w') as f:
        json.dump(new_metadata, f, indent=4)

    # Atomically replace the old files with the new ones
    # This ensures that if the script fails, you don't lose your original data
    if os.path.exists(old_features_path):
        os.remove(old_features_path)
    if os.path.exists(metadata_path): # Check if it exists before trying to remove
        os.remove(metadata_path)

    os.rename(new_features_path, old_features_path)
    os.rename(new_metadata_path, metadata_path)
    
    print("Complete.")

def _combine_h5_to_mmap(feature_dir, output_dir, precision=16):
    """
    Scans a directory of .h5 feature files and converts them into
    memory-mapped .npy files for features, along with a
    metadata JSON file.

    Args:
        feature_dir (str): The directory containing the .h5 files.
    """
    h5_files = [f for f in os.listdir(feature_dir) if f.endswith(".h5")]
    if not h5_files:
        print(f"No .h5 files found in {feature_dir}. Nothing to do.")
        return
    
    print(f"Found {len(h5_files)} .h5 files in {feature_dir}. Starting conversion...")
    print(f"Output will be saved to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)

    features_path = os.path.join(output_dir, "features.npy")
    metadata_path = os.path.join(output_dir, "metadata.json")

    print("--- Stage 1: Scanning H5 files to determine dimensions ---")
    total_patches = 0
    feature_dim = None

    for h5_file in tqdm(h5_files, desc="Scanning files"):
        h5_path = os.path.join(feature_dir, h5_file)
        with h5py.File(h5_path, 'r') as f:
            features = f["features"]
            
            if features.ndim == 1:
                num_patches = 1
                dim = features.shape[0]
            else:
                num_patches = features.shape[0]
                dim = features.shape[1]

            # Handle slides with no patches
            if num_patches == 0:
                continue

            total_patches += num_patches

            if feature_dim is None:
                feature_dim = dim
            elif feature_dim != dim:
                raise ValueError(f"Inconsistent feature dimension in {h5_file}")

    if feature_dim is None:
        print("No features found in any H5 files. Exiting.")
        return

    print(f"Scan complete. Total patches: {total_patches}, Feature dim: {feature_dim}")

    print("--- Stage 2: Creating and populating memory-mapped files ---")
    if precision == 16:
        features_mmap = np.memmap(features_path, dtype=np.float16, mode='w+', shape=(total_patches, feature_dim))
    elif precision == 32:
        features_mmap = np.memmap(features_path, dtype=np.float32, mode='w+', shape=(total_patches, feature_dim))

    metadata = {
        "slides": {},
        "total_patches": total_patches,
        "precision": precision,
        "feature_dim": feature_dim,
    }
    
    bytes_per = 2 if precision == 16 else 4
    est_gb = total_patches * feature_dim * bytes_per / (1024**3)
    print(f"Feature dim: {feature_dim}, total patches: {total_patches:,}")
    print(f"Estimated RAM for tensors: ~{est_gb:.2f} GB (precision={precision})\n")

    current_pos = 0
    for h5_file in tqdm(h5_files, desc="Converting to MMP"):
        wsi_id = h5_file.replace(".h5", "")
        h5_path = os.path.join(feature_dir, h5_file)

        with h5py.File(h5_path, 'r') as f:
            features = f["features"][:]
            if features.ndim == 1:
                features = features[np.newaxis, :]
                
            num_patches = features.shape[0]

            if num_patches == 0:
                metadata["slides"][wsi_id] = {"start_idx": -1, "end_idx": -1, "dummy": True}
                print(f"Warning: WSI {wsi_id} has 0 patches. Skipping.")
                continue

            end_pos = current_pos + num_patches

            features_mmap[current_pos:end_pos, :] = features.astype(np.float16) if precision == 16 else features.astype(np.float32)
           
            metadata["slides"][wsi_id] = {
                "start_idx": current_pos,
                "end_idx": end_pos,
                "dummy": False
            }
            current_pos = end_pos

    features_mmap.flush()
    if current_pos != total_patches:
        print(f"Warning: Expected {total_patches} patches, but wrote {current_pos}. Check data.")

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print("\nConversion complete.")
    print(f"  -> Features: {features_path}")
    print(f"  -> Metadata: {metadata_path}")

def _combine_h5_to_pt(
    feature_dir: str,
    out_path: str,
    precision: int = 16,
    skip_empty: bool = True,
):
    """
    Combine .h5 feature files into a single .pt file storing a dict:
        { slide_id: Tensor [num_patches, feature_dim], ... }

    Each .h5 must contain a dataset named `key` (default: "features") with shape
    [num_patches, feature_dim] (or [feature_dim] for a single patch).

    Args:
        feature_dir: directory containing .h5 files.
        out_path: output .pt file path.
        precision: 16 or 32 (float16 or float32).
        key: dataset name in the .h5 files.
        sort_files: sort file names for deterministic ordering.
        skip_empty: if True, slides with 0 patches are skipped; else saved as [0, D].
    """
    assert precision in (16, 32), "precision must be 16 or 32"
    dtype = torch.float16 if precision == 16 else torch.float32

    h5_files = [f for f in os.listdir(feature_dir) if f.lower().endswith(".h5")]
    if not h5_files:
        print(f"No .h5 files found in {feature_dir}.")
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"Found {len(h5_files)} .h5 files.\nSaving to: {out_path}\n")

    # --- Scan pass: validate feature_dim, count patches, estimate RAM
    feature_dim = None
    total_patches = 0
    candidates = []

    print("--- Stage 1: Scan & validate ---")
    for fn in tqdm(h5_files, desc="Scanning"):
        path = os.path.join(feature_dir, fn)
        with h5py.File(path, "r") as f:
            if "features" not in f:
                print(f"  [WARN] {fn} has no dataset 'features', skipping.")
                continue
            ds = f["features"]
            if ds.ndim == 1:
                num = 1
                dim = int(ds.shape[0])
            else:
                num = int(ds.shape[0])
                dim = int(ds.shape[1])

            if feature_dim is None:
                feature_dim = dim
            elif feature_dim != dim:
                raise ValueError(
                    f"Inconsistent feature dim in {fn}: got {dim}, expected {feature_dim}"
                )

            candidates.append((fn, num))
            total_patches += num

    if feature_dim is None:
        print("No usable datasets found; nothing to do.")
        return

    bytes_per = 2 if precision == 16 else 4
    est_gb = total_patches * feature_dim * bytes_per / (1024**3)
    print(f"Feature dim: {feature_dim}, total patches: {total_patches:,}")
    print(f"Estimated RAM for tensors: ~{est_gb:.2f} GB (precision={precision})\n")

    # --- Load pass: build dict of tensors
    print("--- Stage 2: Load tensors into a dict ---")
    slides = {}
    kept = 0
    empty = 0

    for fn, num in tqdm(candidates, desc="Loading"):
        slide_id = os.path.splitext(fn)[0]
        path = os.path.join(feature_dir, fn)

        with h5py.File(path, "r") as f:
            arr = f["features"][:]
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]

        if arr.shape[0] == 0 and skip_empty:
            empty += 1
            continue

        # Convert to torch with requested dtype
        t = torch.from_numpy(arr)
        if t.dtype != dtype:
            t = t.to(dtype)
        
        # Ensure contiguous for best serialization/read performance
        t = t.contiguous()

        slides[slide_id] = t
        kept += 1

    print(f"Slides kept: {kept}, empty skipped: {empty}\n")

    # --- Save
    print("--- Stage 3: Save .pt ---")
    package = {
        "slides": slides,  # dict: {slide_id: Tensor [Ni, D]}
        "info": {
            "feature_dim": feature_dim,
            "precision": precision,
            "num_slides": kept,
            "total_patches": int(sum(v.shape[0] for v in slides.values())),
            "dim": feature_dim,
            "dataset_key": "features",
        },
    }
    torch.save(package, out_path, _use_new_zipfile_serialization=True)
    print(f"Done. Wrote: {out_path}") 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process WSI features")
    parser.add_argument("-e", type=str, default="uni_v1", help="Encoder to use")
    parser.add_argument("-a", type=str, default="abmil", help="Aggregation method (mean, max, abmil)")
    parser.add_argument("-d", type=str, default="wsi_processed_no_penmarks", help="Major directory for features")
    parser.add_argument("-m", type=str, default="mmap", help="Method to use (mmap or pt)")
    parser.add_argument("-p", type=str, default="CRC_AI", help="Project name")
    args = parser.parse_args()

    # Loop through your encoders and run the conversion for each

    aggregation = args.a
    encoder = args.e
    major_dir = args.d
    method = args.m
    project = args.p
    print(f"\n--- Processing Encoder: {encoder} ---")
    manage_feature(project, major_dir, aggregation, encoder, precision=16, method=method)