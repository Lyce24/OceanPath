import os
from huggingface_hub import snapshot_download

# --- CONFIGURATION ---
REPO_ID = "MahmoodLab/UNI2-h-features"
# Based on your previous prompt, you are in /mnt/d/YC.Liu
# We will download directly there to avoid moving 1TB later.
LOCAL_DIR = "/mnt/d/YC.Liu/UNI2_h_features" 

# --- OPTIMIZATION ---
# Enable the high-performance Rust downloader
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

print(f"Starting turbo download of {REPO_ID} to {LOCAL_DIR}...")
print("This uses the Rust accelerator. If it fails, remove the os.environ line.")

try:
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False, # Essential: Get real files, not symlinks
        resume_download=True,         # Essential: Auto-resume if internet drops
        max_workers=16                # High parallelism for many small files
    )
    print("\n✅ Download Complete!")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("Tip: If you run out of disk space, delete the hidden '.cache' folder in your home dir.")