import time
import torch
import pandas as pd
import h5py
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =============================================================================
# 1. DEFINE DATASETS
# =============================================================================

class StandardDataset(Dataset):
    """Basline: Opens .h5 or .pt file every time."""
    def __init__(self, file_paths):
        self.paths = file_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        # Simulate standard loading
        if path.endswith('.h5'):
            with h5py.File(path, 'r') as f:
                # Note: Copy to tensor to simulate real usage
                return torch.from_numpy(f['features'][:])
        else:
            return torch.load(path, map_location='cpu', weights_only=True)

class MemmapDatasetStub(Dataset):
    """
    PLACEHOLDER: Replace this with your actual MemmapDataset class.
    Assuming you have a class that reads from the binary blob.
    """
    def __init__(self, mmap_dir, csv_df):
        self.mmap_dir = mmap_dir
        # Assuming you have an index or offsets file in that dir
        # self.index = ... 
        # self.data = np.memmap(...)
        
        # FOR BENCHMARKING ONLY: 
        # If you don't have your class ready, we simulate a fast memory read
        # to show the theoretical speed of memmap.
        self.fake_data = torch.randn(10000, 1024) # Standard bag shape

    def __len__(self):
        return 10000 # Pretend we have many slides

    def __getitem__(self, idx):
        # In your real code, this would be: 
        # return torch.from_numpy(self.data[start:end])
        return self.fake_data.clone() # Clone simulates the memory access cost

# =============================================================================
# 2. BENCHMARK ENGINE
# =============================================================================

def measure_throughput(dataset, batch_size=1, num_workers=4, steps=10, name="Method"):
    """
    Measures samples/sec using a real DataLoader.
    
    Args:
        dataset: The torch Dataset to test.
        batch_size: 1 (since bags vary in size, usually BS=1).
        num_workers: Standard is 4-8.
        steps: How many batches to average.
    """
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        collate_fn=None # Default collate usually works if bags are same size, else use custom
    )
    
    iterator = iter(loader)
    
    # --- WARMUP (CRITICAL TO AVOID COLD START) ---
    # This loads libraries and fills OS page cache partially
    print(f"[{name}] Warming up (discarding first 5 batches)...")
    for _ in range(5):
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(loader)
            next(iterator)

    # --- MEASUREMENT ---
    print(f"[{name}] Measuring over {steps} batches...")
    start_time = time.perf_counter()
    
    for _ in range(steps):
        try:
            _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            _ = next(iterator)
            
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    total_samples = steps * batch_size
    throughput = total_samples / total_time
    
    return throughput

# =============================================================================
# 3. RUNNER
# =============================================================================

if __name__ == "__main__":
    # --- CONFIG ---
    CSV_PATH = "./data/slides.csv"  # Your CSV path
    MMAP_DIR = "./src/univ2_mmap/"  # Your Mmap path
    BATCH_SIZE = 1                  # Standard for MIL
    NUM_WORKERS = 4                 # Typical CPU worker count
    TEST_STEPS = 50                 # Average over 50 batches for stability
    
    # 1. Load Paths
    # df = pd.read_csv(CSV_PATH)
    # paths = df['path'].tolist()
    
    # MOCK DATA (Remove this block when running real code)
    paths = [f"sample_{i}.h5" for i in range(100)] 
    # Create a dummy h5 file for the test if it doesn't exist
    if not os.path.exists("sample_0.h5"):
        with h5py.File("sample_0.h5", 'w') as f:
            f.create_dataset('features', data=np.random.randn(10000, 1024))
        paths = ["sample_0.h5"] * 100

    # 2. Instantiate Datasets
    # A. Standard
    std_dataset = StandardDataset(paths)
    
    # B. Ours (Initialize your actual class here)
    # ours_dataset = YourActualMemmapDataset(MMAP_DIR, ...)
    ours_dataset = MemmapDatasetStub(MMAP_DIR, None) 

    # 3. Run Benchmarks
    print(f"--- STARTING IO BENCHMARK (Avg of {TEST_STEPS} Batches) ---\n")
    
    speed_std = measure_throughput(std_dataset, BATCH_SIZE, NUM_WORKERS, TEST_STEPS, "Standard (.h5)")
    speed_ours = measure_throughput(ours_dataset, BATCH_SIZE, NUM_WORKERS, TEST_STEPS, "Ours (Memmap)")
    
    # 4. Calculate Gain
    gain = speed_ours / speed_std
    
    # 5. Output Table
    print("\n" + "="*60)
    print(f"{'Metric':<25} | {'Standard (.h5)':<20} | {'Ours (Memmap)':<15} | {'Gain':<10}")
    print("-" * 78)
    print(f"{'Throughput':<25} | {speed_std:.1f} slides/sec      | {speed_ours:.1f} slides/sec   | {gain:.1f}x")
    print("="*60)