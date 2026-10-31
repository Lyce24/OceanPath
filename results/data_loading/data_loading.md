| Reader | Cold, ms per bag (p50) | Warm, ms per bag (p50 / p95) | Warm DataLoader, 4 workers |
| --- | --- | --- | --- |
| Release HDF5, float32 | 66 (1.0×) | 27.8 / 54.3 | 51 slides/s |
| Release HDF5, sampled rows only | 437 (0.1×) | 360.1 / 457.9 | — |
| HDF5, float16 | 50 (1.3×) | 25.0 / 41.0 | 79 slides/s |
| OceanPath store, float16 memmap | 41 (1.6×) | 20.6 / 42.9 | 94 slides/s |

100 TCGA-NSCLC slides (median 11,771 patches), 2,048-patch spatially stratified bags as float32; cold = one pass after evicting the reader's files from the page cache, warm = 5 passes on cached files. Speed-ups are against the release files.
