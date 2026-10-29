# Slurm job templates

Templates for the runs behind the results in the top-level README. They were
written for Brown's Oscar cluster (`gpu` partition, RTX 3090/A5000-class GPUs) and
take Hydra overrides as arguments, so each job is one command line.

| Template | What it runs | Default resources |
| --- | --- | --- |
| `pretrain.sbatch` | `scripts/pretrain.py`, DDP across one node's GPUs | 2 GPUs, 2 × 6 CPUs, 96 GB, 48 h |
| `train_supervised.sbatch` | `scripts/train_supervised.py`, k-fold, patient-level scoring | 1 GPU, 8 CPUs, 48 GB, 12 h |
| `label_efficiency.sbatch` | `scripts/run_label_efficiency.sh` (linear probes) | 1 GPU, 16 CPUs, 64 GB, 24 h |

Before submitting:

1. Build the environment on a GPU node once: `make sync-gpu` (compiles the Mamba-2
   kernels; load your cluster's CUDA module first if `nvcc` is not on the path).
2. Point `OCEANPATH_ROOT` at the data root (defaults to `~/scratch`); the layout is
   described in `configs/platform/example.yaml`.
3. Edit `--partition` (and add `--account` if your cluster needs one).

Logs go to `logs/<job-name>_<job-id>.{out,err}` in the submission directory
(ignored by git).
