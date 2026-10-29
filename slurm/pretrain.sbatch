#!/bin/bash
# SSL pretraining of a MIL aggregator with DDP across the GPUs of one node.
#
#   sbatch slurm/pretrain.sbatch model=mamba2mil pretrain_training=lejepa_2c
#   sbatch --gres=gpu:4 --ntasks-per-node=4 slurm/pretrain.sbatch platform.num_gpus=4 ...
#
# Arguments are Hydra overrides for scripts/pretrain.py. Lightning picks up the
# Slurm environment, so --ntasks-per-node must equal the GPU count and
# platform.num_gpus (configs/platform/hpc.yaml derives devices from it). Edit the
# partition, account and module lines for your cluster.
#SBATCH --job-name=oceanpath_pretrain
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# module load cuda/12.8   # only needed to build the mamba extra on the cluster

export OCEANPATH_ROOT="${OCEANPATH_ROOT:-${HOME}/scratch}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "Node: $(hostname)  Tasks: ${SLURM_NTASKS}  CPUs/task: ${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "Python: $(uv run --no-sync python -c 'import sys; print(sys.executable, sys.version.split()[0])')"

srun uv run --no-sync python scripts/pretrain.py \
    platform=hpc \
    data=uni2h_pretrain_2048 \
    encoder=uni2h \
    "$@"
