#!/bin/bash
# Supervised k-fold MIL training on one GPU (from scratch or from a pretrained
# aggregator), scored at the patient level.
#
#   sbatch slurm/train_supervised.sbatch data=tcga_nsclc model=abmil \
#       training.seed=43 exp_name=sup_tcga_nsclc_abmil__random__s43
#   sbatch slurm/train_supervised.sbatch data=TCGA_BRAF model=mamba2mil \
#       training.aggregator_weights_path=/path/to/aggregator_weights.pt \
#       exp_name=sup_TCGA_BRAF_mamba2mil__lejepa__s42
#
# Arguments are Hydra overrides for scripts/train_supervised.py. Results go to
# $OCEANPATH_ROOT/outputs/supervised/<exp_name>; the default exp_name names only
# the dataset and model, so give each seed and initialisation its own.
#SBATCH --job-name=oceanpath_supervised
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

export OCEANPATH_ROOT="${OCEANPATH_ROOT:-${HOME}/scratch}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Node: $(hostname)  CPUs: ${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

uv run --no-sync python scripts/train_supervised.py \
    platform=hpc \
    platform.num_gpus=1 \
    platform.strategy=auto \
    platform.num_workers="${SLURM_CPUS_PER_TASK}" \
    encoder=uni2h \
    "$@"
