#!/bin/bash
# Label-efficiency linear-probe sweep (scripts/run_label_efficiency.sh). The probes
# are scikit-learn and run on CPU; the GPU is used only to extract slide
# embeddings the first time a (mode, checkpoint) pair is seen, after which the
# embedding cache makes reruns CPU-only.
#
#   sbatch slurm/label_efficiency.sbatch --ssl-ckpt /path/to/best.ckpt
#   sbatch slurm/label_efficiency.sbatch --task braf --fractions "0.01 0.05 0.10"
#
# Arguments are passed through to scripts/run_label_efficiency.sh.
#SBATCH --job-name=oceanpath_label_eff
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

export OCEANPATH_ROOT="${OCEANPATH_ROOT:-${HOME}/scratch}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

uv run --no-sync bash scripts/run_label_efficiency.sh "$@"
