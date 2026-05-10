#!/usr/bin/env bash
# ── run_supervised_baselines.sh ──────────────────────────────────────────────
#
# Supervised MIL training baselines:
#   3 models  x  4 tasks  =  12 experiments
#
# Models:
#   abmil_scratch       ABMIL supervised from scratch
#   mamba2mil_scratch    Random-init Mamba2MIL supervised from scratch
#   mamba2mil_finetune   SSL-pretrained Mamba2MIL fine-tuning
#
# Tasks:
#   nsclc       NSCLC subtyping (TCGA LUAD vs LUSC)
#   braf        BRAF mutation prediction (CRC)
#   panda       PANDA prostate grading (ISUP 0-5)
#   blca        BLCA bladder cancer (Binary WHO 2022)
#
# All use training=supervised_mil config:
#   batch_size=4, accumulate=4, fixed_n=4096, spatial_stratified, length_bucket
#
# Usage:
#   bash scripts/run_supervised_baselines.sh              # run all 12
#   bash scripts/run_supervised_baselines.sh --dry-run    # print commands only
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs/supervised_baselines_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# ── SSL checkpoint for Mamba2MIL fine-tuning ─────────────────────────────────
# Best VICReg checkpoint (lowest val loss)
SSL_CKPT="/mnt/d/YC.Liu/outputs/pretrain/pretrain_uni2h_pretrain_uni2h_mambamil_vicreg/checkpoints/best-064-5.7908.ckpt"

# ── Shared overrides ────────��───────────────────────────────────────────────
ENCODER=uni2h
SPLITS=kfold5
TRAINING=supervised_mil
COMMON="platform=local encoder=${ENCODER} splits=${SPLITS} training=${TRAINING}"

# ── Model configs ───────────────────────────────────────────────────────────
model_overrides() {
    case "$1" in
        abmil_scratch)
            echo "model=abmil training.aggregator_weights_path=null"
            ;;
        mamba2mil_scratch)
            echo "model=mamba2mil training.aggregator_weights_path=null training.return_coords=true"
            ;;
        mamba2mil_finetune)
            echo "model=mamba2mil training.aggregator_weights_path=${SSL_CKPT} training.return_coords=true training.aggregator_lr=1e-5 training.head_lr=1e-4"
            ;;
        *) echo "ERROR: unknown model $1" >&2; return 1 ;;
    esac
}

MODELS=(abmil_scratch mamba2mil_scratch mamba2mil_finetune)

# ── Tasks (tag | data override) ─────────────────────────────────────────────
TASKS=(
    "nsclc|data=tcga_nsclc"
    "braf|data=TCGA_BRAF"
    "panda|data=panda"
    "blca|data=blca"
)

# ── Runner ────────���──────────────────────────────���──────────────────────────
PASS=0; FAIL=0; SKIP=0

run() {
    local task_tag="$1" model_tag="$2" overrides="$3"
    local run_tag="${task_tag}__${model_tag}"
    local log="${LOG_DIR}/${run_tag}.log"

    echo ""
    echo "================================================================"
    echo "  ${run_tag}"
    echo "  log: ${log}"
    echo "  cmd: python scripts/train_supervised.py ${overrides}"
    echo "================================================================"

    if $DRY_RUN; then
        echo "  [dry run -- skipped]"
        ((SKIP++))
        return 0
    fi

    local rc=0
    python scripts/train_supervised.py ${overrides} 2>&1 | tee "${log}" || rc=$?

    if [ $rc -eq 0 ]; then
        echo "  >>> ${run_tag} -- OK"
        ((PASS++))
    else
        echo "  >>> ${run_tag} -- FAILED (exit ${rc})" >&2
        ((FAIL++))
    fi
}

# ── Main loop ─────────────────────────────────────��─────────────────────────
TOTAL=0
for task_spec in "${TASKS[@]}"; do
    IFS='|' read -r task_tag data_ov <<< "${task_spec}"
    for model in "${MODELS[@]}"; do
        mov=$(model_overrides "${model}")
        run "${task_tag}" "${model}" "${COMMON} ${data_ov} ${mov}"
        ((TOTAL++))
    done
done

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Supervised baseline sweep complete"
echo "    Total: ${TOTAL}  Pass: ${PASS}  Fail: ${FAIL}  Skip: ${SKIP}"
echo "    Logs:  ${LOG_DIR}/"
echo "================================================================"

[ "${FAIL}" -eq 0 ]
