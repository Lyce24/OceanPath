#!/usr/bin/env bash
# ── run_supervised_baselines.sh ──────────────────────────────────────────────
#
# Supervised MIL baselines behind the README's results tables:
#   tasks x initialisations x seeds, each a 5-fold patient-grouped CV run of
#   scripts/train_supervised.py, scored at the patient level.
#
# Tasks:
#   nsclc   TCGA-NSCLC LUAD vs LUSC; every fold model also scores CPTAC-NSCLC
#   braf    TCGA colorectal BRAF mutation
#   panda   PANDA prostate biopsies, ISUP grade 0-5
#
# Initialisations:
#   abmil__random        ABMIL from scratch
#   mamba2mil__random    Mamba-2 MIL from scratch
#   mamba2mil__<method>  Mamba-2 MIL fine-tuned from an SSL checkpoint,
#                        SSL_CKPT_DIR/aggregator_weights_<method>.pt, for
#                        method in vicreg jepa lejepa lejepa_mc
#
# Shared recipe (training=supervised_mil): 60 epochs max, early stopping on
# val loss (patience 10), AdamW lr 1e-4, wd 1e-3, batch 32 slides, bags capped
# at 2,048 patches by spatially stratified sampling. Fine-tuning uses lr 1e-5
# for the pretrained aggregator and 1e-4 for the head. Folds are fixed
# (kfold5_seed42); the seed changes initialisation and data order.
#
# Results land in $OCEANPATH_ROOT/outputs/supervised/<exp_name> with
# exp_name = sup_<data>_<init>__s<seed>; scripts/collect_supervised_results.py
# aggregates them.
#
# Usage:
#   bash scripts/run_supervised_baselines.sh --dry-run
#   bash scripts/run_supervised_baselines.sh --task nsclc --seeds "42 43 44" --jobs 3
#   bash scripts/run_supervised_baselines.sh --init mamba2mil__lejepa --task braf
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${OCEANPATH_ROOT:?set OCEANPATH_ROOT to the data root (see configs/platform/local.yaml)}"
SSL_CKPT_DIR="${SSL_CKPT_DIR:-${OCEANPATH_ROOT}/checkpoints}"
CPTAC_MANIFEST="${OCEANPATH_ROOT}/manifests/SSL/cptac_nsclc_labels.csv"

DRY_RUN=false
TASK_FILTER=""
INIT_FILTER=""
SEEDS_RAW="${SEEDS:-42 43 44}"
JOBS="${JOBS:-1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --task)    TASK_FILTER="${2:-}"; shift 2 ;;
        --init)    INIT_FILTER="${2:-}"; shift 2 ;;
        --seeds)   SEEDS_RAW="${2:-}"; shift 2 ;;
        --jobs)    JOBS="${2:-}"; shift 2 ;;
        -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
        *)         echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done
read -r -a SEEDS <<< "${SEEDS_RAW//,/ }"

COMMON="platform=local encoder=uni2h splits=kfold5 training=supervised_mil \
++training.max_instances=2048 ++training.dataset_max_instances=2048 \
++training.cap_strategy=spatial_stratified"

TASKS=(
    "nsclc|data=tcga_nsclc test_data.mmap_dir=${OCEANPATH_ROOT}/mmap/cptac_nsclc test_data.manifest_csv=${CPTAC_MANIFEST}"
    "braf|data=TCGA_BRAF"
    "panda|data=panda"
)
INITS=(abmil__random mamba2mil__random mamba2mil__vicreg mamba2mil__jepa mamba2mil__lejepa mamba2mil__lejepa_mc)

init_overrides() {
    case "$1" in
        abmil__random)     echo "model=abmil" ;;
        mamba2mil__random) echo "model=mamba2mil training.return_coords=true" ;;
        mamba2mil__*)
            local ckpt="${SSL_CKPT_DIR}/aggregator_weights_${1#mamba2mil__}.pt"
            if [[ ! -f "$ckpt" ]] && ! $DRY_RUN; then
                echo "ERROR: missing SSL checkpoint $ckpt (set SSL_CKPT_DIR)" >&2
                return 1
            fi
            echo "model=mamba2mil training.return_coords=true training.aggregator_weights_path=${ckpt} training.aggregator_lr=1e-5 training.head_lr=1e-4"
            ;;
        *) echo "ERROR: unknown init $1" >&2; return 1 ;;
    esac
}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${OCEANPATH_ROOT}/outputs/supervised/_logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"
MANIFEST="${LOG_DIR}/runs.csv"
echo "task,init,seed,exp_name,command" > "$MANIFEST"

# Collect every command first, then run them JOBS at a time.
COMMANDS=()
for task_spec in "${TASKS[@]}"; do
    IFS='|' read -r task_tag data_ov <<< "${task_spec}"
    [[ -n "$TASK_FILTER" && "$task_tag" != "$TASK_FILTER" ]] && continue
    data_name="${data_ov%% *}"; data_name="${data_name#data=}"
    for init in "${INITS[@]}"; do
        [[ -n "$INIT_FILTER" && "$init" != "$INIT_FILTER" ]] && continue
        iov=$(init_overrides "$init") || exit 2
        for seed in "${SEEDS[@]}"; do
            exp="sup_${data_name}_${init}__s${seed}"
            cmd="python scripts/train_supervised.py ${COMMON} ${data_ov} ${iov} training.seed=${seed} exp_name=${exp}"
            echo "${task_tag},${init},${seed},${exp},\"${cmd}\"" >> "$MANIFEST"
            COMMANDS+=("${exp}|${cmd}")
        done
    done
done

echo "================================================================"
echo "  Supervised baselines: ${#COMMANDS[@]} runs, ${JOBS} at a time"
echo "  Logs and run manifest: ${LOG_DIR}"
echo "================================================================"

run_one() {
    local exp="${1%%|*}" cmd="${1#*|}"
    if $DRY_RUN; then
        echo "  ${cmd}"
        return 0
    fi
    if [[ -f "${OCEANPATH_ROOT}/outputs/supervised/${exp}/summary.json" ]]; then
        echo "  [skip] ${exp} (summary.json exists)"
        return 0
    fi
    echo "  [start $(date +%T)] ${exp}"
    local rc=0
    # shellcheck disable=SC2086
    $cmd > "${LOG_DIR}/${exp}.log" 2>&1 || rc=$?
    echo "  [end   $(date +%T)] ${exp} exit=${rc}"
    return $rc
}

FAIL=0
for entry in "${COMMANDS[@]}"; do
    while (( $(jobs -rp | wc -l) >= JOBS )); do
        wait -n || FAIL=$((FAIL + 1))
    done
    run_one "$entry" &
done
while (( $(jobs -rp | wc -l) > 0 )); do
    wait -n || FAIL=$((FAIL + 1))
done

echo "================================================================"
echo "  Done: ${#COMMANDS[@]} runs, ${FAIL} failed. Logs: ${LOG_DIR}"
echo "================================================================"
[ "$FAIL" -eq 0 ]
