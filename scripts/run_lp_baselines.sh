#!/usr/bin/env bash
# ── run_lp_baselines.sh ─────────────────────────────────────────────────────
#
# Linear-probing baseline sweep:
#   5 tasks  x  4 embedding modes  =  20 experiments
#
# All runs use the same fixed-cap regime as SSL pretraining
#   max_instances=2048 + sampling_mode=spatial_stratified
# (set in configs/linear_probing.yaml). Outputs land under:
#   outputs/lp_2048/<timestamp>/<task>__<mode>/<cache_tag>/
#
# Tasks
#   nsclc_cv    NSCLC subtyping (TCGA LUAD vs LUSC)         grouped_cv     binary
#   nsclc_xfer  NSCLC cross-cohort (TCGA -> CPTAC)          external_test  binary
#   braf        BRAF mutation prediction (TCGA CRC)         grouped_cv     binary
#   panda       PANDA prostate ISUP grading (0..5)          grouped_cv     ordinal (qwk)
#   gej         GEJ Barrett's per-section severity (0..3)   grouped_cv     ordinal (qwk)
#                 - aggregate_to_patient=false (sections of the same case
#                   carry different labels; eval is per-slide, CV groups
#                   by patient to avoid leakage)
#
# Embedding modes
#   mean_pool          mask-aware mean pooling
#   max_pool           mask-aware max pooling
#   mean_max_pool      concat of mean + max pooling
#   aggregator_random  random-init Mamba2MIL (frozen, no checkpoint)
#
# Usage
#   bash scripts/run_lp_baselines.sh                  # run all 20
#   bash scripts/run_lp_baselines.sh --dry-run        # print commands only
#   bash scripts/run_lp_baselines.sh --task gej       # one task, all modes
#   bash scripts/run_lp_baselines.sh --mode mean_pool # all tasks, one mode
#   bash scripts/run_lp_baselines.sh --task panda --mode aggregator_random
#
# Notes
#   - Embedding mmaps are cached under platform.mmap_root/lp_embeddings/, so
#     re-runs across timestamps share extraction work. Cache key includes
#     mode + arch + ckpt + max_instances + sampling_mode, so changing any
#     of those will not silently reuse stale embeddings.
#   - Each run's exit code is captured independently; a failure in one
#     does not abort the sweep.
#   - A CSV summary is written to <sweep_dir>/sweep_summary.csv.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Argument parsing ────────────────────────────────────────────────────────
DRY_RUN=false
TASK_FILTER=""
MODE_FILTER=""

usage() {
    cat <<EOF
Usage: $0 [--dry-run] [--task TASK] [--mode MODE]

Tasks: nsclc_cv  nsclc_xfer  braf  panda  gej
Modes: mean_pool  max_pool  mean_max_pool  aggregator_random
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true; shift ;;
        --task)     TASK_FILTER="${2:-}"; shift 2 ;;
        --mode)     MODE_FILTER="${2:-}"; shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *)          echo "Unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

# ── Validate filters (fail fast on typos) ───────────────────────────────────
VALID_TASKS=" nsclc_cv nsclc_xfer braf panda gej "
VALID_MODES=" mean_pool max_pool mean_max_pool aggregator_random "
if [[ -n "$TASK_FILTER" && "$VALID_TASKS" != *" $TASK_FILTER "* ]]; then
    echo "ERROR: unknown --task '$TASK_FILTER'. Valid:${VALID_TASKS}" >&2
    exit 2
fi
if [[ -n "$MODE_FILTER" && "$VALID_MODES" != *" $MODE_FILTER "* ]]; then
    echo "ERROR: unknown --mode '$MODE_FILTER'. Valid:${VALID_MODES}" >&2
    exit 2
fi

# ── Sweep directories ───────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${PROJECT_ROOT}/outputs/lp_2048/${TIMESTAMP}"
LOG_DIR="${SWEEP_DIR}/logs"
mkdir -p "$LOG_DIR"

SUMMARY_FILE="${SWEEP_DIR}/sweep_summary.csv"
echo "task,mode,status,exit_code,duration_s,output_dir,log" > "${SUMMARY_FILE}"

echo "================================================================"
echo "  LP baseline sweep"
echo "  Sweep dir: ${SWEEP_DIR}"
echo "  Filters:   task=${TASK_FILTER:-<all>} mode=${MODE_FILTER:-<all>}"
echo "================================================================"

# ── Shared overrides ────────────────────────────────────────────────────────
ENCODER=uni2h
N_BOOT=1000
COMMON="platform=local encoder=${ENCODER} n_boot=${N_BOOT}"

# ── Modes ───────────────────────────────────────────────────────────────────
mode_overrides() {
    case "$1" in
        mean_pool)         echo "embedding.mode=mean_pool" ;;
        max_pool)          echo "embedding.mode=max_pool" ;;
        mean_max_pool)     echo "embedding.mode=mean_max_pool" ;;
        aggregator_random) echo "model=mamba2mil embedding.mode=aggregator embedding.ckpt=null" ;;
        *) echo "ERROR: unknown mode $1" >&2; return 1 ;;
    esac
}

MODES=(mean_pool max_pool mean_max_pool aggregator_random)

# ── Tasks ───────────────────────────────────────────────────────────────────
# Format: "tag|data_overrides|extra_overrides"
CPTAC_MANIFEST=/mnt/d/YC.Liu/manifests/SSL/cptac_nsclc_labels.csv

TASKS=(
    "nsclc_cv|data=tcga_nsclc|protocol=grouped_cv"
    "nsclc_xfer|data=tcga_nsclc|protocol=external_test test.mmap_dir=./mmap/cptac_nsclc test.manifest_csv=${CPTAC_MANIFEST}"
    "braf|data=TCGA_BRAF|protocol=grouped_cv"
    "panda|data=panda|protocol=grouped_cv primary_metric=qwk"
    "gej|data=gej_univ2|protocol=grouped_cv aggregate_to_patient=false primary_metric=qwk"
)

# ── Runner ──────────────────────────────────────────────────────────────────
PASS=0; FAIL=0; SKIP=0; TOTAL=0

run() {
    local task_tag="$1" mode_tag="$2" overrides="$3"
    local run_tag="${task_tag}__${mode_tag}"
    local log="${LOG_DIR}/${run_tag}.log"
    local out_dir="${SWEEP_DIR}/${run_tag}"

    echo ""
    echo "================================================================"
    echo "  ${run_tag}"
    echo "  log: ${log}"
    echo "  out: ${out_dir}"
    echo "  cmd: python scripts/linear_probing.py ${overrides} output_dir=${out_dir}"
    echo "================================================================"

    if $DRY_RUN; then
        echo "  [dry run -- skipped]"
        ((SKIP++))
        echo "${task_tag},${mode_tag},dry_run,0,0,${out_dir},${log}" >> "${SUMMARY_FILE}"
        return 0
    fi

    local t0
    t0=$(date +%s)
    local rc=0
    # shellcheck disable=SC2086
    python scripts/linear_probing.py ${overrides} output_dir="${out_dir}" 2>&1 | tee "${log}" || rc=$?
    local t1
    t1=$(date +%s)
    local dur=$((t1 - t0))

    if [ $rc -eq 0 ]; then
        echo "  >>> ${run_tag} -- OK (${dur}s)"
        ((PASS++))
        echo "${task_tag},${mode_tag},pass,${rc},${dur},${out_dir},${log}" >> "${SUMMARY_FILE}"
    else
        echo "  >>> ${run_tag} -- FAILED (exit ${rc}, ${dur}s)" >&2
        ((FAIL++))
        echo "${task_tag},${mode_tag},fail,${rc},${dur},${out_dir},${log}" >> "${SUMMARY_FILE}"
    fi
}

# ── Main loop ───────────────────────────────────────────────────────────────
for task_spec in "${TASKS[@]}"; do
    IFS='|' read -r task_tag data_ov extra_ov <<< "${task_spec}"

    if [[ -n "$TASK_FILTER" && "$task_tag" != "$TASK_FILTER" ]]; then
        continue
    fi

    for mode in "${MODES[@]}"; do
        if [[ -n "$MODE_FILTER" && "$mode" != "$MODE_FILTER" ]]; then
            continue
        fi
        mov=$(mode_overrides "${mode}")
        run "${task_tag}" "${mode}" "${COMMON} ${data_ov} ${extra_ov} ${mov}"
        ((TOTAL++))
    done
done

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Baseline sweep complete"
echo "    Sweep dir: ${SWEEP_DIR}"
echo "    Summary:   ${SUMMARY_FILE}"
echo "    Total: ${TOTAL}  Pass: ${PASS}  Fail: ${FAIL}  Skip: ${SKIP}"
echo "================================================================"

[ "${FAIL}" -eq 0 ]
