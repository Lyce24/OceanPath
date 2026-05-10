#!/usr/bin/env bash
# ── run_lp_ssl.sh ───────────────────────────────────────────────────────────
#
# Linear-probing sweep with pretrained Mamba2MIL aggregator weights from
# three SSL pretraining objectives:
#
#   5 tasks  x  3 SSL ckpts  =  15 experiments
#
# Mirrors run_lp_baselines.sh: same fixed-cap regime (max_instances=2048,
# sampling_mode=spatial_stratified) and same task list. Each ckpt is
# extracted independently — the cache key includes the ckpt content hash,
# so different ckpts never silently share embeddings.
#
# Outputs land under:
#   outputs/lp_2048/<timestamp>_ssl/<task>__aggregator_<ssl>/<cache_tag>/
#
# Tasks (same as run_lp_baselines.sh)
#   nsclc_cv    NSCLC subtyping (TCGA LUAD vs LUSC)         grouped_cv     binary
#   nsclc_xfer  NSCLC cross-cohort (TCGA -> CPTAC)          external_test  binary
#   braf        BRAF mutation prediction (TCGA CRC)         grouped_cv     binary
#   panda       PANDA prostate ISUP grading (0..5)          grouped_cv     ordinal (qwk)
#   gej         GEJ Barrett's per-section severity (0..3)   grouped_cv     ordinal (qwk)
#
# SSL ckpts (under checkpoints/)
#   vicreg     aggregator_weights_vicreg.pt
#   jepa       aggregator_weights_jepa.pt
#   lejepa     aggregator_weights_lejepa.pt
#   lejepa_mc  aggregator_weights_lejepa_mc.pt   (multi-crop variant)
#
# Usage
#   bash scripts/run_lp_ssl.sh                  # run all 15
#   bash scripts/run_lp_ssl.sh --dry-run        # print commands only
#   bash scripts/run_lp_ssl.sh --task gej       # one task, all ckpts
#   bash scripts/run_lp_ssl.sh --ssl vicreg     # all tasks, one ckpt
#   bash scripts/run_lp_ssl.sh --task panda --ssl lejepa
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Argument parsing ────────────────────────────────────────────────────────
DRY_RUN=false
TASK_FILTER=""
SSL_FILTER=""

usage() {
    cat <<EOF
Usage: $0 [--dry-run] [--task TASK] [--ssl SSL]

Tasks: nsclc_cv  nsclc_xfer  braf  panda  gej
SSL:   vicreg    jepa        lejepa
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true; shift ;;
        --task)     TASK_FILTER="${2:-}"; shift 2 ;;
        --ssl)      SSL_FILTER="${2:-}"; shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *)          echo "Unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

# ── Validate filters (fail fast on typos) ───────────────────────────────────
VALID_TASKS=" nsclc_cv nsclc_xfer braf panda gej "
VALID_SSLS=" vicreg jepa lejepa lejepa_mc "
if [[ -n "$TASK_FILTER" && "$VALID_TASKS" != *" $TASK_FILTER "* ]]; then
    echo "ERROR: unknown --task '$TASK_FILTER'. Valid:${VALID_TASKS}" >&2
    exit 2
fi
if [[ -n "$SSL_FILTER" && "$VALID_SSLS" != *" $SSL_FILTER "* ]]; then
    echo "ERROR: unknown --ssl '$SSL_FILTER'. Valid:${VALID_SSLS}" >&2
    exit 2
fi

# ── Sweep directories ───────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${PROJECT_ROOT}/outputs/lp_2048/${TIMESTAMP}_ssl"
LOG_DIR="${SWEEP_DIR}/logs"
mkdir -p "$LOG_DIR"

SUMMARY_FILE="${SWEEP_DIR}/sweep_summary.csv"
echo "task,ssl,status,exit_code,duration_s,output_dir,log" > "${SUMMARY_FILE}"

echo "================================================================"
echo "  LP SSL sweep (Mamba2MIL aggregator)"
echo "  Sweep dir: ${SWEEP_DIR}"
echo "  Filters:   task=${TASK_FILTER:-<all>} ssl=${SSL_FILTER:-<all>}"
echo "================================================================"

# ── Shared overrides ────────────────────────────────────────────────────────
ENCODER=uni2h
N_BOOT=1000
COMMON="platform=local encoder=${ENCODER} model=mamba2mil embedding.mode=aggregator n_boot=${N_BOOT}"

# ── SSL ckpts ───────────────────────────────────────────────────────────────
ssl_ckpt() {
    case "$1" in
        vicreg)     echo "checkpoints/aggregator_weights_vicreg.pt" ;;
        jepa)       echo "checkpoints/aggregator_weights_jepa.pt" ;;
        lejepa)     echo "checkpoints/aggregator_weights_lejepa.pt" ;;
        lejepa_mc)  echo "checkpoints/aggregator_weights_lejepa_mc.pt" ;;
        *) echo "ERROR: unknown ssl $1" >&2; return 1 ;;
    esac
}

SSLS=(vicreg jepa lejepa lejepa_mc)

# ── Pre-flight: ensure ckpt files exist for the SSLs we plan to run ─────────
for ssl in "${SSLS[@]}"; do
    if [[ -n "$SSL_FILTER" && "$ssl" != "$SSL_FILTER" ]]; then continue; fi
    ckpt=$(ssl_ckpt "${ssl}")
    if [[ ! -f "$ckpt" ]]; then
        echo "ERROR: ssl=${ssl} ckpt not found: ${ckpt}" >&2
        echo "Place the file under checkpoints/ or filter with --ssl <other>." >&2
        exit 2
    fi
done

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
    local task_tag="$1" ssl_tag="$2" overrides="$3"
    local run_tag="${task_tag}__aggregator_${ssl_tag}"
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
        echo "${task_tag},${ssl_tag},dry_run,0,0,${out_dir},${log}" >> "${SUMMARY_FILE}"
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
        echo "${task_tag},${ssl_tag},pass,${rc},${dur},${out_dir},${log}" >> "${SUMMARY_FILE}"
    else
        echo "  >>> ${run_tag} -- FAILED (exit ${rc}, ${dur}s)" >&2
        ((FAIL++))
        echo "${task_tag},${ssl_tag},fail,${rc},${dur},${out_dir},${log}" >> "${SUMMARY_FILE}"
    fi
}

# ── Main loop ───────────────────────────────────────────────────────────────
for task_spec in "${TASKS[@]}"; do
    IFS='|' read -r task_tag data_ov extra_ov <<< "${task_spec}"

    if [[ -n "$TASK_FILTER" && "$task_tag" != "$TASK_FILTER" ]]; then
        continue
    fi

    for ssl in "${SSLS[@]}"; do
        if [[ -n "$SSL_FILTER" && "$ssl" != "$SSL_FILTER" ]]; then
            continue
        fi
        ckpt=$(ssl_ckpt "${ssl}")
        ssl_ov="embedding.ckpt=${ckpt}"
        run "${task_tag}" "${ssl}" "${COMMON} ${data_ov} ${extra_ov} ${ssl_ov}"
        ((TOTAL++))
    done
done

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SSL sweep complete"
echo "    Sweep dir: ${SWEEP_DIR}"
echo "    Summary:   ${SUMMARY_FILE}"
echo "    Total: ${TOTAL}  Pass: ${PASS}  Fail: ${FAIL}  Skip: ${SKIP}"
echo "================================================================"

# ── Aggregate results CSV ───────────────────────────────────────────────────
CSV_OUT="${PROJECT_ROOT}/outputs/lp_ssl_${TIMESTAMP}.csv"
if ! $DRY_RUN && [ "${PASS}" -gt 0 ]; then
    echo ""
    echo "Collecting metrics into ${CSV_OUT}"
    python scripts/collect_lp_results.py --root "${SWEEP_DIR}" --csv "${CSV_OUT}" || \
        echo "  >>> CSV aggregation failed (sweep results still in ${SWEEP_DIR})" >&2
fi

[ "${FAIL}" -eq 0 ]
