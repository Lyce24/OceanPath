#!/usr/bin/env bash
# ── run_label_efficiency.sh ─────────────────────────────────────────────────
#
# Label-efficiency sweep for linear probes:
#   default fractions: 1%, 5%, 10%, 25%, 50%, 100%
#   default seeds:     42, 43, 44   (multi-seed Monte Carlo over subsamples)
#   tasks:             same downstream tasks as run_lp_baselines.sh
#   modes:             pooling baselines, random aggregator, optional SSL ckpt
#
# Only the training side is label-subsampled. Validation folds and external
# test cohorts remain full-size. Embedding mmaps are cached and reused across
# fractions and seeds because the sweep only changes probe fitting, not
# extraction.
#
# Each (task, mode, fraction, seed) lands in its own output dir so seed-level
# variance can be aggregated by scripts/collect_lp_results.py.
#
# Usage
#   bash scripts/run_label_efficiency.sh --dry-run
#   bash scripts/run_label_efficiency.sh --ssl-ckpt outputs/pretrain/.../best.ckpt
#   bash scripts/run_label_efficiency.sh --task nsclc_xfer --mode ssl --ssl-ckpt CKPT
#   bash scripts/run_label_efficiency.sh --fraction 0.01 --mode mean_max_pool
#   bash scripts/run_label_efficiency.sh --seeds "42 43 44 45 46"
#   bash scripts/run_label_efficiency.sh --fractions "0.01 0.10 1.00"
#
# Inner CV is enabled by default (3-fold, full C grid). The probe module
# automatically falls back to the C-grid midpoint when a fraction's subsample
# is too thin to honor inner_splits, so the same protocol is consistent across
# fractions.
#   bash scripts/run_label_efficiency.sh --inner-splits 0 --c-grid '[1.0]'
# disables tuning entirely (useful for quick sanity checks).
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DRY_RUN=false
TASK_FILTER=""
MODE_FILTER=""
FRACTION_FILTER=""
SSL_CKPT="${SSL_CKPT:-}"
ENCODER="${ENCODER:-uni2h}"
N_BOOT="${N_BOOT:-0}"
INNER_SPLITS="${INNER_SPLITS:-3}"
C_GRID="${C_GRID:-[0.01,0.1,1.0,10.0]}"
SEEDS_RAW="${SEEDS:-42 43 44}"
FRACTIONS_RAW="${FRACTIONS:-0.01 0.05 0.10 0.25 0.50 1.00}"

usage() {
    cat <<EOF
Usage: $0 [--dry-run] [--task TASK] [--mode MODE] [--fraction FRACTION]
          [--ssl-ckpt CKPT] [--encoder ENCODER] [--n-boot N]
          [--inner-splits N] [--c-grid HYDRA_LIST]
          [--seeds "S1 S2 ..."] [--fractions "F1 F2 ..."]

Tasks:     nsclc_cv  nsclc_xfer  braf  panda  gej
Modes:     mean_pool  max_pool  mean_max_pool  aggregator_random  ssl
Defaults:  fractions="${FRACTIONS_RAW}"
           seeds="${SEEDS_RAW}"
           inner_splits=${INNER_SPLITS}  c_grid=${C_GRID}

SSL checkpoint can also be supplied via SSL_CKPT=/path/to/best.ckpt.
SEEDS and FRACTIONS env vars accept space-separated lists.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)       DRY_RUN=true; shift ;;
        --task)          TASK_FILTER="${2:-}"; shift 2 ;;
        --mode)          MODE_FILTER="${2:-}"; shift 2 ;;
        --fraction)      FRACTION_FILTER="${2:-}"; shift 2 ;;
        --ssl-ckpt)      SSL_CKPT="${2:-}"; shift 2 ;;
        --encoder)       ENCODER="${2:-}"; shift 2 ;;
        --n-boot)        N_BOOT="${2:-}"; shift 2 ;;
        --inner-splits)  INNER_SPLITS="${2:-}"; shift 2 ;;
        --c-grid)        C_GRID="${2:-}"; shift 2 ;;
        --seeds)         SEEDS_RAW="${2:-}"; shift 2 ;;
        --fractions)     FRACTIONS_RAW="${2:-}"; shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *)               echo "Unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

# Parse space-separated lists into bash arrays. Commas are tolerated.
read -r -a SEEDS <<< "${SEEDS_RAW//,/ }"
read -r -a FRACTIONS <<< "${FRACTIONS_RAW//,/ }"

if [[ ${#SEEDS[@]} -eq 0 ]]; then
    echo "ERROR: no seeds provided." >&2
    exit 2
fi
if [[ ${#FRACTIONS[@]} -eq 0 ]]; then
    echo "ERROR: no fractions provided." >&2
    exit 2
fi

# ── Validate filters (fail fast on typos) ───────────────────────────────────
VALID_TASKS=" nsclc_cv nsclc_xfer braf panda gej "
VALID_MODES=" mean_pool max_pool mean_max_pool aggregator_random ssl "
if [[ -n "$TASK_FILTER" && "$VALID_TASKS" != *" $TASK_FILTER "* ]]; then
    echo "ERROR: unknown --task '$TASK_FILTER'. Valid:${VALID_TASKS}" >&2
    exit 2
fi
if [[ -n "$MODE_FILTER" && "$VALID_MODES" != *" $MODE_FILTER "* ]]; then
    echo "ERROR: unknown --mode '$MODE_FILTER'. Valid:${VALID_MODES}" >&2
    exit 2
fi
if [[ -n "$SSL_CKPT" && ! -f "$SSL_CKPT" ]]; then
    echo "ERROR: --ssl-ckpt file not found: $SSL_CKPT" >&2
    exit 2
fi

# Numeric fraction comparison (so --fraction 0.1 matches a list entry of 0.10).
fraction_eq() {
    awk -v a="$1" -v b="$2" 'BEGIN { exit !(a+0 == b+0) }'
}

fraction_tag() {
    awk -v f="$1" 'BEGIN {
        pct = f * 100.0
        rounded = int(pct + (pct >= 0 ? 0.5 : -0.5))
        if (pct - rounded > -1e-9 && pct - rounded < 1e-9) {
            printf "%dpct", rounded
        } else {
            s = sprintf("%g", pct)
            gsub(/\./, "p", s)
            printf "%spct", s
        }
    }'
}

mode_overrides() {
    case "$1" in
        mean_pool)
            echo "embedding.mode=mean_pool"
            ;;
        max_pool)
            echo "embedding.mode=max_pool"
            ;;
        mean_max_pool)
            echo "embedding.mode=mean_max_pool"
            ;;
        aggregator_random)
            echo "model=mamba2mil embedding.mode=aggregator embedding.ckpt=null"
            ;;
        ssl)
            if [[ -z "$SSL_CKPT" ]]; then
                echo "ERROR: mode=ssl requires --ssl-ckpt or SSL_CKPT." >&2
                return 1
            fi
            echo "model=mamba2mil embedding.mode=aggregator embedding.ckpt=${SSL_CKPT}"
            ;;
        *)
            echo "ERROR: unknown mode $1" >&2
            return 1
            ;;
    esac
}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${PROJECT_ROOT}/outputs/label_efficiency_2048/${TIMESTAMP}"
LOG_DIR="${SWEEP_DIR}/logs"
mkdir -p "$LOG_DIR"

SUMMARY_FILE="${SWEEP_DIR}/sweep_summary.csv"
echo "task,mode,fraction,seed,status,exit_code,duration_s,output_dir,log" > "${SUMMARY_FILE}"

echo "================================================================"
echo "  Label-efficiency LP sweep"
echo "  Sweep dir:       ${SWEEP_DIR}"
echo "  Filters:         task=${TASK_FILTER:-<all>} mode=${MODE_FILTER:-<all>} fraction=${FRACTION_FILTER:-<all>}"
echo "  Encoder:         ${ENCODER}"
echo "  Probe tuning:    inner_splits=${INNER_SPLITS} c_grid=${C_GRID}"
echo "  Bootstrap:       n_boot=${N_BOOT}"
echo "  Fractions:       ${FRACTIONS[*]}"
echo "  Seeds:           ${SEEDS[*]}"
echo "  SSL checkpoint:  ${SSL_CKPT:-<none; ssl mode skipped unless provided>}"
echo "================================================================"

CPTAC_MANIFEST=/mnt/d/YC.Liu/manifests/SSL/cptac_nsclc_labels.csv
TASKS=(
    "nsclc_cv|data=tcga_nsclc|protocol=grouped_cv"
    "nsclc_xfer|data=tcga_nsclc|protocol=external_test test.mmap_dir=./mmap/cptac_nsclc test.manifest_csv=${CPTAC_MANIFEST}"
    "braf|data=TCGA_BRAF|protocol=grouped_cv"
    "panda|data=panda|protocol=grouped_cv primary_metric=qwk"
    "gej|data=gej_univ2|protocol=grouped_cv aggregate_to_patient=false primary_metric=qwk"
)
MODES=(mean_pool max_pool mean_max_pool aggregator_random ssl)

PASS=0; FAIL=0; SKIP=0; TOTAL=0

run() {
    local task_tag="$1"
    local mode_tag="$2"
    local fraction="$3"
    local seed="$4"
    local overrides="$5"
    local ftag
    ftag=$(fraction_tag "$fraction")
    local run_tag="${task_tag}__${mode_tag}__${ftag}__seed${seed}"
    local log="${LOG_DIR}/${run_tag}.log"
    local out_dir="${SWEEP_DIR}/frac_${ftag}/${task_tag}__${mode_tag}/seed_${seed}"
    local common="platform=local encoder=${ENCODER} n_boot=${N_BOOT} inner_splits=${INNER_SPLITS} c_grid=${C_GRID} label_fraction_seed=${seed} seed=${seed}"

    echo ""
    echo "================================================================"
    echo "  ${run_tag}"
    echo "  log: ${log}"
    echo "  out: ${out_dir}"
    echo "  cmd: python scripts/linear_probing.py ${common} ${overrides} label_fraction=${fraction} output_dir=${out_dir}"
    echo "================================================================"

    if $DRY_RUN; then
        echo "  [dry run -- skipped]"
        ((SKIP++))
        echo "${task_tag},${mode_tag},${fraction},${seed},dry_run,0,0,${out_dir},${log}" >> "${SUMMARY_FILE}"
        return 0
    fi

    local t0
    t0=$(date +%s)
    local rc=0
    # shellcheck disable=SC2086
    python scripts/linear_probing.py ${common} ${overrides} label_fraction="${fraction}" output_dir="${out_dir}" 2>&1 | tee "${log}" || rc=$?
    local t1
    t1=$(date +%s)
    local dur=$((t1 - t0))

    if [ $rc -eq 0 ]; then
        echo "  >>> ${run_tag} -- OK (${dur}s)"
        ((PASS++))
        echo "${task_tag},${mode_tag},${fraction},${seed},pass,${rc},${dur},${out_dir},${log}" >> "${SUMMARY_FILE}"
    else
        echo "  >>> ${run_tag} -- FAILED (exit ${rc}, ${dur}s)" >&2
        ((FAIL++))
        echo "${task_tag},${mode_tag},${fraction},${seed},fail,${rc},${dur},${out_dir},${log}" >> "${SUMMARY_FILE}"
    fi
}

for task_spec in "${TASKS[@]}"; do
    IFS='|' read -r task_tag data_ov extra_ov <<< "${task_spec}"

    if [[ -n "$TASK_FILTER" && "$task_tag" != "$TASK_FILTER" ]]; then
        continue
    fi

    for fraction in "${FRACTIONS[@]}"; do
        if [[ -n "$FRACTION_FILTER" ]] && ! fraction_eq "$fraction" "$FRACTION_FILTER"; then
            continue
        fi

        for mode in "${MODES[@]}"; do
            if [[ -n "$MODE_FILTER" && "$mode" != "$MODE_FILTER" ]]; then
                continue
            fi
            if [[ "$mode" == "ssl" && -z "$SSL_CKPT" ]]; then
                ftag=$(fraction_tag "$fraction")
                echo "  >>> skipping ${task_tag}__ssl__${ftag} (no SSL checkpoint)"
                SKIP=$((SKIP + ${#SEEDS[@]}))
                TOTAL=$((TOTAL + ${#SEEDS[@]}))
                continue
            fi

            mov=$(mode_overrides "${mode}") || {
                FAIL=$((FAIL + ${#SEEDS[@]}))
                TOTAL=$((TOTAL + ${#SEEDS[@]}))
                continue
            }
            # At fraction=1.0 the per-seed subsample is the full training set
            # and downstream LR is deterministic, so seeds beyond the first
            # produce identical summaries. Run only the first to save time.
            full_data=false
            if fraction_eq "$fraction" "1.0"; then full_data=true; fi
            for seed in "${SEEDS[@]}"; do
                if $full_data && [[ "$seed" != "${SEEDS[0]}" ]]; then
                    ftag=$(fraction_tag "$fraction")
                    echo "  >>> skipping ${task_tag}__${mode}__${ftag}__seed${seed} (redundant at fraction=1.0)"
                    ((SKIP++))
                    ((TOTAL++))
                    continue
                fi
                run "${task_tag}" "${mode}" "${fraction}" "${seed}" "${data_ov} ${extra_ov} ${mov}"
                ((TOTAL++))
            done
        done
    done
done

echo ""
echo "================================================================"
echo "  Label-efficiency sweep complete"
echo "    Sweep dir: ${SWEEP_DIR}"
echo "    Summary:   ${SUMMARY_FILE}"
echo "    Total: ${TOTAL}  Pass: ${PASS}  Fail: ${FAIL}  Skip: ${SKIP}"
echo "================================================================"

[ "${FAIL}" -eq 0 ]
