#!/usr/bin/env bash
# ── run_supervised_braf_nsclc.sh ─────────────────────────────────────────────
#
# Two tasks × two models = 4 experiments:
#
#   1. TCGA→CPTAC NSCLC transfer  (5-fold on TCGA, external test on CPTAC)
#   2. BRAF mutation prediction    (5-fold CV)
#
#   × ABMIL from scratch
#   × Random-init Mamba2MIL
#
# Prerequisites (run once):
#   bash scripts/run_supervised_braf_nsclc.sh --make-splits
#
# Then run experiments:
#   bash scripts/run_supervised_braf_nsclc.sh
#   bash scripts/run_supervised_braf_nsclc.sh --dry-run
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

ACTION="${1:---run}"

# ── Paths ────────────────────────────────────────────────────────────────────
CPTAC_MANIFEST=/mnt/d/YC.Liu/manifests/SSL/cptac_nsclc_labels.csv
CPTAC_MMAP=./mmap/cptac_nsclc

# ─────────────────────────────────────────────────────────────────────────────
# Step 0: Generate splits (run once with --make-splits)
# ─────────────────────────────────────────────────────────────────────────────

make_splits() {
    echo ""
    echo "================================================================"
    echo "  Generating patient-level 5-fold splits"
    echo "================================================================"
    echo ""

    # TCGA NSCLC — 5-fold CV (train/val only, no held-out test)
    # Stratified by label, grouped by patient_id
    # force=true to overwrite stale splits from before the slide_id normalization fix
    echo ">>> tcga_nsclc / kfold5"
    python scripts/make_splits.py \
        platform=local \
        data=tcga_nsclc \
        encoder=uni2h \
        splits=kfold5 \
        force=true

    # TCGA BRAF — 5-fold CV
    echo ""
    echo ">>> TCGA_BRAF / kfold5"
    python scripts/make_splits.py \
        platform=local \
        data=TCGA_BRAF \
        encoder=uni2h \
        splits=kfold5 \
        force=true

    echo ""
    echo "Splits written to:"
    echo "  /mnt/d/YC.Liu/splits/tcga_nsclc/kfold5_seed42/"
    echo "  /mnt/d/YC.Liu/splits/TCGA_BRAF/kfold5_seed42/"
    echo ""
    echo "Verify with:"
    echo "  python -c \"import pandas as pd; df=pd.read_parquet('/mnt/d/YC.Liu/splits/tcga_nsclc/kfold5_seed42/splits.parquet'); print(df['fold'].value_counts().sort_index())\""
    echo "  python -c \"import pandas as pd; df=pd.read_parquet('/mnt/d/YC.Liu/splits/TCGA_BRAF/kfold5_seed42/splits.parquet'); print(df['fold'].value_counts().sort_index())\""
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Run experiments
# ─────────────────────────────────────────────────────────────────────────────

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs/sup_braf_nsclc_${TIMESTAMP}"

COMMON="platform=local encoder=uni2h splits=kfold5 training=supervised_mil"

# No extra overrides needed for BRAF — fixed-N batching means length_bucket
# is already off, and class_weighted_sampling is on by default.
BRAF_EXTRA=""

PASS=0; FAIL=0; SKIP=0; DRY_RUN=false
[[ "$ACTION" == "--dry-run" ]] && DRY_RUN=true

run() {
    local tag="$1" overrides="$2"
    local log="${LOG_DIR}/${tag}.log"

    echo ""
    echo "================================================================"
    echo "  ${tag}"
    echo "  cmd: python scripts/train_supervised.py ${overrides}"
    echo "================================================================"

    if $DRY_RUN; then
        echo "  [dry run -- skipped]"
        ((SKIP++))
        return 0
    fi

    local rc=0
    python scripts/train_supervised.py ${overrides} 2>&1 | tee "${log}" || rc=$?

    if [ $rc -eq 0 ]; then echo "  >>> ${tag} -- OK"; ((PASS++))
    else echo "  >>> ${tag} -- FAILED (exit ${rc})" >&2; ((FAIL++)); fi
}

run_experiments() {
    mkdir -p "$LOG_DIR"

    # ── Task 1: TCGA→CPTAC NSCLC transfer ───────────────────────────────
    # 5-fold on TCGA for train/val, external test on CPTAC after each fold.

    NSCLC_BASE="${COMMON} data=tcga_nsclc test_data.mmap_dir=${CPTAC_MMAP} test_data.manifest_csv=${CPTAC_MANIFEST}"

    run "nsclc_xfer__abmil" \
        "${NSCLC_BASE} model=abmil"

    run "nsclc_xfer__mamba2mil" \
        "${NSCLC_BASE} model=mamba2mil training.return_coords=true"

    # ── Task 2: BRAF mutation 5-fold CV ──────────────────────────────────

    BRAF_BASE="${COMMON} data=TCGA_BRAF ${BRAF_EXTRA}"

    run "braf__abmil" \
        "${BRAF_BASE} model=abmil"

    run "braf__mamba2mil" \
        "${BRAF_BASE} model=mamba2mil training.return_coords=true"

    # ── Summary ──────────────────────────────────────────────────────────
    echo ""
    echo "================================================================"
    echo "  Done. Pass: ${PASS}  Fail: ${FAIL}  Skip: ${SKIP}"
    echo "  Logs: ${LOG_DIR}/"
    echo "================================================================"
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

case "$ACTION" in
    --make-splits)  make_splits ;;
    --run|--dry-run) run_experiments ;;
    *)
        echo "Usage:"
        echo "  $0 --make-splits    # generate kfold5 splits (run once)"
        echo "  $0                  # run all 4 experiments"
        echo "  $0 --dry-run        # print commands only"
        exit 1
        ;;
esac
