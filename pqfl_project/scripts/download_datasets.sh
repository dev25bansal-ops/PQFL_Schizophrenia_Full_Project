#!/bin/bash
# =============================================================================
# PQFL Schizophrenia Dataset Download Script (ALL datasets, LA5c excluded)
# =============================================================================
#
# This script downloads ALL publicly accessible schizophrenia fMRI datasets
# for the PQFL federated quantum learning project.
# LA5c is EXCLUDED since it is already downloaded and processed.
#
# Datasets auto-downloaded (OPEN ACCESS):
#   1. TCP 2025    - OpenNeuro ds005237 (40 SZ, 93 HC)  ~50 GB
#   2. SPINS       - OpenNeuro ds003011 (94 SZ, 94 HC)  ~80 GB
#
# Datasets requiring registration (instructions shown):
#   3. COBRE       - COINS Data Exchange (72 SZ, 74 HC)  ~20 GB
#   4. SRPBS FC    - BICR ATR Japan (146 SZ, 800 HC)     175 MB!
#
# Datasets requiring DUA/IRB (instructions shown):
#   5. MCIC        - COINS DUA (146 SZ, 160 HC)
#   6. BSNIP-2     - NIMH Data Archive (150 SZ, 223 HC)
#   7. FBIRN       - Contact PI (176 SZ, 186 HC)
#
# Prerequisites:
#   - AWS CLI:     pip install awscli
#   - Disk space:  ~130 GB for TCP + SPINS (resting-state only)
#
# Usage:
#   chmod +x scripts/download_datasets.sh
#   ./scripts/download_datasets.sh --data-dir ./data
#   ./scripts/download_datasets.sh --data-dir ./data --site TCP
#   ./scripts/download_datasets.sh --data-dir ./data --rest-only
# =============================================================================

set -euo pipefail

# ---- Defaults ----
DATA_DIR="./data"
SITE=""
REST_ONLY=false
DRY_RUN=false

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --site)
            SITE="$2"
            shift 2
            ;;
        --rest-only)
            REST_ONLY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--data-dir DIR] [--site SITE] [--rest-only] [--dry-run]"
            echo ""
            echo "Options:"
            echo "  --data-dir DIR    Base directory for downloads (default: ./data)"
            echo "  --site SITE       Download only one site: TCP, SPINS, COBRE, SRPBS"
            echo "  --rest-only       Download only resting-state fMRI (saves bandwidth)"
            echo "  --dry-run         Show commands without executing"
            echo "  --help            Show this help"
            echo ""
            echo "NOTE: LA5c is EXCLUDED (already downloaded)."
            echo "  To re-download LA5c: python scripts/download_datasets.py --site LA5c"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ---- Helper functions ----
run_cmd() {
    echo ">>> $@"
    if [ "$DRY_RUN" = false ]; then
        "$@"
    fi
}

check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        echo "ERROR: AWS CLI not found. Install with: pip install awscli"
        exit 1
    fi
}

# =====================================================================
# SITE 1: TCP 2025 (Transdiagnostic Connectome Project) — ds005237
# =====================================================================
download_tcp() {
    echo ""
    echo "============================================================"
    echo "  DOWNLOADING: TCP 2025 (OpenNeuro ds005237)"
    echo "  241 subjects: 148 clinical + 93 HC (transdiagnostic)"
    echo "  HCP-quality: TR=800ms, 2mm isotropic, 6 rest runs"
    echo "  Diagnosis in participants.tsv + phenotype/demos.tsv"
    echo "============================================================"
    echo ""

    local OUTDIR="${DATA_DIR}/TCP2025"
    mkdir -p "$OUTDIR"

    check_aws_cli

    if [ "$REST_ONLY" = true ]; then
        echo "Downloading resting-state BOLD + anatomical + phenotype only..."
        echo "Note: TCP has 6 resting-state runs per subject (~2 GB/subject)"
        run_cmd aws s3 sync --no-sign-request \
            --exclude "*" \
            --include "participants.tsv" \
            --include "dataset_description.json" \
            --include "phenotype/*" \
            --include "*/anat/*" \
            --include "*/task-rest_run-*_bold.nii.gz" \
            --include "*/task-rest_run-*_bold.json" \
            --include "*/task-rest_run-*_desc-confounds_timeseries.tsv" \
            s3://openneuro.org/ds005237 \
            "$OUTDIR/"
    else
        echo "Downloading full TCP dataset (WARNING: very large, ~200+ GB)..."
        run_cmd aws s3 sync --no-sign-request \
            s3://openneuro.org/ds005237 \
            "$OUTDIR/"
    fi

    echo ""
    echo "TCP download complete. Files in: $OUTDIR"
}

# =====================================================================
# SITE 2: SPINS — ds003011
# =====================================================================
download_spins() {
    echo ""
    echo "============================================================"
    echo "  DOWNLOADING: SPINS (OpenNeuro ds003011)"
    echo "  Social Processes Initiative in Neurobiology of the"
    echo "  Schizophrenia(s) — 3 acquisition sites"
    echo "  ~188 subjects: 94 SZ + 94 HC (+ schizoaffective, bipolar)"
    echo "  HCP-quality: TR=800ms, ~750 volumes per run"
    echo "  Diagnosis in participants.tsv: 'diagnosis' column"
    echo "============================================================"
    echo ""

    local OUTDIR="${DATA_DIR}/SPINS"
    mkdir -p "$OUTDIR"

    check_aws_cli

    if [ "$REST_ONLY" = true ]; then
        echo "Downloading resting-state BOLD + anatomical + participants only..."
        run_cmd aws s3 sync --no-sign-request \
            --exclude "*" \
            --include "participants.tsv" \
            --include "dataset_description.json" \
            --include "*/anat/*" \
            --include "*/task-rest_bold.nii.gz" \
            --include "*/task-rest_bold.json" \
            --include "*/task-rest_desc-confounds_timeseries.tsv" \
            s3://openneuro.org/ds003011 \
            "$OUTDIR/"
    else
        echo "Downloading full SPINS dataset (WARNING: very large, ~80+ GB)..."
        run_cmd aws s3 sync --no-sign-request \
            s3://openneuro.org/ds003011 \
            "$OUTDIR/"
    fi

    echo ""
    echo "SPINS download complete. Files in: $OUTDIR"
}

# =====================================================================
# SITE 3: COBRE — COINS Data Exchange (MANUAL)
# =====================================================================
download_cobre() {
    echo ""
    echo "============================================================"
    echo "  COBRE — MANUAL APPLICATION REQUIRED"
    echo "============================================================"
    echo ""
    echo "  COBRE is NOT available via direct download."
    echo "  All Figshare links are 403 Forbidden (dead since 2025)."
    echo "  Not on OpenNeuro."
    echo ""
    echo "  ACCESS STEPS:"
    echo "  1. Create account at: https://coins.trendscenter.org/"
    echo "  2. Go to Data Exchange -> Browse Available Data"
    echo "  3. Drag 'Studies' to workspace -> Filter: Study Name = COBRE"
    echo "  4. Click 'Submit Request'"
    echo "  5. Data available for download within ~1 business day"
    echo ""
    echo "  Dataset: 72 SZ + 74 HC = 146 total"
    echo "  Diagnosis: COBRE_phenotypic_data.csv 'Subject Type' column"
    echo "    'Patient' (SZ) or 'Control' (HC)"
    echo ""
    echo "  After downloading, place files in: ${DATA_DIR}/COBRE/"
    echo "============================================================"
}

# =====================================================================
# SITE 4: SRPBS — BICR ATR Japan (MANUAL APPLICATION)
# =====================================================================
download_srpbs() {
    echo ""
    echo "============================================================"
    echo "  SRPBS — MANUAL APPLICATION REQUIRED"
    echo "============================================================"
    echo ""
    echo "  RECOMMENDED: Download SRPBS FC (precomputed connectivity)"
    echo "  URL: https://bicr-resource.atr.jp/srpbsfc"
    echo "  Size: Only 175.6 MB (vs 89.8 GB for raw fMRI)"
    echo "  Contains: Precomputed FC matrices + diagnosis labels"
    echo ""
    echo "  ACCESS STEPS:"
    echo "  1. Go to: https://bicr-resource.atr.jp/srpbsfc"
    echo "  2. Download and fill 'Application Form for Data Usage'"
    echo "  3. Upload signed form + register"
    echo "  4. Wait for email approval with S3 download link"
    echo ""
    echo "  Dataset: ~146 SZ + ~800 HC across 12 sites"
    echo "  FC format: .mat files with connectivity matrices"
    echo ""
    echo "  After downloading, place files in: ${DATA_DIR}/SRPBS/"
    echo "============================================================"
}

# =====================================================================
# SITES 5-7: MCIC, BSNIP-2 & FBIRN (HIGHLY RESTRICTED)
# =====================================================================
download_restricted() {
    echo ""
    echo "============================================================"
    echo "  MCIC, BSNIP-2 & FBIRN — RESTRICTED ACCESS"
    echo "============================================================"
    echo ""
    echo "  MCIC (COINS DUA, ~weeks approval):"
    echo "    1. Register at: https://coins.trendscenter.org/"
    echo "    2. Search for 'MCICShare' study, accept DUA"
    echo "    Dataset: 146 SZ + 160 HC"
    echo ""
    echo "  BSNIP-2 (NIMH Data Archive, Collection 2165):"
    echo "    1. Create NDA account: https://nda.nih.gov"
    echo "    2. Submit Data Use Certification (requires IRB)"
    echo "    3. Download: pip install nda-tools && downloadcmd -d 2165"
    echo "    Dataset: ~150 SZ + ~223 HC"
    echo ""
    echo "  FBIRN (Contact PI directly):"
    echo "    Contact: Dr. Theo G.M. van Erp (tvanerp@hs.uci.edu)"
    echo "    Dataset: Phase III: 176 SZ + 186 HC"
    echo ""
    echo "  After downloading, place files in:"
    echo "    ${DATA_DIR}/MCIC/"
    echo "    ${DATA_DIR}/BSNIP2/"
    echo "    ${DATA_DIR}/FBIRN/"
    echo "============================================================"
}

# =====================================================================
# Main
# =====================================================================
mkdir -p "$DATA_DIR"

echo "============================================================"
echo "  PQFL Schizophrenia Multi-Dataset Downloader"
echo "  (LA5c EXCLUDED — already downloaded)"
echo "  Data directory: $DATA_DIR"
echo "============================================================"

case "$SITE" in
    TCP|TCP2025)
        download_tcp
        ;;
    SPINS)
        download_spins
        ;;
    COBRE)
        download_cobre
        ;;
    SRPBS)
        download_srpbs
        ;;
    "")
        # Download all open-access sites + show instructions for restricted
        download_tcp
        download_spins
        download_cobre
        download_srpbs
        download_restricted
        ;;
    LA5c)
        echo ""
        echo "  NOTE: LA5c is already downloaded and processed."
        echo "  To re-download, use the original script:"
        echo "    python scripts/download_datasets.py --site LA5c"
        ;;
    *)
        echo "Unknown site: $SITE"
        echo "Available: TCP, SPINS, COBRE, SRPBS (LA5c excluded)"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  DOWNLOAD COMPLETE (or instructions shown)"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Check status: python scripts/check_dataset_status.py --data_dir $DATA_DIR"
echo "  2. Preprocess:   python scripts/preprocess_all_datasets.py --data_dir $DATA_DIR --compute_fdt"
echo "  3. Integrate:    python scripts/integrate_datasets.py --data_dir $DATA_DIR/processed"
echo "  4. Train:        python experiments/final_training.py --data_dir $DATA_DIR/processed"
