#!/bin/bash
# =============================================================================
# PQFL Schizophrenia Dataset Download Script
# =============================================================================
#
# This script downloads all publicly accessible schizophrenia fMRI datasets
# for the PQFL federated quantum learning project.
#
# Datasets downloaded:
#   1. LA5c/CNP   - OpenNeuro ds000030 (OPEN, no registration)
#   2. TCP 2025    - OpenNeuro ds005237 (OPEN, no registration)
#
# Datasets requiring separate application:
#   3. COBRE       - COINS Data Exchange (free account, ~1 day approval)
#   4. SRPBS FC    - BICR ATR Japan (application form, ~days)
#   5. MCIC        - COINS Data Exchange (DUA, ~weeks)
#   6. BSNIP-2     - NIMH Data Archive (IRB + DUC, ~weeks-months)
#   7. FBIRN       - Contact PI directly (months)
#
# Prerequisites:
#   - AWS CLI:     pip install awscli  (or: conda install -c conda-forge awscli)
#   - DataLad:     pip install datalad  (alternative method)
#   - Disk space:  ~30 GB for LA5c + TCP (resting-state only)
#
# Usage:
#   chmod +x scripts/download_datasets.sh
#   ./scripts/download_datasets.sh --data-dir ./data
#   ./scripts/download_datasets.sh --data-dir ./data --site LA5c
#   ./scripts/download_datasets.sh --data-dir ./data --rest-only   # smaller downloads
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
            echo "  --site SITE       Download only one site: LA5c, TCP, COBRE, SRPBS"
            echo "  --rest-only       Download only resting-state fMRI (saves bandwidth)"
            echo "  --dry-run         Show commands without executing"
            echo "  --help            Show this help"
            echo ""
            echo "Available open-access sites: LA5c, TCP"
            echo "Restricted-access sites (manual): COBRE, SRPBS, MCIC, BSNIP2, FBIRN"
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
        echo "  Or use DataLad: pip install datalad"
        exit 1
    fi
}

check_datalad() {
    if ! command -v datalad &> /dev/null; then
        echo "WARNING: DataLad not found. Install with: pip install datalad"
        return 1
    fi
    return 0
}

# =====================================================================
# SITE 1: LA5c/CNP — OpenNeuro ds000030
# =====================================================================
download_la5c() {
    echo ""
    echo "============================================================"
    echo "  DOWNLOADING: LA5c/CNP (OpenNeuro ds000030)"
    echo "  272 subjects: 50 SCHZ + 127 CONTROL + 49 BIPOLAR + 43 ADHD"
    echo "  Resting-state fMRI: TR=2s, ~156 volumes per run"
    echo "  Diagnosis in participants.tsv: 'diagnosis' column"
    echo "============================================================"
    echo ""

    local OUTDIR="${DATA_DIR}/LA5c"
    mkdir -p "$OUTDIR"

    check_aws_cli

    if [ "$REST_ONLY" = true ]; then
        echo "Downloading resting-state BOLD + anatomical + participants only..."
        echo "This saves ~80% bandwidth vs full dataset."
        run_cmd aws s3 sync --no-sign-request \
            --exclude "*" \
            --include "participants.tsv" \
            --include "dataset_description.json" \
            --include "*/anat/*" \
            --include "*/task-rest_bold.nii.gz" \
            --include "*/task-rest_bold.json" \
            --include "*/task-rest_physio.tsv.gz" \
            --include "*/task-rest_physio.json" \
            s3://openneuro/ds000030/ds000030_R1.0.5/uncompressed/ \
            "$OUTDIR/"
    else
        echo "Downloading full LA5c dataset..."
        run_cmd aws s3 sync --no-sign-request \
            s3://openneuro/ds000030/ds000030_R1.0.5/uncompressed/ \
            "$OUTDIR/"
    fi

    echo ""
    echo "LA5c download complete. Files in: $OUTDIR"
    echo "Verify: Check $OUTDIR/participants.tsv has 'diagnosis' column"
    echo "  SZ subjects:   diagnosis == 'SCHZ'"
    echo "  HC subjects:   diagnosis == 'CONTROL'"
}

# =====================================================================
# SITE 2: TCP 2025 (Transdiagnostic Connectome Project) — ds005237
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
    echo "Verify: Check $OUTDIR/participants.tsv and $OUTDIR/phenotype/demos.tsv"
    echo "  Filter by SCID diagnosis to isolate SZ subjects"
    echo "  PANSS scores also available for symptom severity"
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
    echo "  Resting-state fMRI: 150 volumes, TR=2s"
    echo "  Diagnosis in: COBRE_phenotypic_data.csv"
    echo "    'Subject Type' column: 'Patient' (SZ) or 'Control' (HC)"
    echo ""
    echo "  After downloading, place files in: ${DATA_DIR}/COBRE/"
    echo "  Expected structure:"
    echo "    ${DATA_DIR}/COBRE/"
    echo "    ├── participants.tsv (or COBRE_phenotypic_data.csv)"
    echo "    ├── sub-01/"
    echo "    │   ├── func/sub-01_task-rest_bold.nii.gz"
    echo "    │   └── anat/sub-01_T1w.nii.gz"
    echo "    └── ..."
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
    echo "  ALTERNATIVE: Raw fMRI (SRPBS-1600)"
    echo "  URL: https://bicr-resource.atr.jp/srpbs1600"
    echo "  Size: 89.8 GB"
    echo ""
    echo "  ACCESS STEPS:"
    echo "  1. Go to: https://bicr-resource.atr.jp/srpbsfc"
    echo "  2. Download the 'Application Form for Data Usage' (PDF/Word)"
    echo "  3. Fill in and sign the form"
    echo "  4. Upload with registration"
    echo "  5. Wait for email approval with S3 download link"
    echo ""
    echo "  Dataset: ~146 SZ + ~1,421 HC across 12 sites"
    echo "  FC data format: .mat files with connectivity matrices"
    echo "  Diagnosis: participants.tsv or SUBINFO_*.tsv per site"
    echo "    SZ labels: 'SCZ' or 'SZ'"
    echo "    HC labels: 'HC' or 'Control'"
    echo ""
    echo "  After downloading, place files in: ${DATA_DIR}/SRPBS/"
    echo "  Expected structure (FC data):"
    echo "    ${DATA_DIR}/SRPBS/"
    echo "    ├── participants.tsv"
    echo "    ├── fc_matrices.npz (or .mat files)"
    echo "    ├── SUBINFO_*.tsv"
    echo "    └── ..."
    echo "============================================================"
}

# =====================================================================
# SITE 5: MCIC — COINS Data Exchange (MANUAL)
# =====================================================================
download_mcic() {
    echo ""
    echo "============================================================"
    echo "  MCIC — MANUAL APPLICATION REQUIRED"
    echo "============================================================"
    echo ""
    echo "  ACCESS STEPS:"
    echo "  1. Create account at: https://coins.trendscenter.org/"
    echo "  2. Search for 'MCICShare' study"
    echo "  3. Select scan series: resting-state fMRI + structural"
    echo "  4. Accept MCIC Data Use Agreement"
    echo "  5. Wait for download approval (may take 1+ month)"
    echo ""
    echo "  Dataset: 146 SZ + 160 HC = 306 total (3 sites released)"
    echo "  Resting-state fMRI + task fMRI + structural + DWI"
    echo "  SCID-confirmed diagnosis"
    echo ""
    echo "  After downloading, place files in: ${DATA_DIR}/MCIC/"
    echo "============================================================"
}

# =====================================================================
# SITES 6-7: BSNIP-2 & FBIRN (HIGHLY RESTRICTED)
# =====================================================================
download_restricted() {
    echo ""
    echo "============================================================"
    echo "  BSNIP-2 & FBIRN — RESTRICTED ACCESS (WEEKS-MONTHS)"
    echo "============================================================"
    echo ""
    echo "  BSNIP-2 (NIMH Data Archive, Collection 2165):"
    echo "    1. Create NDA account: https://nda.nih.gov"
    echo "    2. Submit Data Use Certification (requires IRB documentation)"
    echo "    3. Wait for Data Access Committee approval"
    echo "    4. Download: pip install nda-tools && downloadcmd -d 2165"
    echo "    Dataset: ~150 SZ + ~223 HC"
    echo ""
    echo "  FBIRN (Contact PI directly):"
    echo "    Contact: Dr. Theo G.M. van Erp"
    echo "    Email: tvanerp@hs.uci.edu"
    echo "    Must facilitate interaction with IRB + sign DUA"
    echo "    Dataset: Phase III has 176 SZ + 186 HC with rest fMRI"
    echo ""
    echo "  After downloading, place files in:"
    echo "    ${DATA_DIR}/BSNIP2/"
    echo "    ${DATA_DIR}/FBIRN/"
    echo "============================================================"
}

# =====================================================================
# Main
# =====================================================================
mkdir -p "$DATA_DIR"

echo "============================================================"
echo "  PQFL Schizophrenia Dataset Downloader"
echo "  Data directory: $DATA_DIR"
echo "============================================================"

case "$SITE" in
    LA5c)
        download_la5c
        ;;
    TCP)
        download_tcp
        ;;
    COBRE)
        download_cobre
        ;;
    SRPBS)
        download_srpbs
        ;;
    MCIC)
        download_mcic
        ;;
    "")
        # Download all open-access sites + show instructions for restricted
        download_la5c
        download_tcp
        download_cobre
        download_srpbs
        download_mcic
        download_restricted
        ;;
    *)
        echo "Unknown site: $SITE"
        echo "Available: LA5c, TCP, COBRE, SRPBS, MCIC"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  DOWNLOAD COMPLETE (or instructions shown)"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Verify downloads: check participants.tsv in each site dir"
echo "  2. Run preprocessing:"
echo "     python scripts/preprocess_real_data.py --data_dir $DATA_DIR --compute_fdt"
echo "  3. Train model:"
echo "     python experiments/train_federated.py --data_dir $DATA_DIR/processed --n_rois 100"
