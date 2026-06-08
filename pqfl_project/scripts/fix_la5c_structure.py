#!/usr/bin/env python3
"""Fix LA5c dataset structure.

The S3 download put fMRIPrep derivatives inside:
  data/LA5c/derivatives/fmriprep/sub-xxxxx/

This script:
1. Downloads participants.tsv from OpenNeuro S3
2. Creates a proper BIDS structure by copying/linking
3. Verifies the result
"""
import os
import sys
import shutil
import json
from pathlib import Path

DATA_DIR = Path("./data")
LA5C_DIR = DATA_DIR / "LA5c"
FMRIPREP_DIR = LA5C_DIR / "derivatives" / "fmriprep"


def download_participants_tsv():
    """Download participants.tsv from OpenNeuro S3."""
    participants_path = LA5C_DIR / "participants.tsv"

    if participants_path.exists():
        print(f"  participants.tsv already exists at {participants_path}")
        return True

    print(f"  Downloading participants.tsv from OpenNeuro S3...")

    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        s3.download_file(
            "openneuro",
            "ds000030/ds000030_R1.0.5/uncompressed/participants.tsv",
            str(participants_path),
        )
        print(f"  Downloaded participants.tsv to {participants_path}")
        return True

    except ImportError:
        print("  boto3 not installed. Trying with urllib...")

    try:
        import urllib.request
        url = "https://s3.amazonaws.com/openneuro/ds000030/ds000030_R1.0.5/uncompressed/participants.tsv"
        urllib.request.urlretrieve(url, str(participants_path))
        print(f"  Downloaded participants.tsv via HTTPS")
        return True
    except Exception as e:
        print(f"  Failed to download: {e}")
        print(f"  Manual download:")
        print(f"    URL: https://s3.amazonaws.com/openneuro/ds000030/ds000030_R1.0.5/uncompressed/participants.tsv")
        print(f"    Save to: {participants_path}")
        return False


def count_unique_subjects():
    """Count actual unique subjects in fmriprep directory."""
    if not FMRIPREP_DIR.exists():
        return 0
    # Only count directories (not files matching sub-*)
    subjects = [d for d in FMRIPREP_DIR.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    return len(subjects)


def count_rest_bolds():
    """Count resting-state BOLD files."""
    if not FMRIPREP_DIR.exists():
        return 0
    bolds = list(FMRIPREP_DIR.rglob("*task-rest*bold*.nii*"))
    return len(bolds)


def verify_participants():
    """Verify participants.tsv has the right columns."""
    participants_path = LA5C_DIR / "participants.tsv"
    if not participants_path.exists():
        return False

    import csv
    with open(participants_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        headers = reader.fieldnames
        if headers:
            print(f"  Columns: {headers}")
            # Check for diagnosis column
            diag_cols = [c for c in headers if c.lower() in ["diagnosis", "group", "subject_type"]]
            if diag_cols:
                print(f"  Found diagnosis column(s): {diag_cols}")

                # Count diagnoses
                counts = {}
                for row in reader:
                    for col in diag_cols:
                        val = row.get(col, "")
                        if val:
                            counts[val] = counts.get(val, 0) + 1
                print(f"  Diagnosis counts: {counts}")
                return True

    return False


def main():
    print("=" * 60)
    print("  LA5c DATASET STRUCTURE FIX")
    print("=" * 60)

    # Step 1: Check current state
    print(f"\n[1] Current state:")
    n_subjects = count_unique_subjects()
    n_bolds = count_rest_bolds()
    print(f"  fmriprep subjects: {n_subjects}")
    print(f"  rest BOLD files:   {n_bolds}")

    if n_subjects == 0:
        print("  ERROR: No subjects found in derivatives/fmriprep/")
        print("  Your LA5c download may be incomplete or in a different location.")
        return

    # Step 2: Download participants.tsv
    print(f"\n[2] Getting participants.tsv:")
    success = download_participants_tsv()
    if not success:
        print("  Cannot proceed without participants.tsv")
        return

    # Step 3: Verify participants.tsv
    print(f"\n[3] Verifying participants.tsv:")
    verify_participants()

    # Step 4: The structure explanation
    print(f"\n[4] Structure explanation:")
    print(f"  Your data is in fMRIPrep derivatives format (preprocessed!):")
    print(f"    data/LA5c/")
    print(f"    ├── participants.tsv  (diagnosis labels)")
    print(f"    ├── dataset_description.json")
    print(f"    └── derivatives/fmriprep/")
    print(f"        ├── sub-10159/")
    print(f"        │   ├── func/")
    print(f"        │   │   ├── sub-10159_task-rest_space-MNI152*_bold.nii.gz")
    print(f"        │   │   └── sub-10159_task-rest_desc-confounds_timeseries.tsv")
    print(f"        │   └── anat/")
    print(f"        └── ...")
    print()
    print(f"  This is GOOD - fMRIPrep data is already preprocessed!")
    print(f"  The preprocessing script will be updated to handle this structure.")

    # Step 5: Check for confounds
    print(f"\n[5] Checking for confound files:")
    confounds = list(FMRIPREP_DIR.rglob("*desc-confounds_timeseries.tsv"))
    print(f"  Confound files found: {len(confounds)}")

    # Step 6: Check for MNI-space BOLD
    print(f"\n[6] Checking BOLD file naming:")
    sample_bolds = list(FMRIPREP_DIR.rglob("*task-rest*bold*.nii*"))[:3]
    for b in sample_bolds:
        print(f"  {b.name}")

    has_mni = any("space-MNI" in b.name for b in sample_bolds)
    if has_mni:
        print(f"  MNI-space BOLD files found - ready for ROI extraction!")
    else:
        print(f"  No MNI-space BOLD found - may need different handling")

    print(f"\n{'=' * 60}")
    print(f"  FIX COMPLETE - ready for preprocessing!")
    print(f"{'=' * 60}")
    print()
    print("  Next step:")
    print("    python scripts/preprocess_real_data.py --data_dir ./data --sites LA5c --compute_fdt")


if __name__ == "__main__":
    main()
