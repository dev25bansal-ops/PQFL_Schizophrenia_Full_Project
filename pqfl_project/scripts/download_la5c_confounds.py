#!/usr/bin/env python3
"""Download missing confound TSV files for LA5c from OpenNeuro S3.

These are tiny files (~10-50 KB each) that were not included in the
fMRIPrep derivatives download. They contain motion parameters, CSF/WM
signals, and other confound regressors needed for clean FC computation.
"""
import os
import sys
from pathlib import Path

DATA_DIR = Path("./data")
LA5C_DIR = DATA_DIR / "LA5c"
FMRIPREP_DIR = LA5C_DIR / "derivatives" / "fmriprep"


def download_confounds():
    """Download confound TSV files from OpenNeuro S3."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        print("ERROR: boto3 not installed. Run: pip install boto3")
        return False

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    # Get list of subjects
    subjects = sorted([d for d in FMRIPREP_DIR.iterdir()
                      if d.is_dir() and d.name.startswith("sub-")])

    if not subjects:
        print("No subjects found!")
        return False

    print(f"Found {len(subjects)} subjects")
    print(f"Downloading confound files from OpenNeuro S3...")
    print()

    downloaded = 0
    skipped = 0
    failed = 0

    for i, sub_dir in enumerate(subjects):
        sub_id = sub_dir.name
        func_dir = sub_dir / "func"

        if not func_dir.exists():
            continue

        # Check which rest BOLD files exist
        bold_files = sorted(func_dir.glob("*task-rest*_preproc.nii.gz"))

        for bold_file in bold_files:
            # Derive expected confound filename from BOLD filename
            # e.g., sub-10159_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz
            #   -> sub-10159_task-rest_bold_desc-confounds_timeseries.tsv
            # The confound file matches the BIDS entity: sub-xxxxx_task-rest_confounds.tsv
            confound_name = f"{sub_id}_task-rest_desc-confounds_timeseries.tsv"
            confound_path = func_dir / confound_name

            if confound_path.exists() and confound_path.stat().st_size > 0:
                skipped += 1
                continue

            # Try downloading from S3
            # The original (non-derivatives) path has the confounds
            s3_keys = [
                f"ds000030/ds000030_R1.0.5/uncompressed/{sub_id}/func/{confound_name}",
                f"ds000030/ds000030_R1.0.5/uncompressed/{sub_id}/func/{sub_id}_task-rest_confounds.tsv",
            ]

            success = False
            for s3_key in s3_keys:
                try:
                    s3.download_file("openneuro", s3_key, str(confound_path))
                    if confound_path.exists() and confound_path.stat().st_size > 0:
                        success = True
                        break
                    else:
                        confound_path.unlink(missing_ok=True)
                except Exception:
                    if confound_path.exists():
                        confound_path.unlink()

            if success:
                downloaded += 1
                if (downloaded) % 10 == 0:
                    print(f"  Progress: {downloaded} downloaded, {skipped} skipped, {failed} failed")
            else:
                failed += 1

    print()
    print(f"Results: {downloaded} downloaded, {skipped} already existed, {failed} failed")

    if failed > 0:
        print(f"\nNote: {failed} confound files could not be downloaded.")
        print("The preprocessing pipeline can still run without them,")
        print("using only bandpass filtering + ROI extraction.")

    return downloaded > 0 or skipped > 0


def main():
    print("=" * 60)
    print("  LA5c CONFOUNDS DOWNLOADER")
    print("=" * 60)
    print()

    if not FMRIPREP_DIR.exists():
        print(f"ERROR: {FMRIPREP_DIR} does not exist!")
        return

    # Check current confound status
    existing_confounds = list(FMRIPREP_DIR.rglob("*confounds*.tsv"))
    print(f"Current confound files: {len(existing_confounds)}")

    if len(existing_confounds) > 0:
        print("Some confounds already exist. Checking which are missing...")

    download_confounds()

    # Re-check after download
    final_confounds = list(FMRIPREP_DIR.rglob("*confounds*.tsv"))
    print(f"\nFinal confound files: {len(final_confounds)}")

    print()
    print("Next step:")
    print("  python scripts/preprocess_real_data.py --data_dir ./data --sites LA5c --compute_fdt")


if __name__ == "__main__":
    main()
