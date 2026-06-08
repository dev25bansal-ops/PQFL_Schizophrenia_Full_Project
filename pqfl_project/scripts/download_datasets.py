#!/usr/bin/env python3
"""Download datasets for PQFL Schizophrenia project.

This script provides a cross-platform (Windows/Linux/Mac) way to download
the publicly accessible schizophrenia fMRI datasets.

Usage:
    # Download all open-access datasets (LA5c + TCP)
    python scripts/download_datasets.py --data_dir ./data

    # Download specific site only
    python scripts/download_datasets.py --data_dir ./data --site LA5c
    python scripts/download_datasets.py --data_dir ./data --site TCP

    # Download only resting-state files (saves bandwidth)
    python scripts/download_datasets.py --data_dir ./data --rest_only

    # Show download instructions for restricted datasets
    python scripts/download_datasets.py --instructions

Prerequisites:
    pip install boto3  # For AWS S3 downloads (alternative to AWS CLI)
"""

import argparse
import os
import sys
from pathlib import Path


# =============================================================================
# Dataset definitions
# =============================================================================

DATASETS = {
    "LA5c": {
        "openneuro_id": "ds000030",
        "s3_bucket": "openneuro",
        "s3_prefix": "ds000030/ds000030_R1.0.5/uncompressed/",
        "access": "open",
        "description": "UCLA Consortium for Neuropsychiatric Phenomics LA5c Study",
        "n_sz": 50,
        "n_hc": 127,
        "diagnosis_col": "diagnosis",
        "sz_label": "SCHZ",
        "hc_label": "CONTROL",
        "url": "https://openneuro.org/datasets/ds000030",
    },
    "TCP2025": {
        "openneuro_id": "ds005237",
        "s3_bucket": "openneuro.org",
        "s3_prefix": "ds005237",
        "access": "open",
        "description": "Transdiagnostic Connectome Project (TCP)",
        "n_sz": 40,
        "n_hc": 93,
        "diagnosis_col": "Group",
        "sz_label": "patient",
        "hc_label": "healthy control",
        "url": "https://openneuro.org/datasets/ds005237",
    },
    "COBRE": {
        "access": "registration",
        "description": "Center for Biomedical Research Excellence - Schizophrenia",
        "n_sz": 72,
        "n_hc": 74,
        "instructions": (
            "COBRE requires COINS Data Exchange account:\n"
            "  1. Register at: https://coins.trendscenter.org/\n"
            "  2. Go to Data Exchange -> Browse Available Data\n"
            "  3. Filter by Study Name = COBRE -> Submit Request\n"
            "  4. Data available within ~1 business day\n"
            "  5. Place downloaded data in: {data_dir}/COBRE/\n"
            "\n"
            "  NOTE: All Figshare links are 403 Forbidden (dead since 2025)\n"
            "  NOTE: COBRE is NOT on OpenNeuro"
        ),
    },
    "SRPBS": {
        "access": "registration",
        "description": "SRPBS-1600 Multi-disorder (Japan, 12 sites)",
        "n_sz": 146,
        "n_hc": 800,
        "instructions": (
            "SRPBS requires application at BICR (ATR Japan):\n"
            "  RECOMMENDED: SRPBS FC (precomputed connectivity, only 175MB)\n"
            "  1. Go to: https://bicr-resource.atr.jp/srpbsfc\n"
            "  2. Download and fill 'Application Form for Data Usage'\n"
            "  3. Upload signed form + register\n"
            "  4. Wait for email approval with S3 download link\n"
            "\n"
            "  ALTERNATIVE: Raw fMRI (89.8 GB) at https://bicr-resource.atr.jp/srpbs1600\n"
            "\n"
            "  After downloading, place files in: {data_dir}/SRPBS/"
        ),
    },
    "MCIC": {
        "access": "dua",
        "description": "Mind Clinical Interface Consortium (MCICShare)",
        "n_sz": 146,
        "n_hc": 160,
        "instructions": (
            "MCIC requires COINS Data Exchange DUA:\n"
            "  1. Register at: https://coins.trendscenter.org/\n"
            "  2. Search for 'MCICShare' study\n"
            "  3. Accept MCIC Data Use Agreement\n"
            "  4. Wait for approval (may take 1+ month)\n"
            "\n"
            "  After downloading, place files in: {data_dir}/MCIC/"
        ),
    },
    "BSNIP2": {
        "access": "nda_controlled",
        "description": "Bipolar and Schizophrenia Network on Intermediate Phenotypes",
        "n_sz": 150,
        "n_hc": 223,
        "instructions": (
            "BSNIP-2 requires NIMH Data Archive access:\n"
            "  1. Create NDA account: https://nda.nih.gov\n"
            "  2. Submit Data Use Certification (requires IRB documentation)\n"
            "  3. Wait for Data Access Committee approval\n"
            "  4. Download: pip install nda-tools && downloadcmd -d 2165\n"
            "\n"
            "  Collection: https://nda.nih.gov/edit_collection.html?id=2165\n"
            "  After downloading, place files in: {data_dir}/BSNIP2/"
        ),
    },
    "FBIRN": {
        "access": "restricted",
        "description": "Function Biomedical Informatics Research Network (Phase III)",
        "n_sz": 176,
        "n_hc": 186,
        "instructions": (
            "FBIRN requires direct contact with PI:\n"
            "  Contact: Dr. Theo G.M. van Erp\n"
            "  Email: tvanerp@hs.uci.edu\n"
            "  Must facilitate interaction with IRB + sign DUA\n"
            "\n"
            "  NOTE: Only Phase III has resting-state fMRI\n"
            "  NOTE: Phase II has only task-based fMRI\n"
            "  After downloading, place files in: {data_dir}/FBIRN/"
        ),
    },
}


def download_via_boto3(dataset_name: str, config: dict, data_dir: Path, rest_only: bool = False):
    """Download dataset using boto3 (AWS SDK for Python)."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        print(f"ERROR: boto3 not installed. Install with: pip install boto3")
        print(f"  Alternative: Install AWS CLI and use the shell script")
        return False

    s3_prefix = config["s3_prefix"]
    bucket = config["s3_bucket"]
    outdir = data_dir / dataset_name
    outdir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    print(f"\n{'='*60}")
    print(f"  DOWNLOADING: {dataset_name}")
    print(f"  {config['description']}")
    print(f"  SZ={config['n_sz']}, HC={config['n_hc']}")
    print(f"  S3: s3://{bucket}/{s3_prefix}")
    print(f"  Output: {outdir}")
    print(f"{'='*60}\n")

    # List objects in the bucket
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix)

    total_files = 0
    downloaded = 0
    skipped = 0

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            total_files += 1

            # Determine local path
            rel_path = key[len(s3_prefix):]
            if not rel_path:
                continue
            local_path = outdir / rel_path

            # Filter for rest-only mode
            if rest_only:
                # Always include essential files
                if "participants.tsv" in rel_path:
                    pass
                elif "dataset_description.json" in rel_path:
                    pass
                elif "phenotype/" in rel_path:
                    pass
                elif "/anat/" in rel_path:
                    pass
                elif "task-rest" in rel_path and (".nii" in rel_path or ".json" in rel_path):
                    pass
                elif "task-rest_physio" in rel_path:
                    pass
                else:
                    skipped += 1
                    continue

            # Skip if already downloaded
            if local_path.exists() and local_path.stat().st_size > 0:
                downloaded += 1
                continue

            # Create parent directory
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download
            try:
                print(f"  [{downloaded+1}] {rel_path[:80]}{'...' if len(rel_path) > 80 else ''}", end="", flush=True)
                s3.download_file(bucket, key, str(local_path))
                print(" OK")
                downloaded += 1
            except Exception as e:
                print(f" FAILED: {e}")

    print(f"\n  Downloaded: {downloaded}/{total_files} files (skipped: {skipped})")
    print(f"  Output: {outdir}")
    return True


def download_via_awscli(dataset_name: str, config: dict, data_dir: Path, rest_only: bool = False):
    """Print AWS CLI commands for downloading."""
    s3_prefix = config["s3_prefix"]
    bucket = config["s3_bucket"]
    outdir = data_dir / dataset_name

    print(f"\n{'='*60}")
    print(f"  AWS CLI COMMAND for {dataset_name}")
    print(f"{'='*60}\n")

    if rest_only:
        print(f"aws s3 sync --no-sign-request \\")
        print(f"  --exclude \"*\" \\")
        print(f"  --include \"participants.tsv\" \\")
        print(f"  --include \"dataset_description.json\" \\")
        print(f"  --include \"*/anat/*\" \\")
        print(f"  --include \"*/task-rest*_bold.nii.gz\" \\")
        print(f"  --include \"*/task-rest*_bold.json\" \\")
        print(f"  s3://{bucket}/{s3_prefix} \\")
        print(f"  {outdir}/")
    else:
        print(f"aws s3 sync --no-sign-request \\")
        print(f"  s3://{bucket}/{s3_prefix} \\")
        print(f"  {outdir}/")

    print()


def show_instructions(data_dir: str):
    """Show download instructions for all datasets."""
    print(f"\n{'='*70}")
    print(f"  PQFL SCHIZOPHRENIA DATASET ACCESS INSTRUCTIONS")
    print(f"{'='*70}\n")

    for name, config in DATASETS.items():
        access = config["access"]
        if access == "open":
            status = "✅ OPEN ACCESS (download now)"
        elif access == "registration":
            status = "⚠️  REGISTRATION REQUIRED (free, days)"
        elif access == "dua":
            status = "⚠️  DUA REQUIRED (free, weeks)"
        elif access == "nda_controlled":
            status = "🔴 NDA + IRB REQUIRED (weeks-months)"
        elif access == "restricted":
            status = "🔴 CONTACT PI (months)"

        print(f"  {name:12s} | {status}")
        print(f"  {'':12s} | {config['description']}")
        print(f"  {'':12s} | SZ={config['n_sz']}, HC={config['n_hc']}")
        if "url" in config:
            print(f"  {'':12s} | {config['url']}")
        if "instructions" in config:
            for line in config["instructions"].format(data_dir=data_dir).split("\n"):
                print(f"  {'':12s} | {line}")
        print()

    print(f"  Output directory: {data_dir}/")
    print(f"  Quick start: python scripts/download_datasets.py --data_dir {data_dir} --site LA5c")


def main():
    parser = argparse.ArgumentParser(
        description="Download PQFL Schizophrenia datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Base directory for downloads")
    parser.add_argument("--site", type=str, default=None,
                        choices=list(DATASETS.keys()),
                        help="Download only one site")
    parser.add_argument("--rest_only", action="store_true",
                        help="Download only resting-state fMRI (saves bandwidth)")
    parser.add_argument("--method", type=str, default="boto3",
                        choices=["boto3", "awscli"],
                        help="Download method (boto3=direct, awscli=print commands)")
    parser.add_argument("--instructions", action="store_true",
                        help="Show access instructions for all datasets")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.instructions:
        show_instructions(args.data_dir)
        return

    # Determine which datasets to download
    if args.site:
        sites = {args.site: DATASETS[args.site]}
    else:
        sites = DATASETS

    # Process each site
    for name, config in sites.items():
        if config["access"] == "open":
            if args.method == "boto3":
                success = download_via_boto3(name, config, data_dir, args.rest_only)
                if not success:
                    print(f"\n  Falling back to AWS CLI commands:")
                    download_via_awscli(name, config, data_dir, args.rest_only)
            else:
                download_via_awscli(name, config, data_dir, args.rest_only)
        else:
            # Show instructions for restricted datasets
            print(f"\n{'='*60}")
            print(f"  {name}: {config['access'].upper()} ACCESS REQUIRED")
            print(f"  {config['description']}")
            print(f"  SZ={config['n_sz']}, HC={config['n_hc']}")
            if "instructions" in config:
                for line in config["instructions"].format(data_dir=args.data_dir).split("\n"):
                    print(f"  {line}")
            print(f"{'='*60}")

    print(f"\n{'='*60}")
    print(f"  NEXT STEPS")
    print(f"{'='*60}")
    print(f"""
  1. Verify downloads: check participants.tsv in each site directory

  2. Run preprocessing:
     python scripts/preprocess_real_data.py --data_dir {data_dir} --compute_fdt

  3. Train model:
     python experiments/train_federated.py --data_dir {data_dir}/processed --n_rois 100
    """)


if __name__ == "__main__":
    main()
