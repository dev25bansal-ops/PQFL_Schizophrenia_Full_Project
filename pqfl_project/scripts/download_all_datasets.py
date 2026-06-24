#!/usr/bin/env python3
"""Download ALL open-access schizophrenia fMRI datasets (excluding LA5c).

This script downloads every publicly accessible dataset for the PQFL
Schizophrenia project. LA5c is SKIPPED since it is already processed.

Available open-access datasets (auto-downloaded):
  1. TCP2025  - OpenNeuro ds005237  (40 SZ, 93 HC)   ~50 GB raw
  2. SPINS    - OpenNeuro ds003011  (94 SZ, 94 HC)   ~80 GB raw

Registration-required datasets (instructions provided):
  3. COBRE    - COINS Data Exchange (72 SZ, 74 HC)   ~20 GB raw
  4. SRPBS    - BICR ATR Japan       (146 SZ, 800 HC) 175 MB FC only!

DUA/Restricted datasets (instructions provided):
  5. MCIC     - COINS DUA           (146 SZ, 160 HC)
  6. BSNIP2   - NDA + IRB           (150 SZ, 223 HC)
  7. FBIRN    - Contact PI          (176 SZ, 186 HC)

Usage:
    # Download all open-access datasets (TCP2025 + SPINS), skip LA5c
    python scripts/download_all_datasets.py --data_dir ./data

    # Download only resting-state files (saves ~60%% bandwidth)
    python scripts/download_all_datasets.py --data_dir ./data --rest_only

    # Download a specific dataset
    python scripts/download_all_datasets.py --data_dir ./data --site TCP2025

    # Show instructions for ALL datasets (including restricted)
    python scripts/download_all_datasets.py --instructions

    # Force re-download even if directory exists
    python scripts/download_all_datasets.py --data_dir ./data --force

Prerequisites:
    pip install boto3          # For AWS S3 downloads
    pip install openneuro-py   # Alternative download method

Note:
    LA5c is EXPLICITLY EXCLUDED from auto-download since it is already
    processed and available at data/processed/LA5c_processed.npz.
    To re-download LA5c, use: python scripts/download_datasets.py --site LA5c
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Dataset definitions — ALL datasets, LA5c excluded from auto-download
# =============================================================================

DATASETS = {
    # ──────────────── OPEN ACCESS (auto-downloaded) ────────────────
    "TCP2025": {
        "openneuro_id": "ds005237",
        "s3_bucket": "openneuro.org",
        "s3_prefix": "ds005237",
        "access": "open",
        "auto_download": True,
        "description": "Transdiagnostic Connectome Project (TCP)",
        "n_sz": 40,
        "n_hc": 93,
        "diagnosis_col": "Group",
        "sz_label": "patient",
        "hc_label": "healthy control",
        "url": "https://openneuro.org/datasets/ds005237",
        "tr": 0.8,
        "task_name": "rest",
        "notes": "HCP-style multiband, 6 rest runs per subject, TR=0.8s",
        "estimated_size_gb": 50,
    },
    "SPINS": {
        "openneuro_id": "ds003011",
        "s3_bucket": "openneuro.org",
        "s3_prefix": "ds003011",
        "access": "open",
        "auto_download": True,
        "description": "Social Processes Initiative in Neurobiology of the Schizophrenia(s)",
        "n_sz": 94,
        "n_hc": 94,
        "diagnosis_col": "diagnosis",
        "sz_label": "schizophrenia",
        "hc_label": "control",
        "url": "https://openneuro.org/datasets/ds003011",
        "tr": 0.8,
        "task_name": "rest",
        "notes": "3 acquisition sites (CAMH, MNI, Zucker Hillside), TR=0.8s",
        "estimated_size_gb": 80,
    },

    # ──────────────── LA5c — EXCLUDED (already downloaded) ──────────
    "LA5c": {
        "openneuro_id": "ds000030",
        "s3_bucket": "openneuro",
        "s3_prefix": "ds000030/ds000030_R1.0.5/uncompressed/",
        "access": "open",
        "auto_download": False,  # <--- EXPLICITLY SKIPPED
        "skip_reason": "Already processed at data/processed/LA5c_processed.npz",
        "description": "UCLA CNP LA5c Study (ALREADY DOWNLOADED)",
        "n_sz": 50,
        "n_hc": 127,
        "url": "https://openneuro.org/datasets/ds000030",
    },

    # ──────────────── REGISTRATION REQUIRED ────────────────
    "COBRE": {
        "access": "registration",
        "auto_download": False,
        "description": "Center for Biomedical Research Excellence - Schizophrenia",
        "n_sz": 72,
        "n_hc": 74,
        "estimated_size_gb": 20,
        "instructions": (
            "COBRE requires COINS Data Exchange account (FREE, ~1 day approval):\n"
            "  1. Register at: https://coins.trendscenter.org/\n"
            "  2. Go to Data Exchange -> Browse Available Data\n"
            "  3. Filter by Study Name = COBRE -> Submit Request\n"
            "  4. Data available within ~1 business day\n"
            "  5. Place downloaded data in: {data_dir}/COBRE/\n"
            "\n"
            "  NOTE: All Figshare links are 403 Forbidden (dead since 2025)\n"
            "  NOTE: COBRE is NOT on OpenNeuro\n"
            "  NOTE: Phenotypic CSV: COBRE_phenotypic_data.csv with 'Subject Type' column"
        ),
    },
    "SRPBS": {
        "access": "registration",
        "auto_download": False,
        "description": "SRPBS-1600 Multi-disorder (Japan, 12 sites)",
        "n_sz": 146,
        "n_hc": 800,
        "estimated_size_gb": 0.175,  # Precomputed FC only 175MB!
        "instructions": (
            "SRPBS requires application at BICR (ATR Japan) -- RECOMMENDED: FC only!\n"
            "  OPTION A -- Precomputed FC (only 175MB, MUCH faster):\n"
            "  1. Go to: https://bicr-resource.atr.jp/srpbsfc\n"
            "  2. Download and fill 'Application Form for Data Usage'\n"
            "  3. Upload signed form + register\n"
            "  4. Wait for email approval with S3 download link\n"
            "\n"
            "  OPTION B -- Raw fMRI (89.8 GB, NOT recommended):\n"
            "  Go to: https://bicr-resource.atr.jp/srpbs1600\n"
            "\n"
            "  After downloading, place files in: {data_dir}/SRPBS/\n"
            "  The FC format is .mat files with precomputed connectivity matrices"
        ),
    },

    # ──────────────── DUA / RESTRICTED ────────────────
    "MCIC": {
        "access": "dua",
        "auto_download": False,
        "description": "Mind Clinical Interface Consortium (MCICShare)",
        "n_sz": 146,
        "n_hc": 160,
        "estimated_size_gb": 30,
        "instructions": (
            "MCIC requires COINS Data Exchange DUA (FREE, slow approval ~weeks):\n"
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
        "auto_download": False,
        "description": "Bipolar and Schizophrenia Network on Intermediate Phenotypes",
        "n_sz": 150,
        "n_hc": 223,
        "estimated_size_gb": 60,
        "instructions": (
            "BSNIP-2 requires NIMH Data Archive access (requires IRB documentation):\n"
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
        "auto_download": False,
        "description": "Function Biomedical Informatics Research Network (Phase III)",
        "n_sz": 176,
        "n_hc": 186,
        "estimated_size_gb": 40,
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

# Track download status
STATUS_FILE = "download_status.json"


def load_status(data_dir: Path) -> Dict:
    """Load download status from JSON file."""
    status_path = data_dir / STATUS_FILE
    if status_path.exists():
        with open(status_path, 'r') as f:
            return json.load(f)
    return {}


def save_status(data_dir: Path, status: Dict):
    """Save download status to JSON file."""
    status_path = data_dir / STATUS_FILE
    with open(status_path, 'w') as f:
        json.dump(status, f, indent=2, default=str)


def estimate_download_time(size_gb: float, speed_mbps: float = 50) -> str:
    """Estimate download time given size and assumed speed."""
    seconds = (size_gb * 1024) / (speed_mbps / 8)
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}min"
    else:
        return f"{seconds/3600:.1f}h"


def download_via_boto3(
    dataset_name: str,
    config: dict,
    data_dir: Path,
    rest_only: bool = False,
    force: bool = False,
) -> bool:
    """Download dataset using boto3 (AWS SDK for Python).

    Args:
        dataset_name: Name of the dataset.
        config: Dataset configuration dictionary.
        data_dir: Base data directory.
        rest_only: If True, only download resting-state files.
        force: If True, re-download even if directory exists.

    Returns:
        True if download was successful, False otherwise.
    """
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        print(f"  ERROR: boto3 not installed. Install with: pip install boto3")
        print(f"  Alternative: Use AWS CLI (see instructions below)")
        return False

    s3_prefix = config["s3_prefix"]
    bucket = config["s3_bucket"]
    outdir = data_dir / dataset_name

    # Check if already downloaded
    if outdir.exists() and not force:
        # Check if it has actual data
        nii_files = list(outdir.rglob("*.nii.gz")) + list(outdir.rglob("*.nii"))
        if nii_files:
            print(f"  SKIP: {dataset_name} already exists at {outdir} ({len(nii_files)} NIfTI files)")
            print(f"  Use --force to re-download")
            return True

    outdir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    size_gb = config.get("estimated_size_gb", 50)
    est_time = estimate_download_time(size_gb)

    print(f"\n{'='*70}")
    print(f"  DOWNLOADING: {dataset_name}")
    print(f"  {config['description']}")
    print(f"  SZ={config['n_sz']}, HC={config['n_hc']}")
    print(f"  S3: s3://{bucket}/{s3_prefix}")
    print(f"  Output: {outdir}")
    print(f"  Estimated size: ~{size_gb} GB  |  Estimated time: ~{est_time}")
    if rest_only:
        print(f"  MODE: Resting-state only (saves ~60%% bandwidth)")
    print(f"{'='*70}\n")

    # List objects in the bucket
    print("  Scanning S3 bucket for files...")
    paginator = s3.get_paginator("list_objects_v2")

    total_files = 0
    to_download = []
    skipped_filters = 0

    try:
        pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix)
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
                        pass  # Need anatomy for spatial normalization
                    elif "task-rest" in rel_path and (".nii" in rel_path or ".json" in rel_path):
                        pass
                    elif "task-rest_physio" in rel_path:
                        pass
                    elif "task-rest_desc-confounds" in rel_path:
                        pass
                    elif "_bold.nii" in rel_path and "task-rest" not in rel_path:
                        skipped_filters += 1
                        continue
                    else:
                        # Check if it's a task-based file we can skip
                        if "task-" in rel_path and "task-rest" not in rel_path:
                            skipped_filters += 1
                            continue

                # Skip if already downloaded
                if local_path.exists() and local_path.stat().st_size > 0:
                    continue

                to_download.append((key, local_path))
    except Exception as e:
        print(f"  ERROR scanning S3 bucket: {e}")
        print(f"  This may be a network issue. Try again or use AWS CLI:")
        print(f"    aws s3 sync --no-sign-request s3://{bucket}/{s3_prefix} {outdir}/")
        return False

    print(f"  Total files in S3: {total_files}")
    print(f"  Files to download: {len(to_download)} (skipped: {skipped_filters} filtered, rest already exist)")
    print()

    if not to_download:
        print(f"  All files already downloaded for {dataset_name}!")
        return True

    # Download files
    downloaded = 0
    failed = 0
    start_time = time.time()

    for i, (key, local_path) in enumerate(to_download):
        # Create parent directory
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download
        try:
            elapsed = time.time() - start_time
            speed = (downloaded / elapsed * 8 / 1024) if elapsed > 0 and downloaded > 0 else 0
            progress = f"[{i+1}/{len(to_download)}]"
            rel_name = str(local_path.relative_to(outdir))
            if len(rel_name) > 65:
                rel_name = "..." + rel_name[-62:]

            print(f"  {progress:14s} {rel_name}", end="", flush=True)
            s3.download_file(bucket, key, str(local_path))
            file_size = local_path.stat().st_size
            downloaded += file_size
            print(f"  OK ({file_size/1024/1024:.1f} MB)")
        except KeyboardInterrupt:
            print(f"\n\n  Download interrupted by user. Partial data saved to {outdir}")
            print(f"  Re-run the same command to resume downloading.")
            return False
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    elapsed = time.time() - start_time
    total_mb = downloaded / 1024 / 1024

    print(f"\n  Download complete for {dataset_name}:")
    print(f"    Downloaded: {total_mb:.1f} MB in {elapsed/60:.1f} min")
    print(f"    Failed: {failed} files")
    print(f"    Output: {outdir}")

    return failed == 0


def download_via_openneuro_py(
    dataset_name: str,
    config: dict,
    data_dir: Path,
    rest_only: bool = False,
    force: bool = False,
) -> bool:
    """Download dataset using openneuro-py package.

    This is an alternative to boto3 that may work better for some datasets.
    """
    try:
        import openneuro
    except ImportError:
        print(f"  openneuro-py not installed. Install with: pip install openneuro-py")
        print(f"  Falling back to boto3 method...")
        return False

    dataset_id = config.get("openneuro_id")
    if not dataset_id:
        return False

    outdir = data_dir / dataset_name

    if outdir.exists() and not force:
        nii_files = list(outdir.rglob("*.nii.gz")) + list(outdir.rglob("*.nii"))
        if nii_files:
            print(f"  SKIP: {dataset_name} already exists at {outdir} ({len(nii_files)} NIfTI files)")
            return True

    print(f"\n  Downloading {dataset_name} via openneuro-py...")
    try:
        openneuro.download(
            dataset=dataset_id,
            target_dir=str(outdir),
        )
        return True
    except Exception as e:
        print(f"  openneuro-py download failed: {e}")
        return False


def print_awscli_commands(
    dataset_name: str,
    config: dict,
    data_dir: Path,
    rest_only: bool = False,
):
    """Print AWS CLI commands for manual downloading."""
    s3_prefix = config["s3_prefix"]
    bucket = config["s3_bucket"]
    outdir = data_dir / dataset_name

    print(f"\n  AWS CLI COMMAND for {dataset_name}:")
    print(f"  --------------------------")

    if rest_only:
        print(f'  aws s3 sync --no-sign-request \\')
        print(f'    --exclude "*" \\')
        print(f'    --include "participants.tsv" \\')
        print(f'    --include "dataset_description.json" \\')
        print(f'    --include "*/anat/*" \\')
        print(f'    --include "*/task-rest*_bold.nii.gz" \\')
        print(f'    --include "*/task-rest*_bold.json" \\')
        print(f'    --include "*/task-rest*_desc-confounds_timeseries.tsv" \\')
        print(f'    s3://{bucket}/{s3_prefix} \\')
        print(f'    {outdir}/')
    else:
        print(f'  aws s3 sync --no-sign-request \\')
        print(f'    s3://{bucket}/{s3_prefix} \\')
        print(f'    {outdir}/')

    print()


def check_existing_data(data_dir: Path) -> Dict[str, Dict]:
    """Check what datasets already exist in the data directory."""
    existing = {}
    for name in DATASETS:
        site_dir = data_dir / name
        if site_dir.exists():
            nii_files = list(site_dir.rglob("*.nii.gz")) + list(site_dir.rglob("*.nii"))
            tsv_files = list(site_dir.rglob("participants.tsv")) + list(site_dir.rglob("*.csv"))
            npz_files = list(site_dir.rglob("*.npz"))
            mat_files = list(site_dir.rglob("*.mat"))

            existing[name] = {
                "nii_count": len(nii_files),
                "tsv_count": len(tsv_files),
                "npz_count": len(npz_files),
                "mat_count": len(mat_files),
                "has_data": len(nii_files) > 0 or len(npz_files) > 0 or len(mat_files) > 0,
            }
    return existing


def show_instructions(data_dir: str):
    """Show download instructions for ALL datasets (including restricted)."""
    print(f"\n{'='*74}")
    print(f"  PQFL SCHIZOPHRENIA -- COMPLETE DATASET ACCESS GUIDE")
    print(f"  (LA5c excluded -- already downloaded)")
    print(f"{'='*74}\n")

    # Summary table
    print(f"  {'Dataset':12s} | {'Access':20s} | {'SZ':>4s} | {'HC':>4s} | {'Size':>8s} | Status")
    print(f"  {'-'*12}-+-{'-'*20}-+-{'-'*4}-+-{'-'*4}-+-{'-'*8}-+-{'-'*20}")

    existing = check_existing_data(Path(data_dir))

    for name, config in DATASETS.items():
        access = config["access"]
        if name == "LA5c":
            status = "ALREADY DOWNLOADED"
        elif name in existing and existing[name]["has_data"]:
            status = "DATA EXISTS"
        else:
            status = "NOT DOWNLOADED"

        size_str = f"~{config.get('estimated_size_gb', '?')} GB"

        if access == "open" and name != "LA5c":
            access_str = "OPEN (auto-download)"
        elif name == "LA5c":
            access_str = "OPEN (excluded)"
        elif access == "registration":
            access_str = "REGISTRATION REQ."
        elif access == "dua":
            access_str = "DUA REQUIRED"
        elif access == "nda_controlled":
            access_str = "NDA + IRB REQ."
        elif access == "restricted":
            access_str = "CONTACT PI"
        else:
            access_str = access

        print(f"  {name:12s} | {access_str:20s} | {config['n_sz']:4d} | {config['n_hc']:4d} | {size_str:>8s} | {status}")

    # Total potential
    total_sz = sum(c["n_sz"] for c in DATASETS.values() if c["n_sz"])
    total_hc = sum(c["n_hc"] for c in DATASETS.values() if c["n_hc"])
    print(f"\n  TOTAL POTENTIAL: {total_sz} SZ + {total_hc} HC = {total_sz + total_hc} subjects")
    print(f"  (LA5c already: 50 SZ + 127 HC = 177 subjects)")

    # Detailed instructions
    print(f"\n{'='*74}")
    print(f"  DETAILED DOWNLOAD INSTRUCTIONS")
    print(f"{'='*74}")

    # Open access
    print(f"\n  +---------------------------------------------------------+")
    print(f"  |  OPEN ACCESS (auto-download with this script)           |")
    print(f"  +---------------------------------------------------------+")

    for name, config in DATASETS.items():
        if config["access"] != "open" or name == "LA5c":
            continue
        print(f"\n  {name} -- {config['description']}")
        print(f"  OpenNeuro: {config['openneuro_id']}")
        print(f"  URL: {config['url']}")
        print(f"  SZ={config['n_sz']}, HC={config['n_hc']}")
        print(f"  Auto-download: python scripts/download_all_datasets.py --data_dir {data_dir} --site {name}")

    # LA5c note
    print(f"\n  +---------------------------------------------------------+")
    print(f"  |  LA5c -- ALREADY DOWNLOADED (excluded from auto-dl)     |")
    print(f"  +---------------------------------------------------------+")
    print(f"\n  LA5c is already processed at data/processed/LA5c_processed.npz")
    print(f"  To re-download: python scripts/download_datasets.py --site LA5c")

    # Registration required
    print(f"\n  +---------------------------------------------------------+")
    print(f"  |  REGISTRATION REQUIRED (free, ~1 day approval)          |")
    print(f"  +---------------------------------------------------------+")

    for name, config in DATASETS.items():
        if config["access"] != "registration":
            continue
        print(f"\n  {name} -- {config['description']}")
        print(f"  SZ={config['n_sz']}, HC={config['n_hc']}")
        if "instructions" in config:
            for line in config["instructions"].format(data_dir=data_dir).split("\n"):
                print(f"  {line}")

    # DUA/Restricted
    print(f"\n  +---------------------------------------------------------+")
    print(f"  |  DUA / RESTRICTED ACCESS (weeks-months for approval)    |")
    print(f"  +---------------------------------------------------------+")

    for name, config in DATASETS.items():
        if config["access"] in ("dua", "nda_controlled", "restricted"):
            print(f"\n  {name} -- {config['description']}")
            print(f"  SZ={config['n_sz']}, HC={config['n_hc']}")
            if "instructions" in config:
                for line in config["instructions"].format(data_dir=data_dir).split("\n"):
                    print(f"  {line}")

    print(f"\n  Output directory: {data_dir}/")
    print(f"  Quick start: python scripts/download_all_datasets.py --data_dir {data_dir} --rest_only")


def show_summary(data_dir: Path, results: Dict):
    """Show download summary and next steps."""
    print(f"\n{'='*74}")
    print(f"  DOWNLOAD SUMMARY")
    print(f"{'='*74}\n")

    existing = check_existing_data(data_dir)

    # Count accessible data
    open_sz = 0
    open_hc = 0

    for name, config in DATASETS.items():
        if name == "LA5c":
            continue
        if config["access"] == "open":
            open_sz += config["n_sz"]
            open_hc += config["n_hc"]

    print(f"  Already have (LA5c):    50 SZ + 127 HC = 177 subjects")
    print(f"  Open access available: {open_sz} SZ + {open_hc} HC = {open_sz + open_hc} subjects")
    print(f"  Registration required: {DATASETS['COBRE']['n_sz']}+{DATASETS['SRPBS']['n_sz']} SZ + {DATASETS['COBRE']['n_hc']}+{DATASETS['SRPBS']['n_hc']} HC")
    print(f"  Full potential (all):  {sum(c['n_sz'] for c in DATASETS.values())} SZ + {sum(c['n_hc'] for c in DATASETS.values())} HC")

    print(f"\n{'='*74}")
    print(f"  NEXT STEPS")
    print(f"{'='*74}")
    print(f"""
  1. Verify downloads:
     Check that participants.tsv and BOLD files exist in each site directory.

  2. Preprocess new datasets:
     python scripts/preprocess_all_datasets.py --data_dir {data_dir} --compute_fdt

  3. Integrate with LA5c for federated training:
     python scripts/integrate_datasets.py --data_dir {data_dir}/processed

  4. For registration-required datasets (COBRE, SRPBS):
     Follow the instructions above, then re-run preprocessing.

  5. If you only downloaded resting-state data, you can still preprocess:
     The pipeline only needs task-rest BOLD + confounds + participants.tsv
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Download ALL open-access schizophrenia fMRI datasets (excluding LA5c)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all open-access datasets (TCP2025 + SPINS), skip LA5c
  python scripts/download_all_datasets.py --data_dir ./data

  # Download only resting-state files (saves ~60%% bandwidth)
  python scripts/download_all_datasets.py --data_dir ./data --rest_only

  # Download a specific dataset
  python scripts/download_all_datasets.py --data_dir ./data --site TCP2025

  # Show instructions for ALL datasets (including restricted)
  python scripts/download_all_datasets.py --instructions

  # Force re-download even if directory exists
  python scripts/download_all_datasets.py --data_dir ./data --force

Note: LA5c is EXCLUDED from auto-download (already processed).
      To download LA5c separately: python scripts/download_datasets.py --site LA5c
        """,
    )
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Base directory for downloads")
    parser.add_argument("--site", type=str, default=None,
                        choices=[k for k in DATASETS.keys() if k != "LA5c"],
                        help="Download only one site (cannot specify LA5c)")
    parser.add_argument("--rest_only", action="store_true",
                        help="Download only resting-state fMRI (saves bandwidth)")
    parser.add_argument("--method", type=str, default="boto3",
                        choices=["boto3", "openneuro", "awscli"],
                        help="Download method")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if directory exists")
    parser.add_argument("--instructions", action="store_true",
                        help="Show access instructions for ALL datasets")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.instructions:
        show_instructions(args.data_dir)
        return

    # Print banner
    print(f"""
{'='*74}
  PQFL SCHIZOPHRENIA -- MULTI-DATASET DOWNLOADER
  Downloading ALL open-access datasets (LA5c EXCLUDED)
{'='*74}
""")

    # Check existing data
    existing = check_existing_data(data_dir)
    print(f"  Data directory: {data_dir}")
    print(f"  Existing datasets: {list(existing.keys()) or 'None'}")

    # Check if LA5c already processed
    la5c_processed = (data_dir / "processed" / "LA5c_processed.npz").exists()
    if la5c_processed:
        print(f"  LA5c: Already processed (skipping download)")
    else:
        print(f"  LA5c: Not yet processed (but will NOT be downloaded by this script)")

    # Determine which datasets to download
    if args.site:
        sites = {args.site: DATASETS[args.site]}
    else:
        # Only auto-download open-access datasets (excluding LA5c)
        sites = {
            name: config for name, config in DATASETS.items()
            if config.get("auto_download", False)
        }

    if not sites:
        print(f"\n  No auto-downloadable datasets selected.")
        print(f"  Use --instructions to see how to access registration-required datasets.")
        return

    # Calculate total download size
    total_gb = sum(c.get("estimated_size_gb", 50) for c in sites.values())
    if args.rest_only:
        total_gb *= 0.4  # ~60% savings with rest-only

    print(f"\n  Datasets to download: {list(sites.keys())}")
    print(f"  Estimated total size: ~{total_gb:.0f} GB")
    print(f"  Method: {args.method}")
    if args.rest_only:
        print(f"  Mode: Resting-state only (saves bandwidth)")

    # Confirm before starting
    try:
        response = input(f"\n  Start downloading ~{total_gb:.0f} GB? [y/N]: ")
        if response.lower() not in ('y', 'yes'):
            print("  Download cancelled.")
            return
    except (EOFError, KeyboardInterrupt):
        print("\n  Download cancelled.")
        return

    # Process each site
    results = {}
    for name, config in sites.items():
        # Skip LA5c -- this is the critical safeguard
        if name == "LA5c":
            print(f"\n  SKIP: LA5c is excluded from auto-download (already processed)")
            results[name] = {"status": "skipped", "reason": "already_downloaded"}
            continue

        if config["access"] != "open":
            print(f"\n  SKIP: {name} requires {config['access']} access -- showing instructions:")
            if "instructions" in config:
                for line in config["instructions"].format(data_dir=args.data_dir).split("\n"):
                    print(f"    {line}")
            results[name] = {"status": "skipped", "reason": config["access"]}
            continue

        # Attempt download
        success = False
        if args.method == "boto3":
            success = download_via_boto3(name, config, data_dir, args.rest_only, args.force)
            if not success:
                print(f"\n  boto3 failed, trying openneuro-py...")
                success = download_via_openneuro_py(name, config, data_dir, args.rest_only, args.force)
                if not success:
                    print(f"\n  All download methods failed. Use AWS CLI:")
                    print_awscli_commands(name, config, data_dir, args.rest_only)
        elif args.method == "openneuro":
            success = download_via_openneuro_py(name, config, data_dir, args.rest_only, args.force)
            if not success:
                print(f"\n  openneuro-py failed, trying boto3...")
                success = download_via_boto3(name, config, data_dir, args.rest_only, args.force)
                if not success:
                    print_awscli_commands(name, config, data_dir, args.rest_only)
        elif args.method == "awscli":
            print_awscli_commands(name, config, data_dir, args.rest_only)
            success = True  # Commands printed, consider it done

        results[name] = {
            "status": "success" if success else "failed",
            "timestamp": datetime.now().isoformat(),
        }

    # Save download status
    status = load_status(data_dir)
    for name, result in results.items():
        status[name] = result
    save_status(data_dir, status)

    # Show summary
    show_summary(data_dir, results)


if __name__ == "__main__":
    main()
