#!/usr/bin/env python3
"""Smart download: only essential data for PQFL federated training.

Strategy:
  TCP2025: Download ONLY pre-parcellated h5 + phenotypic data (~1.5 GB)
           Skip 937 GB of raw data - the h5 files are already denoised
           and parcellated, ready for FC matrix computation.
  SPINS:   Download FULL dataset (~3.84 GB) - it's small enough.
  LA5c:    SKIP - already processed.

Usage:
    # Download everything needed (TCP2025 h5 + SPINS full)
    python smart_download.py --data_dir ./data

    # Download only TCP2025 (h5 + phenotype)
    python smart_download.py --data_dir ./data --site TCP2025

    # Download only SPINS
    python smart_download.py --data_dir ./data --site SPINS

    # Force re-download
    python smart_download.py --data_dir ./data --force
"""

import argparse
import os
import sys
import time
from pathlib import Path, PurePosixPath
from datetime import datetime


def download_tcp2025_h5(data_dir: Path, force: bool = False) -> bool:
    """Download TCP2025 h5 parcellated time series + phenotypic data.

    This downloads ONLY:
    - fMRI_timeseries_clean_denoised_GSR_parcellated/*.h5 (rest runs only)
    - participants.tsv / phenotype files
    - dataset_description.json

    Total: ~1.5 GB instead of 937 GB.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    bucket = "openneuro.org"
    s3_prefix = "ds005237"
    outdir = data_dir / "TCP2025"

    # Check existing
    h5_files = list(outdir.rglob("*.h5")) if outdir.exists() else []
    pheno_files = list(outdir.rglob("participants.tsv")) if outdir.exists() else []

    if h5_files and pheno_files and not force:
        print(f"\n  TCP2025: Already have {len(h5_files)} h5 files + phenotypic data")
        print(f"  Use --force to re-download")
        return True

    outdir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    print(f"\n{'='*70}")
    print(f"  TCP2025 -- Pre-parcellated h5 + Phenotype Download")
    print(f"  Target: ~1.5 GB (instead of 937 GB raw)")
    print(f"  Output: {outdir}")
    print(f"{'='*70}\n")

    # Scan S3 for matching files
    print("  Scanning S3 bucket for h5 + phenotype files...")
    paginator = s3.get_paginator("list_objects_v2")

    to_download = []
    total_scanned = 0

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                total_scanned += 1

                rel_path = key[len(s3_prefix):].lstrip("/")
                if not rel_path:
                    continue
                local_path = outdir / Path(PurePosixPath(rel_path))

                # Filter: only download what we need
                should_download = False

                # 1. Pre-parcellated h5 time series (rest runs only)
                if ("fMRI_timeseries_clean_denoised_GSR_parcellated/" in rel_path
                        and "task-rest" in rel_path
                        and rel_path.endswith(".h5")):
                    should_download = True

                # 2. Phenotypic / participant data
                elif "participants.tsv" in rel_path:
                    should_download = True
                elif "participants.json" in rel_path:
                    should_download = True
                elif "dataset_description.json" in rel_path:
                    should_download = True
                elif "phenotype/" in rel_path and rel_path.endswith(".tsv"):
                    should_download = True

                if not should_download:
                    continue

                # Skip if already downloaded
                if local_path.exists() and local_path.stat().st_size > 0:
                    continue

                to_download.append((key, local_path, obj.get("Size", 0)))

    except Exception as e:
        print(f"  ERROR scanning S3: {e}")
        return False

    total_size_mb = sum(s for _, _, s in to_download) / 1024 / 1024
    print(f"  Total S3 objects scanned: {total_scanned}")
    print(f"  Files to download: {len(to_download)} ({total_size_mb:.1f} MB)")
    print()

    if not to_download:
        print(f"  All required TCP2025 files already downloaded!")
        return True

    # Download
    downloaded_bytes = 0
    failed = 0
    start_time = time.time()

    for i, (key, local_path, expected_size) in enumerate(to_download):
        local_path.parent.mkdir(parents=True, exist_ok=True)

        rel_name = str(local_path.relative_to(outdir))
        if len(rel_name) > 65:
            rel_name = "..." + rel_name[-62:]

        try:
            print(f"  [{i+1:3d}/{len(to_download)}] {rel_name}", end="", flush=True)
            s3.download_file(bucket, key, str(local_path))
            file_size = local_path.stat().st_size
            downloaded_bytes += file_size

            elapsed = time.time() - start_time
            speed = (downloaded_bytes / elapsed / 1024 / 1024) if elapsed > 0 else 0
            print(f"  OK ({file_size/1024/1024:.1f} MB, {speed:.1f} MB/s)")
        except KeyboardInterrupt:
            print(f"\n\n  Download interrupted. Partial data saved to {outdir}")
            print(f"  Re-run to resume downloading.")
            return False
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    elapsed = time.time() - start_time
    total_mb = downloaded_bytes / 1024 / 1024
    print(f"\n  TCP2025 download complete: {total_mb:.1f} MB in {elapsed/60:.1f} min, {failed} failures")
    return failed == 0


def download_spins(data_dir: Path, force: bool = False) -> bool:
    """Download SPINS dataset from OpenNeuro.

    SPINS is only ~3.84 GB total, so we download the full dataset.
    We focus on resting-state BOLD + confounds + anatomy + phenotypic data.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    bucket = "openneuro.org"
    s3_prefix = "ds003011"
    outdir = data_dir / "SPINS"

    # Check existing
    bold_files = list(outdir.rglob("*task-rest*bold.nii.gz")) if outdir.exists() else []
    pheno_files = list(outdir.rglob("participants.tsv")) if outdir.exists() else []

    if bold_files and pheno_files and not force:
        print(f"\n  SPINS: Already have {len(bold_files)} rest BOLD files + phenotypic data")
        print(f"  Use --force to re-download")
        return True

    outdir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    print(f"\n{'='*70}")
    print(f"  SPINS -- Full Dataset Download (resting-state focused)")
    print(f"  Target: ~3.84 GB")
    print(f"  Output: {outdir}")
    print(f"{'='*70}\n")

    # Scan S3
    print("  Scanning S3 bucket for SPINS files...")
    paginator = s3.get_paginator("list_objects_v2")

    to_download = []
    total_scanned = 0

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                total_scanned += 1

                rel_path = key[len(s3_prefix):].lstrip("/")
                if not rel_path:
                    continue
                local_path = outdir / Path(PurePosixPath(rel_path))

                # Filter: resting-state focused
                should_download = False

                # Essential metadata
                if "participants.tsv" in rel_path or "participants.json" in rel_path:
                    should_download = True
                elif "dataset_description.json" in rel_path:
                    should_download = True

                # Resting-state functional
                elif "task-rest" in rel_path:
                    if any(x in rel_path for x in ["_bold.nii", "_bold.json",
                                                      "_desc-confounds",
                                                      "_physio"]):
                        should_download = True

                # Anatomy (needed for normalization)
                elif "/anat/" in rel_path and ("T1w" in rel_path or "T2w" in rel_path):
                    should_download = True

                # DWI (optional, for connectivity validation)
                elif "/dwi/" in rel_path:
                    # Skip DWI to save space - not needed for FC analysis
                    pass

                if not should_download:
                    continue

                # Skip if already downloaded
                if local_path.exists() and local_path.stat().st_size > 0:
                    continue

                to_download.append((key, local_path, obj.get("Size", 0)))

    except Exception as e:
        print(f"  ERROR scanning S3: {e}")
        return False

    total_size_gb = sum(s for _, _, s in to_download) / 1024 / 1024 / 1024
    print(f"  Total S3 objects scanned: {total_scanned}")
    print(f"  Files to download: {len(to_download)} ({total_size_gb:.2f} GB)")
    print()

    if not to_download:
        print(f"  All required SPINS files already downloaded!")
        return True

    # Download
    downloaded_bytes = 0
    failed = 0
    start_time = time.time()

    for i, (key, local_path, expected_size) in enumerate(to_download):
        local_path.parent.mkdir(parents=True, exist_ok=True)

        rel_name = str(local_path.relative_to(outdir))
        if len(rel_name) > 65:
            rel_name = "..." + rel_name[-62:]

        try:
            print(f"  [{i+1:3d}/{len(to_download)}] {rel_name}", end="", flush=True)
            s3.download_file(bucket, key, str(local_path))
            file_size = local_path.stat().st_size
            downloaded_bytes += file_size

            elapsed = time.time() - start_time
            speed = (downloaded_bytes / elapsed / 1024 / 1024) if elapsed > 0 else 0
            print(f"  OK ({file_size/1024/1024:.1f} MB, {speed:.1f} MB/s)")
        except KeyboardInterrupt:
            print(f"\n\n  Download interrupted. Partial data saved to {outdir}")
            print(f"  Re-run to resume downloading.")
            return False
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    elapsed = time.time() - start_time
    total_mb = downloaded_bytes / 1024 / 1024
    print(f"\n  SPINS download complete: {total_mb:.1f} MB in {elapsed/60:.1f} min, {failed} failures")
    return failed == 0


def verify_downloads(data_dir: Path):
    """Verify downloaded data and show summary."""
    print(f"\n{'='*70}")
    print(f"  DOWNLOAD VERIFICATION")
    print(f"{'='*70}\n")

    # TCP2025
    tcp_dir = data_dir / "TCP2025"
    if tcp_dir.exists():
        h5_files = list(tcp_dir.rglob("*.h5"))
        pheno = list(tcp_dir.rglob("participants.tsv"))
        h5_subjects = set()
        for f in h5_files:
            subj = f.parent.name
            h5_subjects.add(subj)
        print(f"  TCP2025:")
        print(f"    h5 files: {len(h5_files)}")
        print(f"    Unique subjects: {len(h5_subjects)}")
        print(f"    Participants.tsv: {'YES' if pheno else 'MISSING!'}")
        du = sum(f.stat().st_size for f in tcp_dir.rglob("*") if f.is_file()) / 1024 / 1024
        print(f"    Total size: {du:.1f} MB")
    else:
        print(f"  TCP2025: NOT DOWNLOADED")

    # SPINS
    spins_dir = data_dir / "SPINS"
    subjects = set()
    if spins_dir.exists():
        bold_files = list(spins_dir.rglob("*task-rest*bold.nii.gz"))
        anat_files = list(spins_dir.rglob("*T1w.nii.gz"))
        pheno = list(spins_dir.rglob("participants.tsv"))
        for f in bold_files:
            subj = f.relative_to(spins_dir).parts[0]
            subjects.add(subj)
        print(f"\n  SPINS:")
        print(f"    Rest BOLD files: {len(bold_files)}")
        print(f"    T1w files: {len(anat_files)}")
        print(f"    Unique subjects: {len(subjects)}")
        print(f"    Participants.tsv: {'YES' if pheno else 'MISSING!'}")
        du = sum(f.stat().st_size for f in spins_dir.rglob("*") if f.is_file()) / 1024 / 1024 / 1024
        print(f"    Total size: {du:.2f} GB")
    else:
        print(f"\n  SPINS: NOT DOWNLOADED")

    # LA5c
    la5c_file = data_dir / "processed" / "LA5c_processed.npz"
    if la5c_file.exists():
        print(f"\n  LA5c: Already processed ({la5c_file.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"\n  LA5c: Not yet processed")

    print(f"\n  Combined multisite potential:")
    tcp_n = len(set(f.parent.name for f in tcp_dir.rglob("*.h5"))) if tcp_dir.exists() else 0
    spins_n = len(subjects) if spins_dir.exists() else 0
    la5c_n = 172
    print(f"    TCP2025: {tcp_n} subjects (h5)")
    print(f"    SPINS: {spins_n} subjects (raw BOLD)")
    print(f"    LA5c: {la5c_n} subjects (processed)")
    print(f"    TOTAL: {tcp_n + spins_n + la5c_n} subjects across 3 datasets")


def main():
    parser = argparse.ArgumentParser(
        description="Smart download: only essential data for PQFL federated training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download everything needed (TCP2025 h5 + SPINS)
  python smart_download.py --data_dir ./data

  # Download only TCP2025 h5 + phenotype (~1.5 GB)
  python smart_download.py --data_dir ./data --site TCP2025

  # Download only SPINS (~3.84 GB)
  python smart_download.py --data_dir ./data --site SPINS

  # Verify existing downloads
  python smart_download.py --data_dir ./data --verify

Note:
  TCP2025 downloads ONLY pre-parcellated h5 time series + phenotype
  (~1.5 GB instead of 937 GB raw). These h5 files are already denoised
  and parcellated, ready for FC matrix computation.
        """,
    )
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Base directory for downloads")
    parser.add_argument("--site", type=str, default=None,
                        choices=["TCP2025", "SPINS"],
                        help="Download only one site")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if data exists")
    parser.add_argument("--verify", action="store_true",
                        help="Only verify existing downloads, don't download")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.verify:
        verify_downloads(data_dir)
        return

    print(f"""
{'='*70}
  PQFL SMART DOWNLOADER -- Essential Data Only
  TCP2025: h5 parcellated time series + phenotype (~1.5 GB)
  SPINS:   Full resting-state dataset (~3.84 GB)
  LA5c:    SKIP (already processed)
{'='*70}
""")

    results = {}

    # TCP2025
    if args.site in (None, "TCP2025"):
        success = download_tcp2025_h5(data_dir, args.force)
        results["TCP2025"] = "OK" if success else "FAILED"

    # SPINS
    if args.site in (None, "SPINS"):
        success = download_spins(data_dir, args.force)
        results["SPINS"] = "OK" if success else "FAILED"

    # Verify
    verify_downloads(data_dir)

    print(f"\n{'='*70}")
    print(f"  DOWNLOAD RESULTS")
    print(f"{'='*70}")
    for name, status in results.items():
        print(f"  {name}: {status}")

    if all(s == "OK" for s in results.values()):
        print(f"\n  Next steps:")
        print(f"  1. python scripts/preprocess_all_datasets.py --data_dir {data_dir}")
        print(f"  2. python scripts/integrate_datasets.py --data_dir {data_dir}/processed")
        print(f"  3. python experiments/train_federated.py --data_dir {data_dir}/processed")


if __name__ == "__main__":
    main()