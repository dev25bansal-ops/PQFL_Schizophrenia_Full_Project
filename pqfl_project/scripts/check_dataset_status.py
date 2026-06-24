#!/usr/bin/env python3
"""Check download and preprocessing status for all PQFL datasets.

Shows a comprehensive status report for each dataset including:
- Whether raw data has been downloaded
- Number of NIfTI/TSV/NPZ files found
- Whether preprocessing has been completed
- Sample counts (SZ/HC) in processed files
- Recommendations for next steps

Usage:
    python scripts/check_dataset_status.py --data_dir ./data
    python scripts/check_dataset_status.py --data_dir ./data --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np


# Dataset metadata
DATASETS = {
    "TCP2025": {
        "access": "open",
        "openneuro_id": "ds005237",
        "expected_sz": 40,
        "expected_hc": 93,
        "role": "validation",
    },
    "SPINS": {
        "access": "open",
        "openneuro_id": "ds003011",
        "expected_sz": 94,
        "expected_hc": 94,
        "role": "training",
    },
    "LA5c": {
        "access": "open",
        "openneuro_id": "ds000030",
        "expected_sz": 50,
        "expected_hc": 127,
        "role": "training",
        "note": "Already downloaded and processed",
    },
    "COBRE": {
        "access": "registration",
        "expected_sz": 72,
        "expected_hc": 74,
        "role": "training",
    },
    "SRPBS": {
        "access": "registration",
        "expected_sz": 146,
        "expected_hc": 800,
        "role": "training",
        "note": "Precomputed FC available (175MB)",
    },
    "MCIC": {
        "access": "dua",
        "expected_sz": 146,
        "expected_hc": 160,
        "role": "training",
    },
    "BSNIP2": {
        "access": "nda_controlled",
        "expected_sz": 150,
        "expected_hc": 223,
        "role": "validation",
    },
    "FBIRN": {
        "access": "restricted",
        "expected_sz": 176,
        "expected_hc": 186,
        "role": "training",
    },
}


def check_raw_status(data_dir: Path, site_name: str) -> Dict:
    """Check if raw data exists for a site."""
    site_dir = data_dir / site_name
    if not site_dir.exists():
        return {"exists": False, "nii_count": 0, "tsv_count": 0, "npz_count": 0, "mat_count": 0}

    nii_files = list(site_dir.rglob("*.nii.gz")) + list(site_dir.rglob("*.nii"))
    tsv_files = list(site_dir.rglob("participants.tsv")) + list(site_dir.rglob("*.csv"))
    npz_files = list(site_dir.rglob("*.npz"))
    mat_files = list(site_dir.rglob("*.mat"))

    has_participants = (site_dir / "participants.tsv").exists() or bool(list(site_dir.rglob("participants.tsv")))
    has_dataset_desc = (site_dir / "dataset_description.json").exists()
    has_bold = any("task-rest" in str(f) and ("bold" in str(f) or "preproc" in str(f)) for f in nii_files)

    return {
        "exists": True,
        "nii_count": len(nii_files),
        "tsv_count": len(tsv_files),
        "npz_count": len(npz_files),
        "mat_count": len(mat_files),
        "has_participants": has_participants,
        "has_dataset_desc": has_dataset_desc,
        "has_bold": has_bold,
        "has_data": len(nii_files) > 0 or len(mat_files) > 0,
    }


def check_processed_status(data_dir: Path, site_name: str) -> Dict:
    """Check if processed data exists for a site."""
    processed_dir = data_dir / "processed"
    npz_path = processed_dir / f"{site_name}_processed.npz"

    if not npz_path.exists():
        return {"exists": False}

    try:
        data = np.load(npz_path, allow_pickle=True)
        fc = data.get("fc_matrices")
        labels = data.get("labels")
        fdt = data.get("fdt_features")

        result = {
            "exists": True,
            "path": str(npz_path),
            "n_samples": int(data.get("n_samples", len(labels) if labels is not None else 0)),
            "n_sz": int(data.get("n_sz", int(labels.sum()) if labels is not None else 0)),
            "n_hc": int(data.get("n_hc", int((1-labels).sum()) if labels is not None else 0)),
            "n_rois": int(data.get("n_rois", fc.shape[1] if fc is not None else 0)),
            "has_fdt": fdt is not None,
        }

        if fdt is not None:
            result["fdt_dim"] = fdt.shape[1] if fdt.ndim > 1 else 1

        return result
    except Exception as e:
        return {"exists": True, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Check dataset download and preprocessing status")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print(f"\n{'='*80}")
    print(f"  PQFL SCHIZOPHRENIA -- DATASET STATUS REPORT")
    print(f"  Data directory: {data_dir}")
    print(f"{'='*80}\n")

    # Check multisite file
    multisite_path = data_dir / "processed" / "multisite_federated.npz"
    if multisite_path.exists():
        data = np.load(multisite_path, allow_pickle=True)
        n_total = int(data.get("n_samples", 0))
        n_sz = int(data.get("n_sz", 0))
        n_hc = int(data.get("n_hc", 0))
        n_sites = int(data.get("n_sites", 0))
        print(f"  Multi-site federated dataset: EXISTS")
        print(f"    {n_total} samples ({n_sz} SZ / {n_hc} HC) from {n_sites} sites")
        print()
    else:
        print(f"  Multi-site federated dataset: NOT YET CREATED")
        print(f"  (Run: python scripts/integrate_datasets.py --data_dir {data_dir})")
        print()

    # Per-dataset status
    print(f"  {'Dataset':12s} | {'Access':8s} | {'Raw':8s} | {'Processed':10s} | {'SZ/HC':>12s} | Next Step")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*30}")

    total_raw = 0
    total_processed = 0
    total_processed_sz = 0
    total_processed_hc = 0

    for name, meta in DATASETS.items():
        raw = check_raw_status(data_dir, name)
        processed = check_processed_status(data_dir, name)

        # Raw status
        if raw["exists"] and raw["has_data"]:
            raw_status = "YES"
            total_raw += 1
        elif raw["exists"]:
            raw_status = "PARTIAL"
        else:
            raw_status = "NO"

        # Processed status
        if processed["exists"] and "error" not in processed:
            proc_status = "YES"
            total_processed += 1
            total_processed_sz += processed.get("n_sz", 0)
            total_processed_hc += processed.get("n_hc", 0)
        elif "error" in processed:
            proc_status = "ERROR"
        else:
            proc_status = "NO"

        # SZ/HC count
        if processed["exists"] and "n_sz" in processed:
            sz_hc = f"{processed['n_sz']}/{processed['n_hc']}"
        else:
            sz_hc = f"{meta['expected_sz']}/{meta['expected_hc']}*"

        # Next step recommendation
        if proc_status == "YES":
            next_step = "Ready for training"
        elif raw_status in ("YES", "PARTIAL"):
            next_step = "Run preprocessing"
        elif meta["access"] == "open" and name != "LA5c":
            next_step = "Download from OpenNeuro"
        elif meta["access"] == "registration":
            next_step = "Register for access"
        elif meta["access"] == "dua":
            next_step = "Sign DUA"
        elif meta["access"] == "nda_controlled":
            next_step = "NDA + IRB required"
        elif meta["access"] == "restricted":
            next_step = "Contact PI"
        else:
            next_step = "Unknown"

        print(f"  {name:12s} | {meta['access']:8s} | {raw_status:8s} | {proc_status:10s} | {sz_hc:>12s} | {next_step}")

        # Verbose details
        if args.verbose:
            if raw["exists"]:
                print(f"  {'':12s} |   Raw: {raw['nii_count']} NIfTI, {raw['tsv_count']} TSV/CSV, {raw['mat_count']} MAT files")
                if raw.get("has_participants"):
                    print(f"  {'':12s} |   participants.tsv: found")
                else:
                    print(f"  {'':12s} |   participants.tsv: MISSING")
            if processed["exists"] and "error" not in processed:
                print(f"  {'':12s} |   Processed: {processed.get('n_rois', '?')} ROIs, FDT={'yes' if processed.get('has_fdt') else 'no'}")

    # Summary
    print(f"\n  {'='*80}")
    print(f"  Summary:")
    print(f"    Raw data downloaded: {total_raw}/{len(DATASETS)} sites")
    print(f"    Preprocessed: {total_processed}/{len(DATASETS)} sites")
    print(f"    Processed samples: {total_processed_sz} SZ + {total_processed_hc} HC = {total_processed_sz + total_processed_hc}")
    print(f"    Potential total: {sum(d['expected_sz'] for d in DATASETS.values())} SZ + {sum(d['expected_hc'] for d in DATASETS.values())} HC")

    # Recommendations
    print(f"\n  Recommendations:")
    missing_open = [n for n, d in DATASETS.items() if d["access"] == "open" and n != "LA5c"
                    and not check_raw_status(data_dir, n)["has_data"]]
    if missing_open:
        print(f"    1. Download open-access datasets: {', '.join(missing_open)}")
        print(f"       python scripts/download_all_datasets.py --data_dir {data_dir}")

    unprocessed = [n for n in DATASETS
                   if check_raw_status(data_dir, n)["has_data"]
                   and not check_processed_status(data_dir, n)["exists"]]
    if unprocessed:
        print(f"    2. Preprocess downloaded datasets: {', '.join(unprocessed)}")
        print(f"       python scripts/preprocess_all_datasets.py --data_dir {data_dir} --compute_fdt")

    if total_processed > 1:
        if not multisite_path.exists():
            print(f"    3. Integrate all processed datasets for federated training:")
            print(f"       python scripts/integrate_datasets.py --data_dir {data_dir}")
        else:
            print(f"    3. Run federated training:")
            print(f"       python experiments/final_training.py --data_dir {data_dir}/processed")

    print()


if __name__ == "__main__":
    main()
