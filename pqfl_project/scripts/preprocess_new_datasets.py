#!/usr/bin/env python3
"""Preprocess the 5 new datasets for PQFL Phase 2.

This script:
  1. Loads each of the 5 new datasets (BrainLat, Transdiagnostic, Depression,
     Kaggle Psychosis, MLSP) using pqfl.data.dataset_adapters
  2. Computes FC matrices (or loads pre-extracted FNC for Kaggle/MLSP)
  3. Saves per-site .npz files in the format expected by experiments/final_training.py:
       {site_name}_processed.npz containing:
         fc_matrices, labels, site_id, site_name, subject_ids

Usage:
    # Preprocess all 5 new datasets (default: Schaefer 100 ROI)
    python scripts/preprocess_new_datasets.py --data_dir F:\\PQFL_Schizophrenia_Full_Project\\pqfl_project\\data

    # Only preprocess Kaggle + MLSP (fast — no BOLD processing needed)
    python scripts/preprocess_new_datasets.py --data_dir F:\\PQFL_Schizophrenia_Full_Project\\pqfl_project\\data --only kaggle mlsp

    # Only BrainLat (single site code, for debugging)
    python scripts/preprocess_new_datasets.py --data_dir ... --only brainlat --brainlat_site CLB

    # Use 200 ROI Schaefer instead of 100
    python scripts/preprocess_new_datasets.py --data_dir ... --n_rois 200

    # Output to a different directory
    python scripts/preprocess_new_datasets.py --data_dir ... --output_dir ./data/processed

Output format (per site):
    data/processed/BrainLat-CLB_processed.npz
    data/processed/BrainLat-COB_processed.npz
    ...
    data/processed/Transdiagnostic_processed.npz
    data/processed/Depression_processed.npz
    data/processed/KagglePsychosis_processed.npz
    data/processed/MLSP2014_processed.npz

Prerequisites:
    pip install nilearn nibabel scipy numpy pandas h5py scikit-learn
"""

import argparse
import sys
import logging
import time
from pathlib import Path
from datetime import datetime

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pqfl.data.dataset_adapters import (
    load_brainlat_site,
    load_transdiagnostic,
    load_depression,
    load_kaggle_psychosis,
    load_mlsp,
    load_all_new_datasets,
    BRAINLAT_SITES,
    LABEL_SZ, LABEL_HC, LABEL_OTHER, LABEL_BP,
    remap_labels_for_n_classes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("preprocess_new_datasets.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess 5 new PQFL datasets")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Root data directory containing the 8 dataset folders")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for .npz files (default: <data_dir>/processed)")
    parser.add_argument("--n_rois", type=int, default=100,
                       help="Number of ROIs (Schaefer atlas) — default 100")
    parser.add_argument("--only", type=str, nargs="+",
                       choices=["brainlat", "transdiag", "depression", "kaggle", "mlsp"],
                       default=None,
                       help="Only process specific datasets (default: all)")
    parser.add_argument("--brainlat_site", type=str, default=None,
                       choices=list(BRAINLAT_SITES.keys()),
                       help="Only process this BrainLat site code (for debugging)")
    parser.add_argument("--n_classes", type=int, default=3, choices=[2, 3, 4],
                       help="Label scheme: 2 (binary SZ/HC), 3 (SZ/HC/Other), 4 (SZ/HC/Other/BP)")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip sites whose output .npz already exists")
    return parser.parse_args()


def save_site_npz(site_data: dict, output_dir: Path, n_classes: int) -> Path:
    """Save a site dict as {site_name}_processed.npz."""
    site_name = site_data["site_name"]
    out_path = output_dir / f"{site_name}_processed.npz"

    # Remap labels for the chosen class scheme
    remapped_labels = remap_labels_for_n_classes(site_data["labels"], n_classes)

    # For 2-class, we need to also filter the FC matrices and subject_ids
    if n_classes == 2:
        from pqfl.data.dataset_adapters import LABEL_SZ, LABEL_HC
        mask = (site_data["labels"] == LABEL_SZ) | (site_data["labels"] == LABEL_HC)
        fc_matrices = site_data["fc_matrices"][mask]
        subject_ids = site_data["subject_ids"][mask]
        # For 2-class: remap SZ=0 → 1, HC=1 → 0 (so label=1 means SZ, the positive class)
        # to match the original pipeline convention
        remapped_labels = np.where(remapped_labels == LABEL_SZ, 1, 0)
    else:
        fc_matrices = site_data["fc_matrices"]
        subject_ids = site_data["subject_ids"]

    np.savez_compressed(
        out_path,
        fc_matrices=fc_matrices,
        labels=remapped_labels,
        site_id=site_data["site_id"],
        site_name=site_name,
        subject_ids=subject_ids,
        n_classes=n_classes,
        preprocessing_date=datetime.now().isoformat(),
        n_rois=site_data["fc_matrices"].shape[1],
    )

    n_total = len(remapped_labels)
    logger.info(f"Saved {out_path.name}: {n_total} subjects, "
               f"FC shape {fc_matrices.shape}, "
               f"label dist: {dict(zip(*np.unique(remapped_labels, return_counts=True)))}")
    return out_path


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PQFL Phase 2: Preprocessing 5 new datasets")
    logger.info("=" * 70)
    logger.info(f"Data dir:    {data_dir}")
    logger.info(f"Output dir:  {output_dir}")
    logger.info(f"N ROIs:      {args.n_rois}")
    logger.info(f"N classes:   {args.n_classes}")
    logger.info(f"Datasets:    {args.only or 'ALL'}")
    logger.info("=" * 70)

    only = args.only or ["brainlat", "transdiag", "depression", "kaggle", "mlsp"]
    start_time = time.time()
    saved_files = []
    skipped_files = []

    # ─── BrainLat ─────────────────────────────────────────────────────────
    if "brainlat" in only:
        brainlat_root = data_dir / "BrainLat"
        if not brainlat_root.exists():
            logger.warning(f"BrainLat dir not found: {brainlat_root}")
        else:
            site_codes = [args.brainlat_site] if args.brainlat_site else list(BRAINLAT_SITES.keys())
            for site_code in site_codes:
                site_info = BRAINLAT_SITES[site_code]
                out_path = output_dir / f"{site_info['site_name']}_processed.npz"
                if args.skip_existing and out_path.exists():
                    logger.info(f"[SKIP] {out_path.name} (already exists)")
                    skipped_files.append(out_path)
                    continue

                logger.info(f"\n--- BrainLat-{site_code} ---")
                t0 = time.time()
                try:
                    site_data = load_brainlat_site(brainlat_root, site_code, n_rois=args.n_rois)
                    if site_data is None:
                        logger.info(f"BrainLat-{site_code}: skipped (too few subjects)")
                        continue
                    save_site_npz(site_data, output_dir, args.n_classes)
                    saved_files.append(out_path)
                    logger.info(f"Time: {time.time()-t0:.1f}s")
                except Exception as e:
                    logger.error(f"BrainLat-{site_code} failed: {e}", exc_info=True)

    # ─── Transdiagnostic ─────────────────────────────────────────────────
    if "transdiag" in only:
        td_root = data_root = data_dir / "Transdiagnostic"
        if not td_root.exists():
            logger.warning(f"Transdiagnostic dir not found: {td_root}")
        else:
            out_path = output_dir / "Transdiagnostic_processed.npz"
            if args.skip_existing and out_path.exists():
                logger.info(f"[SKIP] {out_path.name} (already exists)")
                skipped_files.append(out_path)
            else:
                logger.info("\n--- Transdiagnostic ---")
                t0 = time.time()
                try:
                    site_data = load_transdiagnostic(td_root, n_rois=args.n_rois, use_preprocessed=True)
                    save_site_npz(site_data, output_dir, args.n_classes)
                    saved_files.append(out_path)
                    logger.info(f"Time: {time.time()-t0:.1f}s")
                except Exception as e:
                    logger.error(f"Transdiagnostic failed: {e}", exc_info=True)

    # ─── Depression ──────────────────────────────────────────────────────
    if "depression" in only:
        dep_root = data_dir / "Depression"
        if not dep_root.exists():
            logger.warning(f"Depression dir not found: {dep_root}")
        else:
            out_path = output_dir / "Depression_processed.npz"
            if args.skip_existing and out_path.exists():
                logger.info(f"[SKIP] {out_path.name} (already exists)")
                skipped_files.append(out_path)
            else:
                logger.info("\n--- Depression ---")
                t0 = time.time()
                try:
                    site_data = load_depression(dep_root, n_rois=args.n_rois)
                    save_site_npz(site_data, output_dir, args.n_classes)
                    saved_files.append(out_path)
                    logger.info(f"Time: {time.time()-t0:.1f}s")
                except Exception as e:
                    logger.error(f"Depression failed: {e}", exc_info=True)

    # ─── Kaggle Psychosis ────────────────────────────────────────────────
    if "kaggle" in only:
        kaggle_root = data_dir / "Kaggle_Psychosis_rsFMRI"
        if not kaggle_root.exists():
            logger.warning(f"Kaggle dir not found: {kaggle_root}")
        else:
            out_path = output_dir / "KagglePsychosis_processed.npz"
            if args.skip_existing and out_path.exists():
                logger.info(f"[SKIP] {out_path.name} (already exists)")
                skipped_files.append(out_path)
            else:
                logger.info("\n--- Kaggle Psychosis ---")
                t0 = time.time()
                try:
                    site_data = load_kaggle_psychosis(kaggle_root, n_target_rois=args.n_rois)
                    save_site_npz(site_data, output_dir, args.n_classes)
                    saved_files.append(out_path)
                    logger.info(f"Time: {time.time()-t0:.1f}s")
                except Exception as e:
                    logger.error(f"Kaggle failed: {e}", exc_info=True)

    # ─── MLSP ────────────────────────────────────────────────────────────
    if "mlsp" in only:
        mlsp_root = data_dir / "MLSP"
        if not mlsp_root.exists():
            logger.warning(f"MLSP dir not found: {mlsp_root}")
        else:
            out_path = output_dir / "MLSP2014_processed.npz"
            if args.skip_existing and out_path.exists():
                logger.info(f"[SKIP] {out_path.name} (already exists)")
                skipped_files.append(out_path)
            else:
                logger.info("\n--- MLSP ---")
                t0 = time.time()
                try:
                    site_data = load_mlsp(mlsp_root, n_target_rois=args.n_rois)
                    save_site_npz(site_data, output_dir, args.n_classes)
                    saved_files.append(out_path)
                    logger.info(f"Time: {time.time()-t0:.1f}s")
                except Exception as e:
                    logger.error(f"MLSP failed: {e}", exc_info=True)

    # ─── Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info(f"PREPROCESSING COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Saved:   {len(saved_files)} sites")
    logger.info(f"  Skipped: {len(skipped_files)} sites")
    logger.info("=" * 70)

    # Print site summary table
    if saved_files:
        logger.info("\nSite summary:")
        logger.info(f"  {'Site':<25} {'N':>5} {'SZ':>5} {'HC':>5} {'Other':>6} {'BP':>5}")
        logger.info(f"  {'-'*25} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*5}")
        total_n = 0
        for npz_path in saved_files:
            data = np.load(npz_path, allow_pickle=True)
            labels = data["labels"]
            site_name = str(data["site_name"])
            n = len(labels)
            n_sz = int((labels == LABEL_SZ).sum())
            n_hc = int((labels == LABEL_HC).sum())
            n_other = int((labels == LABEL_OTHER).sum())
            n_bp = int((labels == LABEL_BP).sum()) if args.n_classes == 4 else 0
            logger.info(f"  {site_name:<25} {n:>5} {n_sz:>5} {n_hc:>5} {n_other:>6} {n_bp:>5}")
            total_n += n
        logger.info(f"  {'-'*25} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*5}")
        logger.info(f"  {'TOTAL':<25} {total_n:>5}")

        logger.info(f"\nOutput files saved to: {output_dir}")


if __name__ == "__main__":
    main()
