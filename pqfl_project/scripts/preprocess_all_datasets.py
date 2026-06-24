#!/usr/bin/env python3
"""Enhanced preprocessing pipeline for ALL PQFL Schizophrenia datasets.

This script extends the base preprocess_real_data.py with:
  - Automatic dataset detection and site configuration
  - Dataset-specific preprocessing parameters (TR, confound strategy, etc.)
  - Support for TCP2025 (6 rest runs, HCP-style multiband)
  - Support for SPINS (3 acquisition sites, 0.8s TR)
  - Support for COBRE (CSV phenotype, non-BIDS structure)
  - Support for SRPBS (precomputed FC matrices in .mat format)
  - Robust error handling and progress tracking
  - Resume capability (skips already-processed sites)
  - Comprehensive summary report

Pipeline steps:
  1. Scan data directory for downloaded datasets
  2. Auto-detect site configuration
  3. Load participants.tsv (or CSV for COBRE)
  4. Load preprocessed fMRI (fMRIPrep output) or raw BOLD
  5. Apply confound regression + band-pass filtering
  6. Extract ROI time series (Schaefer 100-parcel)
  7. Compute SPD functional connectivity matrices
  8. Extract FDT (Frequency-Dependent Topology) features
  9. Save per-site .npz files compatible with federated training

Usage:
    # Preprocess all detected datasets
    python scripts/preprocess_all_datasets.py --data_dir ./data --compute_fdt

    # Preprocess specific sites only
    python scripts/preprocess_all_datasets.py --data_dir ./data --sites TCP2025 SPINS --compute_fdt

    # Use precomputed FC matrices (for SRPBS)
    python scripts/preprocess_all_datasets.py --data_dir ./data --use_precomputed_fc

    # Skip already-processed sites
    python scripts/preprocess_all_datasets.py --data_dir ./data --compute_fdt --skip_existing

Prerequisites:
    pip install nilearn nibabel scipy numpy
"""

import argparse
import sys
import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pqfl.data.fmri_pipeline import FMRIPipeline, FMRIConfig
from pqfl.data.fc_construction import FCConstructor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("preprocess_all_datasets.log"),
    ],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Extended site registry with dataset-specific preprocessing parameters
# ──────────────────────────────────────────────────────────────────────────────

SITE_REGISTRY = {
    "TCP2025": {
        "site_id": 6,
        "role": "validation",
        "openneuro_id": "ds005237",
        "source": "https://openneuro.org/datasets/ds005237",
        "diagnosis_col": "Group",
        "alt_diagnosis_cols": ["diagnosis", "group", "subject_type", "diagnosis_verbose"],
        "sz_labels": ["Schizophrenia", "SZ", "schizophrenia", "SCZ", "patient", "Patient"],
        "hc_labels": ["Control", "HC", "Healthy", "control", "healthy control", "CON"],
        "task_name": "rest",
        "space": None,
        "n_expected_sz": 40,
        "n_expected_hc": 93,
        "cross_disorder": True,
        "tr": 0.8,
        "n_rest_runs": 6,
        "confound_strategy": "simple",
        "bandpass_low": 0.009,
        "bandpass_high": 0.08,
        "fd_threshold": 0.5,
        "min_timepoints": 100,
        "concatenate_runs": True,
    },
    "SPINS": {
        "site_id": 7,
        "role": "training",
        "openneuro_id": "ds003011",
        "source": "https://openneuro.org/datasets/ds003011",
        "diagnosis_col": "diagnosis",
        "alt_diagnosis_cols": ["Diagnosis", "group", "subject_type", "clinical_diagnosis"],
        "sz_labels": ["Schizophrenia", "SZ", "schizophrenia", "SCZ"],
        "hc_labels": ["Control", "HC", "control", "Healthy"],
        "task_name": "rest",
        "space": None,
        "n_expected_sz": 94,
        "n_expected_hc": 94,
        "multi_site": True,
        "site_col": "site",
        "tr": 0.8,
        "n_rest_runs": 1,
        "confound_strategy": "simple",
        "bandpass_low": 0.009,
        "bandpass_high": 0.08,
        "fd_threshold": 0.5,
        "min_timepoints": 100,
        "concatenate_runs": False,
    },
    "COBRE": {
        "site_id": 0,
        "role": "training",
        "openneuro_id": None,
        "source": "https://coins.trendscenter.org/ (Study=COBRE)",
        "diagnosis_col": "Subject Type",
        "alt_diagnosis_cols": ["Diagnosis", "diagnosis", "group", "subject_type"],
        "sz_labels": ["Patient", "Schizophrenia", "SZ", "schizophrenia"],
        "hc_labels": ["Control", "Healthy", "HC", "control"],
        "task_name": "rest",
        "space": None,
        "n_expected_sz": 72,
        "n_expected_hc": 74,
        "tr": 2.0,
        "n_rest_runs": 1,
        "confound_strategy": "simple",
        "bandpass_low": 0.01,
        "bandpass_high": 0.08,
        "fd_threshold": 0.5,
        "min_timepoints": 50,
        "concatenate_runs": False,
        "alt_phenotype_file": "COBRE_phenotypic_data.csv",
    },
    "LA5c": {
        "site_id": 3,
        "role": "training",
        "openneuro_id": "ds000030",
        "source": "https://openneuro.org/datasets/ds000030",
        "diagnosis_col": "diagnosis",
        "alt_diagnosis_cols": ["Diagnosis", "group", "subject_type"],
        "sz_labels": ["SCHZ", "Schizophrenia", "SZ", "schizophrenia"],
        "hc_labels": ["CONTROL", "Control", "HC", "control", "Healthy"],
        "task_name": "rest",
        "space": None,
        "n_expected_sz": 50,
        "n_expected_hc": 127,
        "tr": 2.0,
        "n_rest_runs": 1,
        "confound_strategy": "simple",
        "bandpass_low": 0.01,
        "bandpass_high": 0.08,
        "fd_threshold": 0.5,
        "min_timepoints": 50,
        "concatenate_runs": False,
    },
    "SRPBS": {
        "site_id": 4,
        "role": "training",
        "openneuro_id": None,
        "source": "https://bicr-resource.atr.jp/srpbsfc",
        "diagnosis_col": "diagnosis",
        "alt_diagnosis_cols": ["Diagnosis", "subject_type", "condition"],
        "sz_labels": ["SCZ", "Schizophrenia", "SZ", "schizophrenia"],
        "hc_labels": ["HC", "Control", "Healthy", "control"],
        "task_name": "rest",
        "space": None,
        "n_expected_sz": 146,
        "n_expected_hc": 800,
        "has_precomputed_fc": True,
        "fc_format": "mat",
    },
    "MCIC": {
        "site_id": 2,
        "role": "training",
        "openneuro_id": None,
        "source": "https://coins.trendscenter.org/ (Study=MCICShare)",
        "diagnosis_col": "diagnosis",
        "alt_diagnosis_cols": ["Diagnosis", "group", "subject_type", "clinical_diagnosis"],
        "sz_labels": ["Schizophrenia", "SZ", "Patient", "schizophrenia"],
        "hc_labels": ["Control", "HC", "Healthy", "control"],
        "task_name": "rest",
        "space": None,
        "n_expected_sz": 146,
        "n_expected_hc": 160,
        "tr": 2.0,
        "n_rest_runs": 1,
        "confound_strategy": "simple",
        "bandpass_low": 0.01,
        "bandpass_high": 0.08,
        "fd_threshold": 0.5,
        "min_timepoints": 50,
        "concatenate_runs": False,
    },
    "BSNIP2": {
        "site_id": 5,
        "role": "validation",
        "openneuro_id": None,
        "source": "https://nda.nih.gov/edit_collection.html?id=2165",
        "diagnosis_col": "diagnosis",
        "alt_diagnosis_cols": ["Diagnosis", "group", "subject_type", "biotype"],
        "sz_labels": ["Schizophrenia", "SZ", "schizophrenia", "SZP"],
        "hc_labels": ["Control", "HC", "Healthy", "control", "CON"],
        "task_name": "rest",
        "space": None,
        "n_expected_sz": 150,
        "n_expected_hc": 223,
        "tr": 2.0,
        "n_rest_runs": 1,
        "confound_strategy": "simple",
        "bandpass_low": 0.01,
        "bandpass_high": 0.08,
        "fd_threshold": 0.5,
        "min_timepoints": 50,
        "concatenate_runs": False,
    },
    "FBIRN": {
        "site_id": 1,
        "role": "training",
        "openneuro_id": None,
        "source": "Contact Dr. Theo van Erp (tvanerp@hs.uci.edu)",
        "diagnosis_col": "diagnosis",
        "alt_diagnosis_cols": ["Diagnosis", "group", "subject_type"],
        "sz_labels": ["Schizophrenia", "SZ", "Patient", "schizophrenia"],
        "hc_labels": ["Control", "HC", "Healthy", "control"],
        "task_name": "rest",
        "space": None,
        "n_expected_sz": 176,
        "n_expected_hc": 186,
        "tr": 2.0,
        "n_rest_runs": 1,
        "confound_strategy": "simple",
        "bandpass_low": 0.01,
        "bandpass_high": 0.08,
        "fd_threshold": 0.5,
        "min_timepoints": 50,
        "concatenate_runs": False,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# BIDS parsing utilities (enhanced)
# ──────────────────────────────────────────────────────────────────────────────

def parse_participants_tsv(tsv_path: Path) -> Dict[str, Dict]:
    """Parse BIDS participants.tsv to get subject metadata."""
    import csv
    subjects = {}
    with open(tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            sub_id = row.get('participant_id', '')
            if not sub_id.startswith('sub-'):
                sub_id = f"sub-{sub_id}"
            subjects[sub_id] = row
    return subjects


def parse_csv_phenotype(csv_path: Path) -> Dict[str, Dict]:
    """Parse CSV phenotypic file (for COBRE)."""
    import csv
    subjects = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sub_id = row.get('participant_id', row.get('Subject_ID', row.get('subject_id', '')))
            if not sub_id.startswith('sub-'):
                sub_id = f"sub-{sub_id}"
            subjects[sub_id] = row
    return subjects


def get_diagnosis_label(subject_meta: Dict, site_config: Dict) -> Optional[int]:
    """Extract binary diagnosis label (0=HC, 1=SZ) from subject metadata."""
    diag_col = site_config.get("diagnosis_col", "diagnosis")
    diagnosis = subject_meta.get(diag_col, "").strip()

    if not diagnosis:
        alt_cols = site_config.get("alt_diagnosis_cols", [])
        fallback_cols = ["diagnosis", "Diagnosis", "group", "Group", "clinical_diagnosis",
                        "diagnosis_verbose", "subject_type", "Subject Type", "pheno_diagnosis"]
        for alt_col in alt_cols + fallback_cols:
            diagnosis = subject_meta.get(alt_col, "").strip()
            if diagnosis:
                break

    if not diagnosis:
        return None

    diagnosis_lower = diagnosis.lower().strip()

    for label in site_config["sz_labels"]:
        if label.lower() == diagnosis_lower or label.lower() in diagnosis_lower:
            return 1

    for label in site_config["hc_labels"]:
        if label.lower() == diagnosis_lower or label.lower() in diagnosis_lower:
            return 0

    return None


def find_bold_files(
    subject_dir: Path,
    task_name: str = "rest",
    prefer_preprocessed: bool = True,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Find BOLD and confound files for a subject."""
    func_dir = subject_dir / "func"
    if not func_dir.exists():
        return None, None

    bold_files = sorted(func_dir.glob(f"*task-{task_name}*bold*.nii*"))
    if not bold_files:
        bold_files = sorted(func_dir.glob(f"*task-{task_name}*_preproc.nii*"))
    if not bold_files:
        bold_files = sorted(func_dir.glob("*bold*.nii*"))

    if not bold_files:
        return None, None

    bold_path = None
    if prefer_preprocessed:
        for f in bold_files:
            if "space-MNI" in f.name and "preproc" in f.name:
                bold_path = f
                break
        if bold_path is None:
            for f in bold_files:
                if "space-MNI" in f.name:
                    bold_path = f
                    break

    if bold_path is None:
        bold_path = bold_files[0]

    confounds_path = None
    confound_patterns = [
        f"*task-{task_name}*desc-confounds_timeseries.tsv",
        "*desc-confounds_timeseries.tsv",
        f"*task-{task_name}*confounds.tsv",
        "*confounds.tsv",
    ]
    for pattern in confound_patterns:
        matches = list(func_dir.glob(pattern))
        if matches:
            confounds_path = matches[0]
            break

    return bold_path, confounds_path


def load_confounds(confounds_path: Path, strategy: str = "simple") -> Optional[np.ndarray]:
    """Load and select confound regressors from fMRIPrep TSV."""
    import csv
    try:
        with open(confounds_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)
    except Exception as e:
        logger.warning(f"Failed to load confounds: {e}")
        return None

    if not rows:
        return None

    if strategy == "simple":
        cols = [
            "csf", "white_matter", "global_signal",
            "trans_x", "trans_y", "trans_z",
            "rot_x", "rot_y", "rot_z",
        ]
        deriv_cols = [f"{c}_derivative1" for c in cols[:3]]
        cols += deriv_cols
    else:
        cols = ["csf", "white_matter", "global_signal",
                "trans_x", "trans_y", "trans_z",
                "rot_x", "rot_y", "rot_z"]

    available_cols = [c for c in cols if c in rows[0]]
    if not available_cols:
        return None

    confounds = np.zeros((len(rows), len(available_cols)))
    for i, row in enumerate(rows):
        for j, col in enumerate(available_cols):
            try:
                val = float(row[col])
                confounds[i, j] = val if val == val else 0.0
            except (ValueError, TypeError):
                confounds[i, j] = 0.0

    return confounds


# ──────────────────────────────────────────────────────────────────────────────
# Site preprocessing (enhanced with dataset-specific params)
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_site(
    site_name: str,
    site_dir: Path,
    output_dir: Path,
    n_rois: int = 100,
    yeo_networks: int = 7,
    compute_fdt: bool = True,
    n_fdt_features: int = 20,
    skip_existing: bool = False,
) -> Optional[Dict]:
    """Preprocess a single site's fMRI data into FC matrices."""
    site_config = SITE_REGISTRY.get(site_name)
    if site_config is None:
        logger.error(f"Unknown site: {site_name}")
        return None

    # Check if already processed
    output_path = output_dir / f"{site_name}_processed.npz"
    if skip_existing and output_path.exists():
        logger.info(f"  SKIP: {site_name} already processed ({output_path})")
        data = np.load(output_path, allow_pickle=True)
        return {
            "site_name": site_name,
            "site_id": site_config["site_id"],
            "role": site_config["role"],
            "n_samples": int(data["n_samples"]),
            "n_sz": int(data["n_sz"]),
            "n_hc": int(data["n_hc"]),
            "output_path": str(output_path),
            "status": "skipped_existing",
        }

    # Handle precomputed FC (SRPBS)
    if site_config.get("has_precomputed_fc"):
        return preprocess_precomputed_fc(site_name, site_dir, output_dir, n_rois)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Processing site: {site_name} (ID={site_config['site_id']})")
    logger.info(f"  Directory: {site_dir}")
    logger.info(f"{'='*60}")

    # Step 1: Parse participants
    subjects = {}

    alt_pheno = site_config.get("alt_phenotype_file")
    if alt_pheno:
        csv_path = site_dir / alt_pheno
        if csv_path.exists():
            logger.info(f"  Loading CSV phenotype: {csv_path}")
            subjects = parse_csv_phenotype(csv_path)

    if not subjects:
        participants_tsv = site_dir / "participants.tsv"
        if participants_tsv.exists():
            subjects = parse_participants_tsv(participants_tsv)

    if not subjects:
        phenotype_tsv = site_dir / "phenotype" / "demos.tsv"
        if phenotype_tsv.exists():
            subjects = parse_participants_tsv(phenotype_tsv)

    if not subjects:
        logger.error(f"No participants file found in {site_dir}")
        return None

    logger.info(f"  Found {len(subjects)} subjects in participant data")

    # Step 2: Initialize pipeline with dataset-specific params
    tr = site_config.get("tr", 2.0)
    fmri_config = FMRIConfig(
        parcellation="schaefer",
        n_rois=n_rois,
        yeo_networks=yeo_networks,
        tr=tr,
        bandpass_low=site_config.get("bandpass_low", 0.01),
        bandpass_high=site_config.get("bandpass_high", 0.08),
        fd_threshold=site_config.get("fd_threshold", 0.5),
        confound_strategy=site_config.get("confound_strategy", "simple"),
        standardize=True,
        detrend=True,
    )
    pipeline = FMRIPipeline(fmri_config)
    fc_constructor = FCConstructor(
        n_rois=n_rois,
        regularization_lambda=1e-3,
        fc_method="pearson",
    )

    # Step 3: Find subject directories
    subject_dirs = sorted([d for d in site_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    if not subject_dirs:
        fmriprep_dir = site_dir / "derivatives" / "fmriprep"
        if fmriprep_dir.exists():
            subject_dirs = sorted([d for d in fmriprep_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    if not subject_dirs:
        deriv_dir = site_dir / "derivatives"
        if deriv_dir.exists():
            subject_dirs = sorted([d for d in deriv_dir.glob("*/sub-*") if d.is_dir()])

    is_fmriprep = (site_dir / "derivatives" / "fmriprep").exists()
    min_timepoints = site_config.get("min_timepoints", 50)

    logger.info(f"  Found {len(subject_dirs)} subject directories")
    logger.info(f"  fMRIPrep data: {is_fmriprep}")
    logger.info(f"  TR: {tr}s, Min timepoints: {min_timepoints}")

    # Step 4: Process each subject
    fc_matrices = []
    labels = []
    fdt_features_list = []
    subject_ids = []
    skipped = 0
    processed = 0
    failed = 0

    start_time = time.time()

    for i, subj_dir in enumerate(subject_dirs):
        sub_id = subj_dir.name
        subject_meta = subjects.get(sub_id, {})

        label = get_diagnosis_label(subject_meta, site_config)
        if label is None:
            skipped += 1
            continue

        # Find BOLD files
        bold_path, confounds_path = find_bold_files(
            subj_dir,
            task_name=site_config.get("task_name", "rest"),
        )

        if bold_path is None:
            logger.debug(f"  [{i+1}/{len(subject_dirs)}] Skip {sub_id}: no BOLD file")
            skipped += 1
            continue

        # Handle multiple rest runs (TCP has 6 runs)
        if is_fmriprep:
            func_dir = subj_dir / "func"
            bold_candidates = sorted(func_dir.glob("*task-rest*preproc.nii*"))
            mni_preproc = [f for f in bold_candidates if "space-MNI" in f.name]
            if mni_preproc:
                all_bold_files = mni_preproc
            else:
                all_bold_files = [bold_path]
        else:
            all_bold_files = sorted(
                (subj_dir / "func").glob(f"*task-{site_config.get('task_name', 'rest')}*_bold*.nii*")
            )
            mni_bold = [f for f in all_bold_files if "space-MNI" in f.name]
            if mni_bold:
                all_bold_files = mni_bold

        try:
            import nibabel as nib

            all_run_ts = []

            for run_bold_path in all_bold_files:
                run_confounds_path = None
                confound_stem = run_bold_path.name.replace("_bold.nii.gz", "").replace("_bold.nii", "")
                confound_pattern = f"{confound_stem}_desc-confounds_timeseries.tsv"
                run_conf_matches = list((subj_dir / "func").glob(confound_pattern))
                if run_conf_matches:
                    run_confounds_path = run_conf_matches[0]

                run_confounds = None
                if run_confounds_path is not None:
                    run_confounds = load_confounds(run_confounds_path, strategy=fmri_config.confound_strategy)

                fmri_img = nib.load(str(run_bold_path))

                # Auto-detect TR
                tr_detected = fmri_img.header.get_zooms()[-1]
                if isinstance(tr_detected, (int, float)) and 0.5 < tr_detected < 5.0:
                    fmri_config.tr = tr_detected

                run_ts = pipeline.extract_time_series(fmri_img, confounds=run_confounds, is_fmriprep=is_fmriprep)

                if run_ts.shape[0] == 0:
                    continue

                if run_confounds is not None:
                    fd = pipeline.compute_fd(run_confounds)
                    run_ts, valid_mask = pipeline.scrub_timepoints(run_ts, fd)

                if run_ts.shape[0] >= min_timepoints // 2:
                    all_run_ts.append(run_ts)

            if not all_run_ts:
                skipped += 1
                continue

            # Concatenate runs
            if site_config.get("concatenate_runs", False):
                time_series = np.concatenate(all_run_ts, axis=0)
            else:
                time_series = max(all_run_ts, key=lambda x: x.shape[0])

            if time_series.shape[0] < min_timepoints:
                logger.debug(f"  Skip {sub_id}: {time_series.shape[0]} timepoints < {min_timepoints}")
                skipped += 1
                continue

            # Compute FC and FDT
            fc_matrix = fc_constructor.compute_static_fc(time_series)

            if compute_fdt:
                fdt = fc_constructor.compute_fdt_features(
                    time_series, n_top=n_fdt_features, tr=fmri_config.tr
                )
                fdt_features_list.append(fdt)

            fc_matrices.append(fc_matrix)
            labels.append(label)
            subject_ids.append(sub_id)
            processed += 1

            # Progress reporting
            if processed % 10 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                logger.info(f"  Progress: {processed} processed, {skipped} skipped, {failed} failed")

        except Exception as e:
            logger.warning(f"  Error processing {sub_id}: {e}")
            failed += 1
            continue

    # Step 5: Save results
    if len(fc_matrices) == 0:
        logger.error(f"  No subjects successfully processed for {site_name}!")
        return None

    fc_matrices = np.stack(fc_matrices)
    labels = np.array(labels)
    fdt_arr = np.stack(fdt_features_list) if compute_fdt and fdt_features_list else None

    save_dict = {
        "fc_matrices": fc_matrices,
        "labels": labels,
        "subject_ids": np.array(subject_ids),
        "site_id": site_config["site_id"],
        "site_name": site_name,
        "n_rois": n_rois,
        "n_samples": len(labels),
        "n_sz": int(labels.sum()),
        "n_hc": int((1 - labels).sum()),
        "role": site_config["role"],
        "tr": fmri_config.tr,
        "preprocessing_params": json.dumps({
            "bandpass": f"{fmri_config.bandpass_low}-{fmri_config.bandpass_high} Hz",
            "fd_threshold": fmri_config.fd_threshold,
            "confound_strategy": fmri_config.confound_strategy,
            "parcellation": f"schaefer_{n_rois}",
        }),
    }
    if fdt_arr is not None:
        save_dict["fdt_features"] = fdt_arr

    np.savez_compressed(output_path, **save_dict)

    elapsed = time.time() - start_time
    logger.info(
        f"  Site {site_name}: {processed} processed, {skipped} skipped, {failed} failed "
        f"({int(labels.sum())} SZ / {int((1-labels).sum())} HC) in {elapsed/60:.1f} min"
    )
    logger.info(f"  Saved to: {output_path}")

    return {
        "site_name": site_name,
        "site_id": site_config["site_id"],
        "role": site_config["role"],
        "n_samples": len(labels),
        "n_sz": int(labels.sum()),
        "n_hc": int((1-labels).sum()),
        "n_skipped": skipped,
        "n_failed": failed,
        "output_path": str(output_path),
        "processing_time_sec": elapsed,
    }


def preprocess_precomputed_fc(
    site_name: str,
    site_dir: Path,
    output_dir: Path,
    n_rois: int = 100,
) -> Optional[Dict]:
    """Preprocess a site with precomputed FC matrices (SRPBS)."""
    site_config = SITE_REGISTRY.get(site_name)
    if site_config is None:
        return None

    logger.info(f"\n  Processing site (precomputed FC): {site_name}")

    subjects = {}
    participants_tsv = site_dir / "participants.tsv"
    if participants_tsv.exists():
        subjects = parse_participants_tsv(participants_tsv)

    fc_data = None
    fc_path = None

    search_patterns = [
        "**/fc_matrices.npz", "**/fc_matrices.npy",
        "**/connectivity*.npz", "**/connectivity*.npy",
        "**/correlation*.npz", "**/correlation*.npy",
        "**/*FC*.mat", "**/*functional_connectivity*.npy",
        "**/*.mat", "**/SUBINFO_*.tsv",
    ]

    for pattern in search_patterns:
        matches = list(site_dir.glob(pattern))
        if matches:
            fc_path = matches[0]
            break

    if fc_path is None:
        logger.error(f"No precomputed FC files found in {site_dir}")
        return None

    logger.info(f"  Loading FC from: {fc_path}")

    try:
        if fc_path.suffix == ".npz":
            data = np.load(fc_path, allow_pickle=True)
            for key in ["fc_matrices", "fc", "connectivity", "data", "matrices"]:
                if key in data:
                    fc_data = data[key]
                    break
            if fc_data is None:
                fc_data = list(data.values())[0]
        elif fc_path.suffix == ".npy":
            fc_data = np.load(fc_path, allow_pickle=True)
        elif fc_path.suffix == ".mat":
            from scipy.io import loadmat
            mat = loadmat(str(fc_path))
            for key in ["fc_matrices", "fc", "connectivity", "data"]:
                if key in mat:
                    fc_data = mat[key]
                    break
            if fc_data is None:
                for k, v in mat.items():
                    if not k.startswith("__") and isinstance(v, np.ndarray):
                        fc_data = v
                        break

        if fc_data is None:
            logger.error(f"Could not load FC data from {fc_path}")
            return None

        if fc_data.ndim == 3 and fc_data.shape[1] == fc_data.shape[2]:
            logger.info(f"  FC shape: {fc_data.shape}")
        elif fc_data.ndim == 2:
            fc_data = fc_data[np.newaxis, ...]
        else:
            logger.error(f"Unexpected FC shape: {fc_data.shape}")
            return None

    except Exception as e:
        logger.error(f"Error loading FC data: {e}")
        return None

    labels = None
    if subjects:
        labels = []
        for sub_id in sorted(subjects.keys()):
            label = get_diagnosis_label(subjects[sub_id], site_config)
            labels.append(label if label is not None else -1)
        labels = np.array(labels)

        if len(labels) != fc_data.shape[0]:
            logger.warning(f"  Subject count mismatch: participants={len(labels)}, FC={fc_data.shape[0]}")
            labels = None

    if labels is None:
        logger.error(f"Could not determine labels for {site_name}")
        return None

    valid_mask = labels >= 0
    fc_data = fc_data[valid_mask]
    labels = labels[valid_mask]

    from scipy import linalg as la
    for i in range(len(fc_data)):
        fc = fc_data[i]
        fc = 0.5 * (fc + fc.T)
        eigvals = la.eigvalsh(fc)
        if np.min(eigvals) < 1e-6:
            fc += (abs(np.min(eigvals)) + 1e-3) * np.eye(fc.shape[0])
        fc_data[i] = fc

    output_path = output_dir / f"{site_name}_processed.npz"
    np.savez_compressed(
        output_path,
        fc_matrices=fc_data,
        labels=labels,
        site_id=site_config["site_id"],
        site_name=site_name,
        n_rois=fc_data.shape[1],
        n_samples=len(labels),
        n_sz=int(labels.sum()),
        n_hc=int((1-labels).sum()),
        role=site_config["role"],
    )

    logger.info(f"  Site {site_name}: {len(labels)} subjects ({int(labels.sum())} SZ / {int((1-labels).sum())} HC)")

    return {
        "site_name": site_name,
        "site_id": site_config["site_id"],
        "role": site_config["role"],
        "n_samples": len(labels),
        "n_sz": int(labels.sum()),
        "n_hc": int((1-labels).sum()),
        "output_path": str(output_path),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced preprocessing for ALL PQFL Schizophrenia datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess all detected datasets with FDT features
  python scripts/preprocess_all_datasets.py --data_dir ./data --compute_fdt

  # Preprocess specific sites only
  python scripts/preprocess_all_datasets.py --data_dir ./data --sites TCP2025 SPINS --compute_fdt

  # Skip already-processed sites
  python scripts/preprocess_all_datasets.py --data_dir ./data --compute_fdt --skip_existing

  # Use precomputed FC matrices (for SRPBS)
  python scripts/preprocess_all_datasets.py --data_dir ./data --use_precomputed_fc
        """,
    )

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing downloaded datasets")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: <data_dir>/processed/)")
    parser.add_argument("--sites", nargs="+", default=None,
                        help="Specific sites to process (e.g., TCP2025 SPINS)")
    parser.add_argument("--n_rois", type=int, default=100,
                        help="Number of Schaefer ROIs (default: 100)")
    parser.add_argument("--yeo_networks", type=int, default=7, choices=[7, 17],
                        help="Number of Yeo networks (default: 7)")
    parser.add_argument("--compute_fdt", action="store_true",
                        help="Compute Frequency-Dependent Topology features")
    parser.add_argument("--use_precomputed_fc", action="store_true",
                        help="Look for pre-computed FC matrices instead of raw fMRI")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip sites that already have processed .npz files")
    parser.add_argument("--min_timepoints", type=int, default=None,
                        help="Override minimum timepoints after scrubbing (default: site-specific)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    if args.sites:
        sites_to_process = args.sites
    else:
        sites_to_process = []
        for name in SITE_REGISTRY:
            if (data_dir / name).exists():
                sites_to_process.append(name)

    if not sites_to_process:
        logger.error(f"No dataset directories found in {data_dir}")
        logger.info(f"Expected subdirectories: {list(SITE_REGISTRY.keys())}")
        logger.info("Download datasets first:")
        logger.info("  python scripts/download_all_datasets.py --data_dir ./data")
        return

    logger.info(f"Sites to process: {sites_to_process}")
    logger.info(f"Output directory: {output_dir}")

    results = []
    for site_name in sites_to_process:
        site_dir = data_dir / site_name

        if not site_dir.exists():
            logger.warning(f"Directory not found: {site_dir} -- skipping")
            continue

        if args.use_precomputed_fc or site_name == "SRPBS" or SITE_REGISTRY.get(site_name, {}).get("has_precomputed_fc"):
            result = preprocess_precomputed_fc(
                site_name=site_name,
                site_dir=site_dir,
                output_dir=output_dir,
                n_rois=args.n_rois,
            )
        else:
            result = preprocess_site(
                site_name=site_name,
                site_dir=site_dir,
                output_dir=output_dir,
                n_rois=args.n_rois,
                yeo_networks=args.yeo_networks,
                compute_fdt=args.compute_fdt,
                skip_existing=args.skip_existing,
            )

        if result is not None:
            results.append(result)

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_rois": args.n_rois,
            "yeo_networks": args.yeo_networks,
            "compute_fdt": args.compute_fdt,
        },
        "sites": results,
        "total_samples": sum(r["n_samples"] for r in results),
        "total_sz": sum(r["n_sz"] for r in results),
        "total_hc": sum(r["n_hc"] for r in results),
    }

    summary_path = output_dir / "preprocessing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*74}")
    print(f"  PREPROCESSING COMPLETE")
    print(f"{'='*74}\n")

    print(f"  {'Site':12s} | {'Role':10s} | {'SZ':>4s} | {'HC':>4s} | {'Total':>5s} | Status")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*4}-+-{'-'*4}-+-{'-'*5}-+-{'-'*20}")

    for r in results:
        status = r.get("status", "success")
        print(f"  {r['site_name']:12s} | {r['role']:10s} | {r['n_sz']:4d} | {r['n_hc']:4d} | {r['n_samples']:5d} | {status}")

    total_sz = sum(r["n_sz"] for r in results)
    total_hc = sum(r["n_hc"] for r in results)
    print(f"  {'TOTAL':12s} | {'':10s} | {total_sz:4d} | {total_hc:4d} | {total_sz+total_hc:5d} |")

    print(f"\n  Output directory: {output_dir}")
    print(f"  Summary saved to: {summary_path}")

    print(f"""
========================================
NEXT STEPS:
========================================

1. Verify processed data

2. Integrate all sites for federated training:
   python scripts/integrate_datasets.py --data_dir {output_dir}

3. Train model:
   python experiments/final_training.py --data_dir {output_dir}
""")


if __name__ == "__main__":
    main()
