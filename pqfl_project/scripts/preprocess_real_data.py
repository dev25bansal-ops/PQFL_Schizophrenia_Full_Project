#!/usr/bin/env python3
"""Real fMRI data preprocessing pipeline for PQFL.

This script takes downloaded OpenNeuro/NDA fMRI datasets and converts them
into FC matrices ready for federated quantum learning.

Pipeline steps:
  1. Scan BIDS directory for subjects
  2. Load preprocessed fMRI (fMRIPrep output) or raw BOLD
  3. Apply confound regression + band-pass filtering
  4. Extract ROI time series (Schaefer 100-parcel)
  5. Compute SPD functional connectivity matrices
  6. Extract FDT (Frequency-Dependent Topology) features
  7. Save per-site .npz files for training

Usage:
    python scripts/preprocess_real_data.py --data_dir /path/to/downloaded/data
    python scripts/preprocess_real_data.py --data_dir ./data --sites COBRE MCIC
    python scripts/preprocess_real_data.py --data_dir ./data --use_precomputed_fc

Prerequisites:
    pip install nilearn nibabel scipy numpy

Directory structure expected:
    data/
    ├── COBRE/
    │   ├── sub-01/
    │   │   ├── func/
    │   │   │   ├── sub-01_task-rest_bold.nii.gz
    │   │   │   └── sub-01_task-rest_desc-confounds_timeseries.tsv
    │   │   └── anat/...
    │   ├── sub-02/...
    │   ├── participants.tsv   (must have 'diagnosis' or 'group' column)
    │   └── dataset_description.json
    ├── FBIRN/...
    ├── MCIC/...
    └── ...
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pqfl.data.fmri_pipeline import FMRIPipeline, FMRIConfig
from pqfl.data.fc_construction import FCConstructor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Site metadata registry
# ──────────────────────────────────────────────────────────────────────────────

SITE_REGISTRY = {
    "COBRE": {
        "site_id": 0,
        "role": "training",
        "openneuro_id": None,  # NOT on OpenNeuro; use COINS Data Exchange
        "source": "https://coins.trendscenter.org/ (Study=COBRE)",
        "diagnosis_col": "Subject Type",  # COBRE phenotypic CSV column
        "alt_diagnosis_cols": ["Diagnosis", "diagnosis", "group", "subject_type"],
        "sz_labels": ["Patient", "Schizophrenia", "SZ", "schizophrenia"],
        "hc_labels": ["Control", "Healthy", "HC", "control"],
        "task_name": "rest",
        "space": None,
        "n_expected_sz": 72,
        "n_expected_hc": 74,
    },
    "FBIRN": {
        "site_id": 1,
        "role": "training",
        "openneuro_id": None,  # Not publicly downloadable; contact PI: tvanerp@hs.uci.edu
        "source": "Contact Dr. Theo van Erp (tvanerp@hs.uci.edu) for IRB access",
        "diagnosis_col": "diagnosis",
        "alt_diagnosis_cols": ["Diagnosis", "group", "subject_type"],
        "sz_labels": ["Schizophrenia", "SZ", "Patient", "schizophrenia"],
        "hc_labels": ["Control", "HC", "Healthy", "control"],
        "task_name": "rest",
        "space": None,
        "n_expected_sz": 176,
        "n_expected_hc": 186,
    },
    "MCIC": {
        "site_id": 2,
        "role": "training",
        "openneuro_id": None,  # NOT on OpenNeuro; use COINS Data Exchange
        "source": "https://coins.trendscenter.org/ (Study=MCICShare)",
        "diagnosis_col": "diagnosis",
        "alt_diagnosis_cols": ["Diagnosis", "group", "subject_type", "clinical_diagnosis"],
        "sz_labels": ["Schizophrenia", "SZ", "Patient", "schizophrenia"],
        "hc_labels": ["Control", "HC", "Healthy", "control"],
        "task_name": "rest",
        "space": None,
        "n_expected_sz": 146,
        "n_expected_hc": 160,
    },
    "LA5c": {
        "site_id": 3,
        "role": "training",
        "openneuro_id": "ds000030",  # UCLA CNP LA5c Study (VERIFIED)
        "source": "https://openneuro.org/datasets/ds000030",
        "diagnosis_col": "diagnosis",  # participants.tsv: SCHZ, CONTROL, BIPOLAR, ADHD
        "alt_diagnosis_cols": ["Diagnosis", "group", "subject_type"],
        "sz_labels": ["SCHZ", "Schizophrenia", "SZ", "schizophrenia"],
        "hc_labels": ["CONTROL", "Control", "HC", "control", "Healthy"],
        "task_name": "rest",
        "space": None,
        "n_expected_sz": 50,
        "n_expected_hc": 127,
    },
    "SRPBS": {
        "site_id": 4,
        "role": "training",
        "openneuro_id": None,  # NOT on OpenNeuro; use BICR ATR Japan
        "source": "https://bicr-resource.atr.jp/srpbsfc (precomputed FC, 175MB)",
        "diagnosis_col": "diagnosis",
        "alt_diagnosis_cols": ["Diagnosis", "subject_type", "condition"],
        "sz_labels": ["SCZ", "Schizophrenia", "SZ", "schizophrenia"],
        "hc_labels": ["HC", "Control", "Healthy", "control"],
        "task_name": "rest",
        "space": None,
        "n_expected_sz": 146,
        "n_expected_hc": 800,
        "has_precomputed_fc": True,
    },
    "BSNIP2": {
        "site_id": 5,
        "role": "validation",
        "openneuro_id": None,  # NIMH Data Archive only
        "source": "https://nda.nih.gov/edit_collection.html?id=2165",
        "diagnosis_col": "diagnosis",
        "alt_diagnosis_cols": ["Diagnosis", "group", "subject_type", "biotype"],
        "sz_labels": ["Schizophrenia", "SZ", "schizophrenia", "SZP"],
        "hc_labels": ["Control", "HC", "Healthy", "control", "CON"],
        "task_name": "rest",
        "space": None,
        "n_expected_sz": 150,
        "n_expected_hc": 223,
    },
    "TCP2025": {
        "site_id": 6,
        "role": "validation",
        "openneuro_id": "ds005237",  # Transdiagnostic Connectome Project (VERIFIED)
        "source": "https://openneuro.org/datasets/ds005237",
        "diagnosis_col": "Group",  # participants.tsv: patient/healthy control
        "alt_diagnosis_cols": ["diagnosis", "group", "subject_type", "diagnosis_verbose"],
        "sz_labels": ["Schizophrenia", "SZ", "schizophrenia", "SCZ", "patient", "Patient"],
        "hc_labels": ["Control", "HC", "Healthy", "control", "healthy control", "CON"],
        "task_name": "rest",
        "space": None,
        "n_expected_sz": 40,  # Approximate; transdiagnostic, filter by SCID
        "n_expected_hc": 93,
        "cross_disorder": True,  # Has SZ + BD + MDD + anxiety
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# BIDS parsing utilities
# ──────────────────────────────────────────────────────────────────────────────

def parse_participants_tsv(tsv_path: Path) -> Dict[str, Dict]:
    """Parse BIDS participants.tsv to get subject metadata.

    Returns:
        Dictionary mapping subject_id (e.g. 'sub-01') to metadata dict.
    """
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


def get_diagnosis_label(subject_meta: Dict, site_config: Dict) -> Optional[int]:
    """Extract binary diagnosis label (0=HC, 1=SZ) from subject metadata.

    Handles various column names across datasets:
      - LA5c (ds000030): 'diagnosis' col = SCHZ / CONTROL / BIPOLAR / ADHD
      - TCP (ds005237): 'Group' col = patient / healthy control
      - COBRE: 'Subject Type' col = Patient / Control
      - SRPBS: 'diagnosis' col = SCZ / HC
      - MCIC: 'diagnosis' col = Schizophrenia / Control
      - BSNIP2: 'diagnosis' col = SZ / HC / SAD / BDP

    Args:
        subject_meta: Row from participants.tsv.
        site_config: Site configuration from SITE_REGISTRY.

    Returns:
        0 (HC), 1 (SZ), or None if unknown / other disorder.
    """
    # Try primary diagnosis column first
    diag_col = site_config.get("diagnosis_col", "diagnosis")
    diagnosis = subject_meta.get(diag_col, "").strip()

    # Try alternative columns if primary is empty
    if not diagnosis:
        alt_cols = site_config.get("alt_diagnosis_cols", [])
        # Also try common fallback columns
        fallback_cols = ["diagnosis", "Diagnosis", "group", "Group", "clinical_diagnosis",
                        "diagnosis_verbose", "subject_type", "Subject Type", "pheno_diagnosis"]
        for alt_col in alt_cols + fallback_cols:
            diagnosis = subject_meta.get(alt_col, "").strip()
            if diagnosis:
                break

    if not diagnosis:
        return None

    diagnosis_stripped = diagnosis.strip()
    diagnosis_lower = diagnosis_stripped.lower()

    # Check SZ labels (exact match or substring)
    for label in site_config["sz_labels"]:
        label_lower = label.lower()
        if label_lower == diagnosis_lower or label_lower in diagnosis_lower:
            return 1

    # Check HC labels (exact match or substring)
    for label in site_config["hc_labels"]:
        label_lower = label.lower()
        if label_lower == diagnosis_lower or label_lower in diagnosis_lower:
            return 0

    # For cross-disorder datasets (TCP, LA5c, BSNIP2), return None for non-SZ/non-HC
    # This means BIPOLAR, ADHD, MDD etc. are excluded from binary classification
    return None


def find_bold_files(
    subject_dir: Path,
    task_name: str = "rest",
    prefer_preprocessed: bool = True,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Find BOLD and confound files for a subject.

    Prefers fMRIPrep preprocessed data if available.
    Handles naming patterns from both raw BIDS and fMRIPrep derivatives:
      - Raw: sub-01_task-rest_bold.nii.gz
      - fMRIPrep: sub-01_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz

    Args:
        subject_dir: Path to subject directory (e.g., data/COBRE/sub-01/).
        task_name: BIDS task name.
        prefer_preprocessed: Prefer fMRIPrep outputs.

    Returns:
        Tuple of (bold_path, confounds_path) or (None, None).
    """
    func_dir = subject_dir / "func"
    if not func_dir.exists():
        return None, None

    # Try multiple BOLD file patterns
    bold_files = sorted(func_dir.glob(f"*task-{task_name}*bold*.nii*"))
    if not bold_files:
        bold_files = sorted(func_dir.glob(f"*task-{task_name}*_preproc.nii*"))
    if not bold_files:
        bold_files = sorted(func_dir.glob("*bold*.nii*"))

    if not bold_files:
        return None, None

    bold_path = None
    if prefer_preprocessed:
        # Prefer MNI-space preprocessed (fMRIPrep output)
        for f in bold_files:
            if "space-MNI" in f.name and "preproc" in f.name:
                bold_path = f
                break
        if bold_path is None:
            # Fall back to any MNI-space file
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

    if confounds_path is None:
        logger.debug(f"  No confounds file found for {subject_dir.name} - will preprocess without confound regression")

    return bold_path, confounds_path


def load_confounds(confounds_path: Path, strategy: str = "simple") -> Optional[np.ndarray]:
    """Load and select confound regressors from fMRIPrep TSV.

    Args:
        confounds_path: Path to confounds.tsv file.
        strategy: Confound strategy.

    Returns:
        Confound array, shape (n_timepoints, n_regressors), or None.
    """
    import csv

    try:
        with open(confounds_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)
    except Exception as e:
        logger.warning(f"Failed to load confounds from {confounds_path}: {e}")
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
    elif strategy == "scrubbing":
        cols = [
            "csf", "white_matter", "global_signal",
            "trans_x", "trans_y", "trans_z",
            "rot_x", "rot_y", "rot_z",
            "framewise_displacement",
        ]
        deriv_cols = [f"{c}_derivative1" for c in cols[:3]]
        cols += deriv_cols
    elif strategy == "acompcor":
        cols = [
            "csf", "white_matter", "global_signal",
            "trans_x", "trans_y", "trans_z",
            "rot_x", "rot_y", "rot_z",
        ]
        for i in range(6):
            cols.append(f"w_comp_cor_{i:02d}")
            cols.append(f"c_comp_cor_{i:02d}")
    else:
        cols = ["csf", "white_matter", "global_signal",
                "trans_x", "trans_y", "trans_z",
                "rot_x", "rot_y", "rot_z"]

    available_cols = [c for c in cols if c in rows[0]]
    if not available_cols:
        logger.warning(f"No matching confound columns found. Available: {list(rows[0].keys())[:20]}")
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
# Per-site preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_site(
    site_name: str,
    site_dir: Path,
    output_dir: Path,
    n_rois: int = 100,
    yeo_networks: int = 7,
    confound_strategy: str = "simple",
    fd_threshold: float = 0.5,
    fc_method: str = "pearson",
    regularization_lambda: float = 1e-3,
    compute_fdt: bool = True,
    n_fdt_features: int = 20,
    min_timepoints: int = 50,
    tr_override: Optional[float] = None,
) -> Optional[Dict]:
    """Preprocess a single site's fMRI data into FC matrices.

    Args:
        site_name: Site name (e.g., 'COBRE').
        site_dir: Path to site's BIDS directory.
        output_dir: Path to save processed .npz files.
        n_rois: Number of Schaefer ROIs.
        yeo_networks: Number of Yeo networks.
        confound_strategy: Confound regression strategy.
        fd_threshold: Framewise displacement threshold (mm).
        fc_method: FC computation method.
        regularization_lambda: SPD regularization parameter.
        compute_fdt: Whether to compute FDT features.
        n_fdt_features: Number of FDT features.
        min_timepoints: Minimum acceptable timepoints after scrubbing.
        tr_override: Override TR value (None = auto-detect from NIfTI header).

    Returns:
        Dictionary with processing results, or None on failure.
    """
    site_config = SITE_REGISTRY.get(site_name)
    if site_config is None:
        logger.error(f"Unknown site: {site_name}")
        return None

    logger.info(f"Processing site: {site_name} (ID={site_config['site_id']})")

    # ── Step 1: Parse participants.tsv (or .csv for COBRE) ──
    subjects = {}
    participants_tsv = site_dir / "participants.tsv"

    # COBRE may use CSV instead of TSV
    if not participants_tsv.exists():
        participants_csv = site_dir / "COBRE_phenotypic_data.csv"
        if participants_csv.exists():
            logger.info(f"  Loading COBRE CSV phenotypic data")
            import csv
            with open(participants_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # COBRE CSV may use 'Subject_ID' or 'participant_id'
                    sub_id = row.get('participant_id', row.get('Subject_ID', row.get('subject_id', '')))
                    if not sub_id.startswith('sub-'):
                        sub_id = f"sub-{sub_id}"
                    subjects[sub_id] = row
            logger.info(f"  Found {len(subjects)} subjects in CSV")

    if not subjects:
        if not participants_tsv.exists():
            # Try phenotype/ subdirectory (TCP ds005237 structure)
            phenotype_tsv = site_dir / "phenotype" / "demos.tsv"
            if phenotype_tsv.exists():
                logger.info(f"  Loading phenotype/demos.tsv")
                subjects = parse_participants_tsv(phenotype_tsv)
            else:
                logger.error(f"No participants.tsv found in {site_dir}")
                return None
        else:
            subjects = parse_participants_tsv(participants_tsv)

    logger.info(f"  Found {len(subjects)} subjects in participant data")

    # ── Step 2: Initialize pipeline components ──
    fmri_config = FMRIConfig(
        parcellation="schaefer",
        n_rois=n_rois,
        yeo_networks=yeo_networks,
        tr=2.0,
        bandpass_low=0.01,
        bandpass_high=0.08,
        fd_threshold=fd_threshold,
        confound_strategy=confound_strategy,
        standardize=True,
        detrend=True,
    )
    pipeline = FMRIPipeline(fmri_config)
    fc_constructor = FCConstructor(
        n_rois=n_rois,
        regularization_lambda=regularization_lambda,
        fc_method=fc_method,
    )

    # ── Step 3: Process each subject ──
    fc_matrices = []
    labels = []
    fdt_features_list = []
    subject_ids = []
    skipped = 0

    # Look for subjects - handle fMRIPrep derivatives structure
    # e.g., data/LA5c/derivatives/fmriprep/sub-xxxxx/
    subject_dirs = sorted([d for d in site_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    if not subject_dirs:
        # Try derivatives/fmriprep/ subdirectory
        fmriprep_dir = site_dir / "derivatives" / "fmriprep"
        if fmriprep_dir.exists():
            subject_dirs = sorted([d for d in fmriprep_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")])
            if subject_dirs:
                logger.info(f"  Found {len(subject_dirs)} subjects in derivatives/fmriprep/")
        # Also try just derivatives/ 
        if not subject_dirs:
            deriv_dir = site_dir / "derivatives"
            if deriv_dir.exists():
                for sub in deriv_dir.glob("*/sub-*"):
                    if sub.is_dir() and sub.name.startswith("sub-"):
                        subject_dirs.append(sub)
                subject_dirs = sorted(subject_dirs)

    # Detect if data is fMRIPrep preprocessed
    is_fmriprep = (site_dir / "derivatives" / "fmriprep").exists()

    for subj_dir in subject_dirs:
        sub_id = subj_dir.name
        subject_meta = subjects.get(sub_id, {})

        label = get_diagnosis_label(subject_meta, site_config)
        if label is None:
            logger.debug(f"  Skipping {sub_id}: unknown diagnosis")
            skipped += 1
            continue

        # Find BOLD files
        bold_path, confounds_path = find_bold_files(
            subj_dir,
            task_name=site_config.get("task_name", "rest"),
        )

        if bold_path is None:
            logger.debug(f"  Skipping {sub_id}: no BOLD file found")
            skipped += 1
            continue

        # For fMRIPrep data: use ONLY the MNI-space preproc file
        # (NOT the brainmask or T1w-space version)
        if is_fmriprep:
            # Find the single best BOLD file: MNI-space preproc
            func_dir = subj_dir / "func"
            bold_candidates = sorted(func_dir.glob("*task-rest*preproc.nii*"))
            mni_preproc = [f for f in bold_candidates if "space-MNI" in f.name]
            if mni_preproc:
                bold_path = mni_preproc[0]  # Use ONLY the MNI preproc
            # Don't treat T1w/brainmask as separate runs!
            all_bold_files = [bold_path]
            logger.info(f"  Processing {sub_id} (fMRIPrep): {bold_path.name}")
        else:
            # For raw data with multiple rest runs (e.g., TCP has 6 runs)
            all_bold_files = sorted(
                (subj_dir / "func").glob(f"*task-{site_config.get('task_name', 'rest')}*_bold*.nii*")
            )
            # Filter to MNI space if available
            mni_bold = [f for f in all_bold_files if "space-MNI" in f.name or "space-MNI152" in f.name]
            if mni_bold:
                all_bold_files = mni_bold

            if len(all_bold_files) > 1:
                logger.info(f"  Processing {sub_id}: {len(all_bold_files)} rest runs found")
            else:
                logger.info(f"  Processing {sub_id}: {bold_path.name}")
                all_bold_files = [bold_path]

        try:
            import nibabel as nib

            # Collect time series from all runs
            all_run_ts = []
            confounds_all = None

            for run_idx, run_bold_path in enumerate(all_bold_files if len(all_bold_files) > 1 else [bold_path]):
                run_confounds_path = None
                # Find matching confounds file
                confound_stem = run_bold_path.name.replace("_bold.nii.gz", "").replace("_bold.nii", "")
                confound_pattern = f"{confound_stem}_desc-confounds_timeseries.tsv"
                run_conf_matches = list((subj_dir / "func").glob(confound_pattern))
                if run_conf_matches:
                    run_confounds_path = run_conf_matches[0]

                run_confounds = None
                if run_confounds_path is not None:
                    run_confounds = load_confounds(run_confounds_path, strategy=confound_strategy)

                fmri_img = nib.load(str(run_bold_path))

                tr = fmri_img.header.get_zooms()[-1]
                if tr_override is not None:
                    fmri_config.tr = tr_override
                    logger.info(f"    TR overridden to {tr_override}s")
                elif isinstance(tr, (int, float)) and 0.5 < tr < 5.0:
                    fmri_config.tr = tr
                    logger.info(f"    TR auto-detected: {tr}s")
                else:
                    logger.warning(f"    Invalid TR in header ({tr}), using default {fmri_config.tr}s")

                run_ts = pipeline.extract_time_series(fmri_img, confounds=run_confounds, is_fmriprep=is_fmriprep)

                # Check for empty time series (e.g., 3D mask loaded instead of 4D BOLD)
                if run_ts.shape[0] == 0:
                    logger.warning(f"    Empty time series for {run_bold_path.name} - skipping this run")
                    continue

                if run_confounds is not None:
                    fd = pipeline.compute_fd(run_confounds)
                    run_ts, valid_mask = pipeline.scrub_timepoints(run_ts, fd)

                if run_ts.shape[0] >= min_timepoints // 2:  # Allow shorter individual runs
                    all_run_ts.append(run_ts)

            if not all_run_ts:
                logger.warning(f"  Skipping {sub_id}: no usable timepoints from any run")
                skipped += 1
                continue

            # Concatenate runs for FC computation
            time_series = np.concatenate(all_run_ts, axis=0)

            if time_series.shape[0] < min_timepoints:
                logger.warning(f"  Skipping {sub_id}: only {time_series.shape[0]} total timepoints (min={min_timepoints})")
                skipped += 1
                continue
            
            logger.info(f"  {sub_id}: {time_series.shape[0]} timepoints after scrubbing, computing FC...")

            fc_matrix = fc_constructor.compute_static_fc(time_series)

            if compute_fdt:
                fdt = fc_constructor.compute_fdt_features(
                    time_series, n_top=n_fdt_features, tr=fmri_config.tr
                )
                fdt_features_list.append(fdt)

            fc_matrices.append(fc_matrix)
            labels.append(label)
            subject_ids.append(sub_id)

        except Exception as e:
            logger.warning(f"  Error processing {sub_id}: {e}")
            skipped += 1
            continue

    # ── Step 4: Save results ──
    if len(fc_matrices) == 0:
        logger.error(f"  No subjects successfully processed for {site_name}!")
        return None

    fc_matrices = np.stack(fc_matrices)
    labels = np.array(labels)
    fdt_arr = np.stack(fdt_features_list) if compute_fdt and fdt_features_list else None

    output_path = output_dir / f"{site_name}_processed.npz"
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
    }
    if fdt_arr is not None:
        save_dict["fdt_features"] = fdt_arr

    np.savez_compressed(output_path, **save_dict)

    logger.info(
        f"  Site {site_name}: {len(labels)} subjects processed "
        f"({int(labels.sum())} SZ / {int((1-labels).sum())} HC), "
        f"{skipped} skipped"
    )
    logger.info(f"  Saved to: {output_path}")

    return {
        "site_name": site_name,
        "site_id": site_config["site_id"],
        "role": site_config["role"],
        "n_samples": len(labels),
        "n_sz": int(labels.sum()),
        "n_hc": int((1 - labels).sum()),
        "n_skipped": skipped,
        "output_path": str(output_path),
    }


def preprocess_site_from_precomputed_fc(
    site_name: str,
    site_dir: Path,
    output_dir: Path,
    n_rois: int = 100,
) -> Optional[Dict]:
    """Preprocess a site that already has pre-computed FC matrices (e.g., SRPBS).

    Looks for .npy, .npz, .mat, or .csv files containing FC matrices.

    Args:
        site_name: Site name.
        site_dir: Path to site directory.
        output_dir: Path to save processed files.
        n_rois: Expected number of ROIs.

    Returns:
        Dictionary with processing results, or None on failure.
    """
    site_config = SITE_REGISTRY.get(site_name)
    if site_config is None:
        logger.error(f"Unknown site: {site_name}")
        return None

    logger.info(f"Processing site (precomputed FC): {site_name}")

    participants_tsv = site_dir / "participants.tsv"
    subjects = parse_participants_tsv(participants_tsv) if participants_tsv.exists() else {}

    fc_data = None
    fc_path = None

    search_patterns = [
        "**/fc_matrices.npz",
        "**/fc_matrices.npy",
        "**/connectivity*.npz",
        "**/connectivity*.npy",
        "**/correlation*.npz",
        "**/correlation*.npy",
        "**/*FC*.mat",
        "**/*functional_connectivity*.npy",
        "**/*.mat",  # SRPBS BICR: .mat files with FC per site
        "**/SUBINFO_*.tsv",  # SRPBS BICR: subject info files per site
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
    label_path = site_dir / "labels.npy"
    if label_path.exists():
        labels = np.load(label_path)
    else:
        if subjects:
            labels = []
            for sub_id in sorted(subjects.keys()):
                label = get_diagnosis_label(subjects[sub_id], site_config)
                labels.append(label if label is not None else -1)
            labels = np.array(labels)

            if len(labels) != fc_data.shape[0]:
                logger.warning(
                    f"  Subject count mismatch: participants.tsv={len(labels)}, "
                    f"FC matrices={fc_data.shape[0]}"
                )
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
        n_hc=int((1 - labels).sum()),
        role=site_config["role"],
    )

    logger.info(
        f"  Site {site_name}: {len(labels)} subjects "
        f"({int(labels.sum())} SZ / {int((1-labels).sum())} HC)"
    )

    return {
        "site_name": site_name,
        "site_id": site_config["site_id"],
        "role": site_config["role"],
        "n_samples": len(labels),
        "n_sz": int(labels.sum()),
        "n_hc": int((1 - labels).sum()),
        "output_path": str(output_path),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess real fMRI datasets for PQFL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess all sites in data directory
  python scripts/preprocess_real_data.py --data_dir ./data

  # Preprocess specific sites only
  python scripts/preprocess_real_data.py --data_dir ./data --sites COBRE MCIC

  # Use precomputed FC matrices (for SRPBS)
  python scripts/preprocess_real_data.py --data_dir ./data --use_precomputed_fc

  # Custom parcellation
  python scripts/preprocess_real_data.py --data_dir ./data --n_rois 200 --yeo_networks 17
        """,
    )

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing downloaded datasets")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: <data_dir>/processed/)")
    parser.add_argument("--sites", nargs="+", default=None,
                        help="Specific sites to process (e.g., COBRE MCIC)")
    parser.add_argument("--n_rois", type=int, default=100,
                        help="Number of Schaefer ROIs (default: 100)")
    parser.add_argument("--yeo_networks", type=int, default=7, choices=[7, 17],
                        help="Number of Yeo networks (default: 7)")
    parser.add_argument("--confound_strategy", type=str, default="simple",
                        choices=["simple", "scrubbing", "acompcor"],
                        help="Confound regression strategy")
    parser.add_argument("--fd_threshold", type=float, default=0.5,
                        help="Framewise displacement threshold in mm")
    parser.add_argument("--fc_method", type=str, default="pearson",
                        choices=["pearson", "partial", "covariance"],
                        help="FC computation method")
    parser.add_argument("--regularization_lambda", type=float, default=1e-3,
                        help="SPD regularization lambda")
    parser.add_argument("--compute_fdt", action="store_true",
                        help="Compute Frequency-Dependent Topology features")
    parser.add_argument("--use_precomputed_fc", action="store_true",
                        help="Look for pre-computed FC matrices instead of raw fMRI")
    parser.add_argument("--min_timepoints", type=int, default=50,
                        help="Minimum timepoints after scrubbing (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--tr", type=float, default=None,
                        help="Override repetition time in seconds (default: auto-detect from NIfTI header)")

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
        logger.info("Download datasets first using OpenNeuro CLI:")
        for name, cfg in SITE_REGISTRY.items():
            if cfg["openneuro_id"]:
                logger.info(f"  openneuro-py download --dataset={cfg['openneuro_id']}  # {name}")
        return

    logger.info(f"Sites to process: {sites_to_process}")
    logger.info(f"Output directory: {output_dir}")

    results = []
    for site_name in sites_to_process:
        site_dir = data_dir / site_name

        if not site_dir.exists():
            logger.warning(f"Directory not found: {site_dir} -- skipping")
            continue

        if args.use_precomputed_fc or site_name == "SRPBS":
            result = preprocess_site_from_precomputed_fc(
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
                confound_strategy=args.confound_strategy,
                fd_threshold=args.fd_threshold,
                fc_method=args.fc_method,
                regularization_lambda=args.regularization_lambda,
                compute_fdt=args.compute_fdt,
                min_timepoints=args.min_timepoints,
                tr_override=args.tr,
            )

        if result is not None:
            results.append(result)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_rois": args.n_rois,
            "yeo_networks": args.yeo_networks,
            "confound_strategy": args.confound_strategy,
            "fd_threshold": args.fd_threshold,
            "fc_method": args.fc_method,
            "regularization_lambda": args.regularization_lambda,
            "compute_fdt": args.compute_fdt,
        },
        "sites": results,
        "total_samples": sum(r["n_samples"] for r in results),
        "total_sz": sum(r["n_sz"] for r in results),
        "total_hc": sum(r["n_hc"] for r in results),
    }

    summary_path = output_dir / "processing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info(f"  Sites processed: {len(results)}")
    logger.info(f"  Total subjects: {summary['total_samples']}")
    logger.info(f"  Total SZ: {summary['total_sz']}")
    logger.info(f"  Total HC: {summary['total_hc']}")
    logger.info(f"  Summary saved to: {summary_path}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("=" * 60)

    print(f"""
========================================
NEXT STEPS:
========================================

1. Verify processed data:
   python -c "import numpy as np; d=np.load('{output_dir}/COBRE_processed.npz'); print(d['fc_matrices'].shape, d['labels'].shape)"

2. Run federated training with real data:
   python experiments/train_federated.py --data_dir {output_dir} --n_qubits 12 --n_rois {args.n_rois} --n_rounds 50

3. Quick demo with real data:
   python experiments/demo_e2e.py --data_dir {output_dir}
""")


if __name__ == "__main__":
    main()
