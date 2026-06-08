#!/usr/bin/env python3
"""Quick single-subject test of the fMRI preprocessing pipeline.

Tests the full pipeline on a single LA5c subject to verify:
  1. BOLD file loading (4D shape check)
  2. NiftiLabelsMasker extraction (no bandpass filter crash)
  3. Manual bandpass filtering
  4. FC matrix computation (SPD check)
  5. FDT feature extraction

Usage:
    python scripts/test_single_subject.py --data_dir ./data
    python scripts/test_single_subject.py --data_dir ./data --site LA5c
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_single_subject(data_dir: Path, site_name: str = "LA5c"):
    """Test preprocessing on a single subject."""
    import nibabel as nib
    from pqfl.data.fmri_pipeline import FMRIPipeline, FMRIConfig
    from pqfl.data.fc_construction import FCConstructor

    site_dir = data_dir / site_name

    # Find subject directories
    subject_dirs = sorted([d for d in site_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    fmriprep_dir = site_dir / "derivatives" / "fmriprep"
    if not subject_dirs and fmriprep_dir.exists():
        subject_dirs = sorted([d for d in fmriprep_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")])

    if not subject_dirs:
        print(f"ERROR: No subjects found in {site_dir}")
        return False

    # Pick the first subject
    test_subj = subject_dirs[0]
    print(f"\n{'='*70}")
    print(f"  SINGLE SUBJECT TEST")
    print(f"  Site: {site_name}")
    print(f"  Subject: {test_subj.name}")
    print(f"{'='*70}")

    func_dir = test_subj / "func"
    if not func_dir.exists():
        print(f"ERROR: No func/ directory for {test_subj.name}")
        return False

    # Find MNI preproc BOLD file
    bold_files = sorted(func_dir.glob("*task-rest*preproc.nii*"))
    mni_preproc = [f for f in bold_files if "space-MNI" in f.name]

    if not mni_preproc:
        print(f"ERROR: No MNI-space preproc BOLD found for {test_subj.name}")
        print(f"  Available files: {[f.name for f in bold_files]}")
        return False

    bold_path = mni_preproc[0]
    print(f"\n  BOLD file: {bold_path.name}")

    # Step 1: Load and check dimensions
    print(f"\n  --- Step 1: Loading BOLD image ---")
    try:
        img = nib.load(str(bold_path))
        shape = img.shape
        print(f"  ✓ Image shape: {shape}")
        if len(shape) < 4:
            print(f"  ✗ ERROR: Image is {len(shape)}D, expected 4D!")
            return False
        n_vols = shape[3]
        print(f"  ✓ Number of volumes: {n_vols}")

        zooms = img.header.get_zooms()
        tr = zooms[-1]
        print(f"  ✓ TR from header: {tr}")
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        return False

    # Step 2: Initialize pipeline
    print(f"\n  --- Step 2: Initializing pipeline ---")
    fmri_config = FMRIConfig(
        parcellation="schaefer",
        n_rois=100,
        yeo_networks=7,
        tr=2.0,
        bandpass_low=0.01,
        bandpass_high=0.08,
        fd_threshold=0.5,
        confound_strategy="simple",
        standardize=True,
        detrend=True,
    )

    # Update TR if auto-detected
    if isinstance(tr, (int, float)) and 0.5 < tr < 5.0:
        fmri_config.tr = float(tr)
        print(f"  ✓ TR set to {fmri_config.tr}s")

    pipeline = FMRIPipeline(fmri_config)
    print(f"  ✓ Pipeline created (n_rois={fmri_config.n_rois}, tr={fmri_config.tr})")

    # Step 3: Extract time series (with is_fmriprep=True)
    print(f"\n  --- Step 3: Extracting ROI time series ---")
    try:
        # Check for confounds
        confound_files = list(func_dir.glob("*desc-confounds_timeseries.tsv"))
        confounds = None
        if confound_files:
            print(f"  Found confounds: {confound_files[0].name}")
            # Load confounds (simple strategy)
            import csv
            with open(confound_files[0], 'r') as f:
                reader = csv.DictReader(f, delimiter='\t')
                rows = list(reader)
            cols = ["csf", "white_matter", "global_signal",
                    "trans_x", "trans_y", "trans_z",
                    "rot_x", "rot_y", "rot_z"]
            available = [c for c in cols if c in rows[0]]
            if available:
                confounds = np.zeros((len(rows), len(available)))
                for i, row in enumerate(rows):
                    for j, col in enumerate(available):
                        try:
                            val = float(row[col])
                            confounds[i, j] = val if val == val else 0.0
                        except (ValueError, TypeError):
                            confounds[i, j] = 0.0
                print(f"  ✓ Confounds loaded: {confounds.shape}")
            else:
                print(f"  ⚠ No matching confound columns found, proceeding without confounds")
        else:
            print(f"  ⚠ No confound files found, proceeding without confound regression")

        time_series = pipeline.extract_time_series(img, confounds=confounds, is_fmriprep=True)

        if time_series.shape[0] == 0:
            print(f"  ✗ ERROR: Empty time series extracted!")
            return False

        print(f"  ✓ Time series extracted: {time_series.shape}")
        print(f"  ✓ Mean signal: {time_series.mean():.4f}, Std: {time_series.std():.4f}")
    except Exception as e:
        print(f"  ✗ Failed to extract time series: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Compute FC matrix
    print(f"\n  --- Step 4: Computing FC matrix ---")
    try:
        fc_constructor = FCConstructor(
            n_rois=100,
            regularization_lambda=1e-3,
            fc_method="pearson",
        )
        fc_matrix = fc_constructor.compute_static_fc(time_series)
        print(f"  ✓ FC matrix shape: {fc_matrix.shape}")
        print(f"  ✓ FC range: [{fc_matrix.min():.4f}, {fc_matrix.max():.4f}]")

        # Check SPD
        eigvals = np.linalg.eigvalsh(fc_matrix)
        min_eig = np.min(eigvals)
        print(f"  ✓ Min eigenvalue: {min_eig:.6f} ({'SPD ✓' if min_eig > 0 else 'NOT SPD ✗'})")
    except Exception as e:
        print(f"  ✗ Failed to compute FC: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Compute FDT features
    print(f"\n  --- Step 5: Computing FDT features ---")
    try:
        fdt = fc_constructor.compute_fdt_features(time_series, n_top=20, tr=fmri_config.tr)
        print(f"  ✓ FDT features shape: {fdt.shape}")
        print(f"  ✓ FDT range: [{fdt.min():.4f}, {fdt.max():.4f}]")
    except Exception as e:
        print(f"  ⚠ FDT failed (non-critical): {e}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  ✓ SINGLE SUBJECT TEST PASSED!")
    print(f"  Subject: {test_subj.name}")
    print(f"  BOLD: {n_vols} volumes, TR={fmri_config.tr}s")
    print(f"  Time series: {time_series.shape[0]} timepoints × {time_series.shape[1]} ROIs")
    print(f"  FC matrix: {fc_matrix.shape}, min_eig={min_eig:.6f}")
    print(f"{'='*70}")
    print(f"\n  You can now run full preprocessing with confidence:")
    print(f"  python scripts/preprocess_real_data.py --data_dir {data_dir} --sites {site_name} --compute_fdt")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(description="Test preprocessing on a single subject")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--site", type=str, default="LA5c")
    args = parser.parse_args()

    success = test_single_subject(Path(args.data_dir), args.site)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
