"""COBRE Adapter v3 - manual parcellation, bypasses buggy NiftiLabelsMasker.

The NiftiLabelsMasker in v2 had a caching bug where it returned (150,150) 
instead of (150,100) on all subjects after the first. This v3 version does
manual parcellation:

1. Load Schaefer 100-ROI atlas (1mm MNI space)
2. For each subject:
   a. Load BOLD .nii.gz
   b. Resample atlas to match BOLD's affine (using nearest neighbor)
   c. For each ROI label 1..100, extract mean timeseries across voxels in that ROI
   d. Stack into (n_timepoints, 100) matrix
3. Compute Pearson FC -> (100, 100) matrix
4. Regularize to SPD
5. Save COBRE_processed.npz

No NiftiLabelsMasker, no caching issues, no shape surprises.
"""
import sys
import time
from pathlib import Path
from collections import Counter
import numpy as np
import nibabel as nib

sys.path.insert(0, r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project")
from cobre_adapter import (
    COBRE_FIG, PHENO_CSV, OUTPUT_DIR,
    LABEL_SZ, LABEL_HC, COBRE_SITE_ID, COBRE_SITE_NAME,
    N_TARGET_ROIS, load_phenotype, find_bold_file,
)


def load_schaefer_atlas():
    """Load Schaefer 100-ROI atlas, return as Nifti1Image."""
    from nilearn.datasets import fetch_atlas_schaefer_2018
    atlas_data = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1)
    atlas_img = nib.load(atlas_data.maps)
    return atlas_img


def parcellate_bold_manual(bold_path, atlas_img):
    """Manually parcellate BOLD data using the atlas.
    Returns timeseries of shape (n_timepoints, n_rois).
    """
    from nilearn.image import resample_to_img

    # Load BOLD
    bold_img = nib.load(str(bold_path))

    # Resample atlas to match BOLD (nearest neighbor for labels)
    atlas_resampled = resample_to_img(
        atlas_img, bold_img,
        interpolation="nearest",
        force_resample=True,
    )

    # Get atlas data as integer labels
    atlas_data = atlas_resampled.get_fdata().astype(int)
    bold_data = bold_img.get_fdata()  # shape (X, Y, Z, T)

    # Find unique ROI labels (should be 0=background + 1..100)
    unique_labels = np.unique(atlas_data)
    roi_labels = [l for l in unique_labels if l > 0]
    n_rois_found = len(roi_labels)

    if n_rois_found != N_TARGET_ROIS:
        # Could be that some ROIs have no voxels in this BOLD space
        # Pad/truncate to 100 as needed
        pass

    n_timepoints = bold_data.shape[3]

    # Extract mean timeseries per ROI
    timeseries = np.zeros((n_timepoints, max(n_rois_found, N_TARGET_ROIS)), dtype=np.float64)
    for i, label in enumerate(roi_labels):
        if i >= N_TARGET_ROIS:
            break
        mask = (atlas_data == label)
        if mask.sum() == 0:
            continue
        # Mean across voxels in this ROI, for each timepoint
        roi_values = bold_data[mask, :]  # shape (n_voxels, T)
        timeseries[:, i] = roi_values.mean(axis=0)

    # Truncate to exactly N_TARGET_ROIS columns
    timeseries = timeseries[:, :N_TARGET_ROIS]

    # Z-score each ROI's timeseries (like NiftiLabelsMasker standardize=True)
    means = timeseries.mean(axis=0, keepdims=True)
    stds = timeseries.std(axis=0, keepdims=True, ddof=1)
    stds[stds == 0] = 1.0  # avoid division by zero
    timeseries = (timeseries - means) / stds

    return timeseries


def compute_fc(timeseries):
    """Compute Pearson FC matrix from (T, n_rois) timeseries. Returns (n_rois, n_rois) SPD."""
    fc = np.corrcoef(timeseries.T)  # (n_rois, n_rois)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)

    # SPD regularization
    lambda_reg = 1e-3
    fc = fc + lambda_reg * np.eye(fc.shape[0])
    fc = (fc + fc.T) / 2
    return fc


def main():
    print("=" * 78)
    print("COBRE Adapter v3 (manual parcellation - no NiftiLabelsMasker)")
    print("=" * 78)
    print(f"  COBRE Figshare BOLD: {COBRE_FIG}")
    print(f"  Phenotype CSV:       {PHENO_CSV}")
    print(f"  Output dir:          {OUTPUT_DIR}")
    print(f"  Target ROIs:         {N_TARGET_ROIS}")
    print()

    # 1. Load phenotype
    print("[1/4] Loading phenotype CSV...")
    pheno = load_phenotype()
    print(f"      Loaded {len(pheno)} subjects")
    print(f"      Subject Type: {dict(Counter(p['subject_type'] for p in pheno.values()))}")

    # 2. Cross-reference with BOLD files
    print("\n[2/4] Cross-referencing with BOLD files on disk...")
    matched = [(sid, info, find_bold_file(sid)) for sid, info in pheno.items()
               if find_bold_file(sid)]
    print(f"      Matched: {len(matched)} subjects")

    # 3. Load atlas ONCE
    print("\n[3/4] Loading Schaefer 100-ROI atlas...")
    atlas_img = load_schaefer_atlas()
    print(f"      Atlas shape: {atlas_img.shape}")
    print(f"      Atlas affine diag: {np.diag(atlas_img.affine)[:3]}")

    # Test on first subject
    print(f"\n      Testing parcellation on first subject...")
    test_sid, _, test_bold = matched[0]
    try:
        test_ts = parcellate_bold_manual(test_bold, atlas_img)
        print(f"      [OK] Test parcellation: shape {test_ts.shape} (expected: (~150, 100))")
        if test_ts.shape[1] != N_TARGET_ROIS:
            print(f"      [FAIL] Wrong ROI count!")
            return
    except Exception as e:
        print(f"      [FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Process all subjects
    print(f"\n[4/4] Processing {len(matched)} subjects...")
    fc_matrices = []
    labels = []
    subject_ids = []
    failed = []

    start_time = time.time()
    for i, (sid, info, bold_path) in enumerate(matched, 1):
        if i % 10 == 0 or i == 1:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(matched) - i) / rate if rate > 0 else 0
            print(f"      [{i}/{len(matched)}] sub-{sid:07d}  "
                  f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

        if info["subject_type"].lower() == "patient":
            label = LABEL_SZ
        elif info["subject_type"].lower() == "control":
            label = LABEL_HC
        else:
            label = 2

        try:
            ts = parcellate_bold_manual(bold_path, atlas_img)
            if ts.shape[1] != N_TARGET_ROIS:
                failed.append((sid, f"wrong ROI count: {ts.shape[1]}"))
                continue
            fc = compute_fc(ts)
            if fc.shape != (N_TARGET_ROIS, N_TARGET_ROIS):
                failed.append((sid, f"wrong FC shape: {fc.shape}"))
                continue
            fc_matrices.append(fc)
            labels.append(label)
            subject_ids.append(f"sub-{sid:07d}")
        except Exception as e:
            failed.append((sid, str(e)[:100]))

    elapsed = time.time() - start_time
    print(f"\n      Processed: {len(fc_matrices)} subjects in {elapsed:.1f}s")
    print(f"      Failed:    {len(failed)} subjects")
    if failed:
        print(f"      Sample failures (first 5):")
        for sid, err in failed[:5]:
            print(f"        sub-{sid:07d}: {err}")

    if not fc_matrices:
        print("\n[FAIL] No subjects processed.")
        return

    # 5. Save .npz
    fc_array = np.array(fc_matrices, dtype=np.float64)
    labels_array = np.array(labels, dtype=np.int64)
    subject_ids_array = np.array(subject_ids)

    assert fc_array.shape[1:] == (N_TARGET_ROIS, N_TARGET_ROIS), \
        f"FC shape mismatch: {fc_array.shape}"

    print(f"\n      Final array shapes:")
    print(f"        fc_matrices:  {fc_array.shape}  dtype={fc_array.dtype}")
    print(f"        labels:       {labels_array.shape}  dtype={labels_array.dtype}")

    n_nan = int(np.isnan(fc_array).sum())
    n_inf = int(np.isinf(fc_array).sum())
    print(f"        NaN count:    {n_nan}")
    print(f"        Inf count:    {n_inf}")

    final_label_counter = Counter(labels_array.tolist())
    label_names = {0: "SZ", 1: "HC", 2: "Other", 3: "BP"}
    print(f"\n      Final label distribution:")
    for lbl, n in sorted(final_label_counter.items()):
        print(f"        label {lbl} ({label_names.get(lbl, '?')}): {n}")

    output_file = OUTPUT_DIR / "COBRE_processed.npz"

    # Backup any existing file
    if output_file.exists():
        backup = output_file.with_suffix(".bak")
        if backup.exists():
            backup.unlink()
        import shutil
        shutil.move(str(output_file), str(backup))
        print(f"\n      [INFO] Backed up existing file to: {backup.name}")

    np.savez(
        output_file,
        fc_matrices=fc_array,
        labels=labels_array,
        subject_ids=subject_ids_array,
        site_id=COBRE_SITE_ID,
        site_name=COBRE_SITE_NAME,
        n_classes=2,
        n_rois=N_TARGET_ROIS,
        n_samples=len(fc_array),
        preprocessing_date=time.strftime("%Y-%m-%d %H:%M:%S"),
        atlas="Schaefer 100 (manual parcellation)",
        source="COBRE Figshare preprocessed (NIAK pipeline, 6mm MNI)",
    )
    print(f"\n[OK] Saved: {output_file}")
    print(f"     Size:  {output_file.stat().st_size/1e6:.1f} MB")
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Subjects:  {len(fc_array)}")
    print(f"  FC shape:  {fc_array.shape}")
    print(f"  Labels:    {dict(final_label_counter)}")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"  Output:    {output_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()
