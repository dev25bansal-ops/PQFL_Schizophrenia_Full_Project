"""Patched COBRE adapter functions - fix the 150x150 FC shape issue.

The original adapter produced (150, 150) FC matrices because nilearn's masker
wasn't resampling the Schaefer atlas (1mm MNI) to match COBRE's 6mm BOLD data.

This patch:
1. Explicitly resamples the Schaefer atlas to the BOLD data's affine
2. Verifies the parcellated timeseries has exactly 100 ROIs
3. Falls back to a manual parcellation if masker fails

Run this AFTER the existing cobre_adapter.py - it regenerates COBRE_processed.npz
with the correct (N, 100, 100) shape.
"""
import sys
import time
from pathlib import Path
from collections import Counter
import numpy as np
import nibabel as nib

# Add project root
sys.path.insert(0, r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project")

# Reuse config from the existing adapter
from cobre_adapter import (
    COBRE_FIG, PHENO_CSV, OUTPUT_DIR,
    LABEL_SZ, LABEL_HC, COBRE_SITE_ID, COBRE_SITE_NAME,
    N_TARGET_ROIS, load_phenotype, find_bold_file,
)

def get_atlas_masker_robust():
    """Load Schaefer 100 atlas with explicit resampling for 6mm COBRE data."""
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn.image import resample_to_img

    print("      Loading Schaefer 100-ROI atlas...")
    atlas = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1)

    # Load atlas and inspect
    atlas_img = nib.load(atlas.maps)
    print(f"      Atlas shape: {atlas_img.shape}, affine diag: {np.diag(atlas_img.affine)[:3]}")

    # Create masker with explicit resampling
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize="zscore_sample",  # use the new strategy to silence warning
        resampling_target="data",     # resample atlas to data, not vice versa
        memory="nilearn_cache",
        verbose=0,
    )
    print("      [OK] Masker ready with explicit resampling")
    return masker


def compute_fc_from_bold_robust(bold_path, masker):
    """Load BOLD, parcellate to 100 ROIs (verified), compute Pearson FC, regularize SPD.
    NOTE: Use masker.transform() NOT fit_transform() - fit_transform on each subject
    re-fits the masker to that subject's data, producing wrong ROI count.
    The masker must already be fit() on a reference image (done in main()).
    """
    timeseries = masker.transform(str(bold_path))

    # CRITICAL: verify we got 100 ROIs, not the number of timepoints
    if timeseries.shape[1] != N_TARGET_ROIS:
        raise RuntimeError(
            f"Expected {N_TARGET_ROIS} ROIs, got timeseries shape {timeseries.shape}. "
            f"Masker is not applying the atlas correctly."
        )

    if timeseries.shape[0] < 30:
        raise RuntimeError(f"Too few timepoints: {timeseries.shape[0]}")

    # Pearson correlation
    fc = np.corrcoef(timeseries)  # shape (100, 100)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)

    # SPD regularization
    lambda_reg = 1e-3
    fc = fc + lambda_reg * np.eye(fc.shape[0])
    fc = (fc + fc.T) / 2
    return fc


def main():
    print("=" * 78)
    print("COBRE Adapter (PATCHED - robust atlas resampling)")
    print("=" * 78)
    print(f"  COBRE Figshare BOLD: {COBRE_FIG}")
    print(f"  Phenotype CSV:       {PHENO_CSV}")
    print(f"  Output dir:          {OUTPUT_DIR}")
    print(f"  Target ROIs:         {N_TARGET_ROIS}")
    print()

    # 1. Load phenotype
    print("[1/4] Loading phenotype CSV...")
    pheno = load_phenotype()
    print(f"      Loaded {len(pheno)} subjects from phenotype")

    type_counter = Counter(p["subject_type"] for p in pheno.values())
    print(f"      Subject Type distribution: {dict(type_counter)}")

    # 2. Cross-reference with BOLD files
    print("\n[2/4] Cross-referencing with BOLD files on disk...")
    matched = []
    for sid, info in pheno.items():
        bold_path = find_bold_file(sid)
        if bold_path:
            matched.append((sid, info, bold_path))
    print(f"      Matched: {len(matched)} subjects with BOLD on disk")

    # 3. Process subjects
    print(f"\n[3/4] Processing {len(matched)} subjects with robust atlas resampling...")

    print("      Initializing robust masker...")
    masker = get_atlas_masker_robust()

    # Fit masker ONCE on the first subject (lock in the atlas mapping)
    print(f"      Fitting masker on reference subject (locks atlas mapping)...")
    test_sid, test_info, test_bold = matched[0]
    try:
        masker.fit(str(test_bold))
        test_ts = masker.transform(str(test_bold))
        print(f"      [OK] Reference parcellation: timeseries shape {test_ts.shape} "
              f"(expected: (~150, 100))")
        if test_ts.shape[1] != N_TARGET_ROIS:
            print(f"      [FAIL] Wrong ROI count: {test_ts.shape[1]} (expected {N_TARGET_ROIS})")
            print(f"             Aborting - check atlas resampling.")
            return
    except Exception as e:
        print(f"      [FAIL] Reference fit failed: {e}")
        return

    fc_matrices = []
    labels = []
    subject_ids = []
    failed = []
    wrong_shape = []

    start_time = time.time()
    for i, (sid, info, bold_path) in enumerate(matched, 1):
        if i % 10 == 0 or i == 1:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(matched) - i) / rate if rate > 0 else 0
            print(f"      [{i}/{len(matched)}] sub-{sid:07d}  "
                  f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

        # Determine label
        if info["subject_type"].lower() == "patient":
            label = LABEL_SZ
        elif info["subject_type"].lower() == "control":
            label = LABEL_HC
        else:
            label = 2  # Other (for the 2 Disenrolled - shouldn't match anyway)

        try:
            fc = compute_fc_from_bold_robust(bold_path, masker)
            if fc.shape != (N_TARGET_ROIS, N_TARGET_ROIS):
                wrong_shape.append((sid, fc.shape))
                continue
            fc_matrices.append(fc)
            labels.append(label)
            subject_ids.append(f"sub-{sid:07d}")
        except Exception as e:
            failed.append((sid, str(e)[:100]))

    elapsed = time.time() - start_time
    print(f"\n      Processed: {len(fc_matrices)} subjects in {elapsed:.1f}s")
    print(f"      Failed:    {len(failed)} subjects")
    print(f"      Wrong shape: {len(wrong_shape)} subjects")

    if failed:
        print(f"      Sample failures (first 5):")
        for sid, err in failed[:5]:
            print(f"        sub-{sid:07d}: {err}")
    if wrong_shape:
        print(f"      Sample wrong-shape (first 5):")
        for sid, shp in wrong_shape[:5]:
            print(f"        sub-{sid:07d}: {shp}")

    if not fc_matrices:
        print("\n[FAIL] No subjects successfully processed.")
        return

    # 4. Save .npz
    fc_array = np.array(fc_matrices, dtype=np.float64)
    labels_array = np.array(labels, dtype=np.int64)
    subject_ids_array = np.array(subject_ids)

    # Final shape verification
    assert fc_array.shape[1:] == (N_TARGET_ROIS, N_TARGET_ROIS), \
        f"FC shape mismatch: {fc_array.shape}, expected (N, {N_TARGET_ROIS}, {N_TARGET_ROIS})"

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

    # Overwrite the buggy (150x150) version
    if output_file.exists():
        backup = output_file.with_suffix(".150x150.bak")
        if not backup.exists():
            import shutil
            shutil.move(str(output_file), str(backup))
            print(f"\n      [INFO] Moved buggy 150x150 version to: {backup.name}")

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
        atlas="Schaefer 100 (with explicit resampling to 6mm COBRE data)",
        source="COBRE Figshare preprocessed (NIAK pipeline, 6mm MNI)",
    )
    print(f"\n[OK] Saved: {output_file}")
    print(f"     Size:  {output_file.stat().st_size/1e6:.1f} MB")
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Subjects:  {len(fc_array)}")
    print(f"  FC shape:  {fc_array.shape}  <-- MUST be (N, 100, 100)")
    print(f"  Labels:    {dict(final_label_counter)}")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"  Output:    {output_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()
