"""COBRE Figshare adapter - load pre-fMRIPrep'd BOLD + raw phenotype CSV.

Strategy:
1. Load phenotype from _dropped/COBRE/COBRE_phenotypic_data.csv
   (Columns: ID, Current Age, Gender, Handedness, Subject Type, Diagnosis)
   - Subject Type == 'Patient'  -> LABEL_SZ (Diagnosis == 'Schizophrenia')
   - Subject Type == 'Control'  -> LABEL_HC
2. For each subject in phenotype:
   - Check if fmri_{ID}.nii.gz exists in COBRE Preprocessed (Figshare)/extracted/
   - Apply Schaefer 100-ROI parcellation using nilearn
   - Compute Pearson correlation FC matrix
   - Regularize to SPD
3. Save COBRE_processed.npz

Expected: ~146 subjects (72 SZ + 74 HC)
"""
import argparse
import csv
import sys
import time
from pathlib import Path
from collections import Counter
import numpy as np

# ─── Paths ─────────────────────────────────────────────────────────────────
COBRE_FIG = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data\COBRE Preprocessed (Figshare)\extracted")
PHENO_CSV = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data\_dropped\COBRE\COBRE_phenotypic_data.csv")
OUTPUT_DIR = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data\processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Constants ─────────────────────────────────────────────────────────────
LABEL_SZ = 0
LABEL_HC = 1
LABEL_OTHER = 2
LABEL_BP = 3

COBRE_SITE_ID = 4
COBRE_SITE_NAME = "COBRE"
N_TARGET_ROIS = 100


def load_phenotype():
    """Load COBRE phenotype CSV from _dropped/COBRE/.
    Returns dict: {subject_id_int: {'subject_type': str, 'diagnosis': str, 'age': int, 'gender': str}}
    """
    if not PHENO_CSV.exists():
        raise FileNotFoundError(
            f"COBRE phenotype CSV not found at: {PHENO_CSV}\n"
            f"Restore it with: Move-Item data\\_dropped\\COBRE\\COBRE_phenotypic_data.csv data\\"
        )

    subjects = {}
    with open(PHENO_CSV, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        # Try to detect the ID column (first column might be unnamed)
        cols = list(reader.fieldnames)
        print(f"      Phenotype columns: {cols}")

        # Find ID column (might be '', 'ID', 'Subject', etc.)
        id_col = next((c for c in cols if c.strip().lower() in ("id", "subject", "subject_id", "subjectid")), cols[0])
        type_col = next((c for c in cols if "type" in c.lower() or "subject type" in c.lower()), None)
        diag_col = next((c for c in cols if "diag" in c.lower()), None)
        age_col = next((c for c in cols if c.lower() == "current age" or c.lower() == "age"), None)
        gender_col = next((c for c in cols if c.lower() == "gender" or c.lower() == "sex"), None)

        print(f"      ID column: '{id_col}', Type column: '{type_col}', "
              f"Diagnosis column: '{diag_col}'")

        for row in reader:
            sid_str = (row.get(id_col) or "").strip()
            if not sid_str:
                continue
            try:
                # ID might be '0040000' or '40000' - normalize to int
                sid = int(sid_str)
            except ValueError:
                continue
            subjects[sid] = {
                "subject_type": (row.get(type_col) or "").strip() if type_col else "",
                "diagnosis": (row.get(diag_col) or "").strip() if diag_col else "",
                "age": (row.get(age_col) or "").strip() if age_col else "",
                "gender": (row.get(gender_col) or "").strip() if gender_col else "",
            }
    return subjects


def find_bold_file(subject_id_int):
    """Find the BOLD .nii.gz file for a given subject ID.
    Files are named fmri_{ID:07d}.nii.gz (e.g., fmri_0040000.nii.gz for ID=40000).
    """
    # Try the standard format: fmri_0040000.nii.gz for ID 40000
    candidates = [
        COBRE_FIG / f"fmri_{subject_id_int:07d}.nii.gz",   # 7-digit with leading 0
        COBRE_FIG / f"fmri_{subject_id_int:06d}.nii.gz",   # 6-digit
        COBRE_FIG / f"fmri_{subject_id_int:05d}.nii.gz",   # 5-digit
        COBRE_FIG / f"fmri_{subject_id_int}.nii.gz",       # no padding
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def get_atlas_masker():
    """Lazily load Schaefer 100-ROI atlas masker (cached by nilearn)."""
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn.datasets import fetch_atlas_schaefer_2018
    atlas = fetch_atlas_schaefer_2018(n_rois=N_TARGET_ROIS, yeo_networks=7)
    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize=True,
        memory="nilearn_cache",
        verbose=0,
    )
    return masker


def compute_fc_from_bold(bold_path, masker):
    """Load BOLD, parcellate to 100 ROIs, compute Pearson FC, regularize SPD."""
    timeseries = masker.fit_transform(str(bold_path))
    if timeseries.shape[0] < 30:
        raise RuntimeError(f"Too few timepoints: {timeseries.shape[0]}")

    fc = np.corrcoef(timeseries)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)

    # SPD regularization
    lambda_reg = 1e-3
    fc = fc + lambda_reg * np.eye(fc.shape[0])
    fc = (fc + fc.T) / 2
    return fc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, default=2, choices=[2, 3, 4],
                       help="COBRE is binary (SZ vs HC) so 2-class is default")
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview subject matching without processing BOLD")
    parser.add_argument("--max_subjects", type=int, default=None,
                       help="Limit subjects (for testing)")
    args = parser.parse_args()

    print("=" * 78)
    print("COBRE Figshare Adapter")
    print("=" * 78)
    print(f"  COBRE Figshare BOLD: {COBRE_FIG}")
    print(f"  Phenotype CSV:       {PHENO_CSV}")
    print(f"  Output dir:          {OUTPUT_DIR}")
    print(f"  N classes:           {args.n_classes}")
    print(f"  Dry run:             {args.dry_run}")
    print()

    # 1. Load phenotype
    print("[1/4] Loading phenotype CSV...")
    pheno = load_phenotype()
    print(f"      Loaded {len(pheno)} subjects from phenotype")

    # Diagnosis distribution
    type_counter = Counter(p["subject_type"] for p in pheno.values())
    diag_counter = Counter(p["diagnosis"] for p in pheno.values())
    print(f"\n      Subject Type distribution: {dict(type_counter)}")
    print(f"      Diagnosis distribution (top 5): {dict(diag_counter.most_common(5))}")

    # 2. Cross-reference with BOLD files on disk
    print("\n[2/4] Cross-referencing with BOLD files on disk...")
    matched = []
    no_bold = []
    for sid, info in pheno.items():
        bold_path = find_bold_file(sid)
        if bold_path:
            matched.append((sid, info, bold_path))
        else:
            no_bold.append(sid)

    print(f"      Subjects in phenotype:     {len(pheno)}")
    print(f"      Subjects with BOLD on disk: {len(matched)}")
    print(f"      Subjects without BOLD:      {len(no_bold)}")
    if no_bold:
        print(f"      Sample missing: {no_bold[:5]}")

    # Compute label distribution for matched subjects
    label_counter = Counter()
    for sid, info, _ in matched:
        if info["subject_type"].lower() == "patient":
            label_counter["SZ"] += 1
        elif info["subject_type"].lower() == "control":
            label_counter["HC"] += 1
        else:
            label_counter["Other"] += 1

    print(f"\n      Label distribution for matched subjects: {dict(label_counter)}")

    if args.max_subjects:
        matched = matched[:args.max_subjects]
        print(f"      Limited to first {len(matched)} subjects (--max_subjects)")

    if args.dry_run:
        print("\n[DRY RUN] Skipping BOLD processing. Use without --dry-run to process.")
        return

    # 3. Process subjects
    print(f"\n[3/4] Processing {len(matched)} subjects...")
    print(f"      (loading BOLD .nii.gz, Schaefer 100 parcellation, FC computation)")

    # Initialize atlas masker
    print("      Loading Schaefer 100-ROI atlas (cached on first run)...")
    masker = get_atlas_masker()
    print("      [OK] Atlas ready")

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

        # Determine label
        if info["subject_type"].lower() == "patient":
            label = LABEL_SZ
        elif info["subject_type"].lower() == "control":
            label = LABEL_HC
        else:
            label = LABEL_OTHER

        try:
            fc = compute_fc_from_bold(bold_path, masker)
            fc_matrices.append(fc)
            labels.append(label)
            subject_ids.append(f"sub-{sid:07d}")
        except Exception as e:
            failed.append((sid, str(e)))

    elapsed = time.time() - start_time
    print(f"\n      Processed: {len(fc_matrices)} subjects in {elapsed:.1f}s "
          f"({elapsed/max(1, len(fc_matrices)):.2f}s/subject)")
    print(f"      Failed:    {len(failed)} subjects")

    if failed:
        print(f"      Sample failures (first 5):")
        for sid, err in failed[:5]:
            print(f"        sub-{sid:07d}: {err}")

    if not fc_matrices:
        print("\n[FAIL] No subjects successfully processed. Aborting.")
        return

    # 4. Save .npz - handle inhomogeneous shapes (some subjects may have wrong ROI count)
    print(f"\n      Checking FC matrix shapes...")
    shapes = [fc.shape for fc in fc_matrices]
    from collections import Counter as _C
    shape_counter = _C(shapes)
    print(f"      Shape distribution: {dict(shape_counter)}")

    if len(shape_counter) > 1:
        # Find the most common shape and keep only those
        most_common_shape, _ = shape_counter.most_common(1)[0]
        print(f"      [WARN] Inhomogeneous shapes detected. Most common: {most_common_shape}")
        print(f"      [WARN] Filtering to keep only subjects with shape {most_common_shape}")

        filtered_fc = []
        filtered_labels = []
        filtered_ids = []
        dropped = []
        for fc, lbl, sid in zip(fc_matrices, labels, subject_ids):
            if fc.shape == most_common_shape:
                filtered_fc.append(fc)
                filtered_labels.append(lbl)
                filtered_ids.append(sid)
            else:
                dropped.append((sid, fc.shape))
        print(f"      Dropped {len(dropped)} subjects with wrong shape:")
        for sid, shp in dropped[:10]:
            print(f"        {sid}: shape {shp}")
        fc_matrices = filtered_fc
        labels = filtered_labels
        subject_ids = filtered_ids
        print(f"      Remaining: {len(fc_matrices)} subjects")

    fc_array = np.array(fc_matrices, dtype=np.float64)
    labels_array = np.array(labels, dtype=np.int64)
    subject_ids_array = np.array(subject_ids)

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
    np.savez(
        output_file,
        fc_matrices=fc_array,
        labels=labels_array,
        subject_ids=subject_ids_array,
        site_id=COBRE_SITE_ID,
        site_name=COBRE_SITE_NAME,
        n_classes=args.n_classes,
        n_rois=N_TARGET_ROIS,
        n_samples=len(fc_array),
        preprocessing_date=time.strftime("%Y-%m-%d %H:%M:%S"),
        atlas="Schaefer 100",
        source="COBRE Figshare preprocessed (NIAK pipeline, 6mm MNI)",
    )
    print(f"\n[OK] Saved: {output_file}")
    print(f"     Size:  {output_file.stat().st_size/1e6:.1f} MB")
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Subjects:  {len(fc_array)}")
    print(f"  Labels:    {dict(final_label_counter)}")
    print(f"  FC shape:  {fc_array.shape}")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"  Output:    {output_file}")
    print()
    print("Next: re-run verify_all_data.py to confirm COBRE .npz is valid.")
    print("=" * 78)


if __name__ == "__main__":
    main()
