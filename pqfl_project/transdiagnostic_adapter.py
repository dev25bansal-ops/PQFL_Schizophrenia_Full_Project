"""Transdiagnostic adapter - load .h5 parcellated timeseries and build FC matrices.

Strategy:
1. Read demos.tsv (skip row 0 metadata, use row 1 as header) for Primary_Dx
2. Normalize Primary_Dx -> PQFL label (SZ/HC/Other for 3-class, +BP for 4-class)
3. For each subject with .h5 files:
   a. Load all available rest runs (restAP/PA_run-01/02)
   b. Concatenate along time dimension (better FC estimate)
   c. Drop subcortical ROIs (last 34 rows) -> 400 cortical Schaefer
   d. Average groups of 4 consecutive ROIs -> 100 ROIs (Schaefer 100 equivalent)
   e. Compute Pearson correlation FC matrix (100x100)
   f. Regularize to SPD
4. Save Transdiagnostic_processed.npz

Output: ~241 subjects (13 SZ + 92 HC + 25 BP + 111 Other with .h5 on disk)

Usage:
    python transdiagnostic_adapter.py --n_classes 3
    python transdiagnostic_adapter.py --n_classes 4
    python transdiagnostic_adapter.py --dry-run   # preview only, no save
"""
import argparse
import csv
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import h5py

# ─── Paths ─────────────────────────────────────────────────────────────────
TD_ROOT = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data\Transdiagnostic")
DEMOS_TSV = TD_ROOT / "phenotype" / "demos.tsv"
H5_DIR = TD_ROOT / "fMRI_timeseries_clean_denoised_GSR_parcellated"
OUTPUT_DIR = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data\processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Constants ─────────────────────────────────────────────────────────────
LABEL_SZ = 0
LABEL_HC = 1
LABEL_OTHER = 2
LABEL_BP = 3

TRANSDIAGNOSTIC_SITE_ID = 19
TRANSDIAGNOSTIC_SITE_NAME = "Transdiagnostic"

# Schaefer 400 -> 100 downsampling
N_CORTICAL_ROIS = 400  # first 400 rows of .h5 are Schaefer 400 cortical
N_SUBCORTICAL = 34      # last 34 rows are Tian subcortical (we drop these)
N_TARGET_ROIS = 100
GROUP_SIZE = N_CORTICAL_ROIS // N_TARGET_ROIS  # = 4

# Rest runs to use (in priority order)
REST_RUN_PATTERNS = [
    "task-restAP_run-01_bold_Atlas_hp2000_clean_GSR_parcellated.h5",
    "task-restAP_run-02_bold_Atlas_hp2000_clean_GSR_parcellated.h5",
    "task-restPA_run-01_bold_Atlas_hp2000_clean_GSR_parcellated.h5",
    "task-restPA_run-02_bold_Atlas_hp2000_clean_GSR_parcellated.h5",
]


def normalize_primary_dx(dx: str) -> str:
    """Normalize Primary_Dx string to a canonical category.
    Returns one of: 'SZ', 'BP', 'MDD', 'ANXIETY', 'PTSD', 'OTHER_PSYCH',
                    'SUBSTANCE', 'OTHER', 'HC'
    """
    if not dx or dx.strip() in ("999", ""):
        return "HC"

    dx_lower = dx.lower().strip()

    # Schizophrenia spectrum
    if dx_lower in ("sz", "sza") or "schiz" in dx_lower or dx_lower.startswith("sz "):
        return "SZ"
    if "sz or sza" in dx_lower or "ruleout sza" in dx_lower or "ruleout sz" in dx_lower:
        return "SZ"

    # Bipolar spectrum
    if dx_lower in ("bp1", "bp2", "bpi", "bpii", "bp"):
        return "BP"
    if "bipolar" in dx_lower or "cyclothym" in dx_lower:
        return "BP"

    # MDD / Depression
    if "mdd" in dx_lower or "depression" in dx_lower or "depressive" in dx_lower:
        return "MDD"
    if "dysthym" in dx_lower:
        return "MDD"

    # Anxiety
    if any(kw in dx_lower for kw in ["gad", "anxiety", "anxious", "panic", "phobia", "agoraphobia"]):
        return "ANXIETY"

    # PTSD
    if "ptsd" in dx_lower or "post-traumatic" in dx_lower:
        return "PTSD"

    # ADHD
    if "adhd" in dx_lower or "attention-deficit" in dx_lower:
        return "OTHER_PSYCH"

    # OCD
    if "ocd" in dx_lower or "obsessive" in dx_lower:
        return "OTHER_PSYCH"

    # Eating disorders
    if any(kw in dx_lower for kw in ["eating disorder", "binge", "anorex", "bulim"]):
        return "OTHER_PSYCH"

    # PMDD
    if "pmdd" in dx_lower:
        return "OTHER_PSYCH"

    # Substance use disorders
    if any(kw in dx_lower for kw in ["aud", "cud", "sud", "polysub", "substance"]):
        return "SUBSTANCE"

    return "OTHER"


def dx_category_to_label(category: str, n_classes: int) -> int:
    """Map normalized category to PQFL label."""
    if category == "HC":
        return LABEL_HC
    if category == "SZ":
        return LABEL_SZ
    if category == "BP":
        return LABEL_BP if n_classes == 4 else LABEL_OTHER
    # All other categories (MDD, ANXIETY, PTSD, OTHER_PSYCH, SUBSTANCE, OTHER)
    return LABEL_OTHER


def load_demos() -> dict:
    """Load demos.tsv -> {subjectkey: Primary_Dx}. Returns dict of 245 subjects."""
    with open(DEMOS_TSV, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        all_rows = list(reader)
    header = all_rows[1]  # row 0 is metadata, row 1 is real header
    data_rows = all_rows[2:]

    sk_idx = header.index("subjectkey")
    dx_idx = header.index("Primary_Dx")

    mapping = {}
    for row in data_rows:
        if len(row) <= max(sk_idx, dx_idx):
            continue
        sk = row[sk_idx].strip()
        dx = row[dx_idx].strip()
        if sk.startswith("NDAR_INV"):
            mapping[sk] = dx
    return mapping


def load_h5_timeseries(h5_path: Path) -> np.ndarray:
    """Load a single .h5 parcellated timeseries. Returns shape (434, n_timepoints)."""
    with h5py.File(h5_path, "r") as f:
        # Top-level key is 'dataset'
        ts = f["dataset"][:]
    return ts  # shape (434, n_timepoints)


def downsample_to_100(timeseries_434: np.ndarray) -> np.ndarray:
    """Downsample Schaefer 400+34 to Schaefer 100 by:
    1. Drop subcortical (last 34 rows) -> 400 cortical
    2. Average groups of 4 consecutive ROIs -> 100
    Returns shape (100, n_timepoints).
    """
    # Drop subcortical
    cortical = timeseries_434[:N_CORTICAL_ROIS, :]  # (400, T)
    # Reshape (400, T) -> (100, 4, T) -> mean over axis 1 -> (100, T)
    n_timepoints = cortical.shape[1]
    reshaped = cortical.reshape(N_TARGET_ROIS, GROUP_SIZE, n_timepoints)
    return reshaped.mean(axis=1)


def compute_fc(timeseries: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation FC matrix from timeseries (ROIs x timepoints).
    Returns (n_rois, n_rois) SPD matrix.
    """
    # Pearson correlation = normalized covariance
    fc = np.corrcoef(timeseries)  # (n_rois, n_rois)
    # Handle NaN (constant ROI) - replace with 0
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    # Regularize to SPD: C + lambda*I
    lambda_reg = 1e-3
    fc = fc + lambda_reg * np.eye(fc.shape[0])
    # Ensure symmetric
    fc = (fc + fc.T) / 2
    return fc


def process_subject(subject_key: str, dx_label: int, n_classes: int) -> dict:
    """Load all rest runs for a subject, compute FC, return dataset dict.
    Returns None if no rest .h5 files found.
    """
    subject_dir = H5_DIR / subject_key
    if not subject_dir.exists():
        return None

    # Load all available rest runs
    all_timeseries = []
    used_runs = []
    for run_pattern in REST_RUN_PATTERNS:
        h5_path = subject_dir / run_pattern
        if h5_path.exists():
            try:
                ts = load_h5_timeseries(h5_path)
                # Downsample 434 -> 100
                ts_100 = downsample_to_100(ts)
                all_timeseries.append(ts_100)
                used_runs.append(run_pattern)
            except Exception as e:
                print(f"    [WARN] Failed to load {h5_path.name}: {e}")

    if not all_timeseries:
        return None

    # Concatenate all runs along time dimension
    if len(all_timeseries) == 1:
        combined_ts = all_timeseries[0]
    else:
        combined_ts = np.concatenate(all_timeseries, axis=1)  # (100, total_T)

    # Compute FC
    fc = compute_fc(combined_ts)

    return {
        "fc": fc,
        "label": dx_label,
        "subject_id": subject_key,
        "n_runs": len(all_timeseries),
        "n_timepoints_total": combined_ts.shape[1],
        "used_runs": used_runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview label distribution without processing .h5 files")
    parser.add_argument("--max_subjects", type=int, default=None,
                       help="Limit subjects (for testing)")
    args = parser.parse_args()

    print("=" * 78)
    print("Transdiagnostic Adapter")
    print("=" * 78)
    print(f"  Demos file:  {DEMOS_TSV}")
    print(f"  H5 dir:      {H5_DIR}")
    print(f"  Output dir:  {OUTPUT_DIR}")
    print(f"  N classes:   {args.n_classes}")
    print(f"  Dry run:     {args.dry_run}")
    print()

    # 1. Load demos.tsv
    print("[1/4] Loading demos.tsv...")
    dx_map = load_demos()
    print(f"      Loaded {len(dx_map)} subject diagnoses")

    # 2. Normalize diagnoses and map to labels
    print("\n[2/4] Normalizing diagnoses -> PQFL labels...")
    sub_to_label = {}
    sub_to_category = {}
    for sk, dx in dx_map.items():
        category = normalize_primary_dx(dx)
        label = dx_category_to_label(category, args.n_classes)
        sub_to_label[sk] = label
        sub_to_category[sk] = category

    # Print distribution
    cat_counter = Counter(sub_to_category.values())
    print(f"      Normalized category distribution:")
    for cat, n in sorted(cat_counter.items(), key=lambda x: -x[1]):
        label = dx_category_to_label(cat, args.n_classes)
        label_name = {0: "SZ", 1: "HC", 2: "Other", 3: "BP"}.get(label, "?")
        print(f"        {cat:<15}  ->  label {label} ({label_name})  :  {n}")

    label_counter = Counter(sub_to_label.values())
    print(f"\n      Final label distribution:")
    for lbl in sorted(label_counter.keys()):
        label_name = {0: "SZ", 1: "HC", 2: "Other", 3: "BP"}.get(lbl, "?")
        print(f"        label {lbl} ({label_name}):  {label_counter[lbl]}")

    # 3. Cross-reference with .h5 files on disk
    print("\n[3/4] Cross-referencing with .h5 files on disk...")
    h5_subjects = set()
    if H5_DIR.exists():
        for d in H5_DIR.iterdir():
            if d.is_dir() and d.name.startswith("NDAR_INV"):
                if any(d.glob("task-rest*_bold_*.h5")):
                    h5_subjects.add(d.name)
    print(f"      Subjects with .h5 rest files on disk: {len(h5_subjects)}")

    # Subjects we can process
    usable_subjects = sorted([s for s in h5_subjects if s in sub_to_label])
    print(f"      Subjects with both Dx + .h5:         {len(usable_subjects)}")

    # Final label distribution for usable subjects
    usable_label_counter = Counter(sub_to_label[s] for s in usable_subjects)
    print(f"\n      Final usable label distribution:")
    for lbl in sorted(usable_label_counter.keys()):
        label_name = {0: "SZ", 1: "HC", 2: "Other", 3: "BP"}.get(lbl, "?")
        print(f"        label {lbl} ({label_name}):  {usable_label_counter[lbl]}")

    if args.max_subjects:
        usable_subjects = usable_subjects[:args.max_subjects]
        print(f"      Limited to first {len(usable_subjects)} subjects (--max_subjects)")

    if args.dry_run:
        print("\n[DRY RUN] Skipping .h5 processing. Use without --dry-run to actually process.")
        return

    # 4. Process subjects
    print(f"\n[4/4] Processing {len(usable_subjects)} subjects...")
    print(f"      (loading .h5, downsampling 434->100, computing FC)")

    fc_matrices = []
    labels = []
    subject_ids = []
    n_runs_used = []
    n_timepoints_used = []
    failed = []

    start_time = time.time()
    for i, sk in enumerate(usable_subjects, 1):
        if i % 20 == 0 or i == 1:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(usable_subjects) - i) / rate if rate > 0 else 0
            print(f"      [{i}/{len(usable_subjects)}] {sk}  "
                  f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

        label = sub_to_label[sk]
        try:
            result = process_subject(sk, label, args.n_classes)
            if result is None:
                failed.append((sk, "no rest .h5 files"))
                continue
            fc_matrices.append(result["fc"])
            labels.append(result["label"])
            subject_ids.append(result["subject_id"])
            n_runs_used.append(result["n_runs"])
            n_timepoints_used.append(result["n_timepoints_total"])
        except Exception as e:
            failed.append((sk, str(e)))

    elapsed = time.time() - start_time
    print(f"\n      Processed: {len(fc_matrices)} subjects in {elapsed:.1f}s "
          f"({len(fc_strings := fc_matrices) and elapsed/max(1, len(fc_matrices)):.2f}s/subject)")
    print(f"      Failed:    {len(failed)} subjects")

    if failed:
        print(f"      Sample failures (first 5):")
        for sk, err in failed[:5]:
            print(f"        {sk}: {err}")

    if not fc_matrices:
        print("\n[FAIL] No subjects successfully processed. Aborting.")
        return

    # 5. Save .npz
    fc_array = np.array(fc_matrices, dtype=np.float64)
    labels_array = np.array(labels, dtype=np.int64)
    subject_ids_array = np.array(subject_ids)

    print(f"\n      Final array shapes:")
    print(f"        fc_matrices:  {fc_array.shape}  dtype={fc_array.dtype}")
    print(f"        labels:       {labels_array.shape}  dtype={labels_array.dtype}")
    print(f"        subject_ids:  {subject_ids_array.shape}")

    # Verify SPD
    n_nan = int(np.isnan(fc_array).sum())
    n_inf = int(np.isinf(fc_array).sum())
    print(f"        NaN count:    {n_nan}")
    print(f"        Inf count:    {n_inf}")

    # Stats on runs used
    print(f"\n      Runs used per subject: min={min(n_runs_used)}, "
          f"max={max(n_runs_used)}, mean={np.mean(n_runs_used):.2f}")
    print(f"      Timepoints per subject: min={min(n_timepoints_used)}, "
          f"max={max(n_timepoints_used)}, mean={np.mean(n_timepoints_used):.0f}")

    output_file = OUTPUT_DIR / "Transdiagnostic_processed.npz"
    np.savez(
        output_file,
        fc_matrices=fc_array,
        labels=labels_array,
        subject_ids=subject_ids_array,
        site_id=TRANSDIAGNOSTIC_SITE_ID,
        site_name=TRANSDIAGNOSTIC_SITE_NAME,
        n_classes=args.n_classes,
        n_rois=N_TARGET_ROIS,
        n_samples=len(fc_array),
        preprocessing_date=time.strftime("%Y-%m-%d %H:%M:%S"),
        atlas="Schaefer 100 (downsampled from Schaefer 400 + Tian 34 subcortical)",
        downsampling_method="drop_subcortical_then_average_groups_of_4",
        multi_run_strategy="concatenate_all_rest_runs",
    )
    print(f"\n[OK] Saved: {output_file}")
    print(f"     Size:  {output_file.stat().st_size/1e6:.1f} MB")
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Subjects:  {len(fc_array)}")
    print(f"  Labels:    {dict(Counter(labels_array.tolist()))}")
    print(f"  FC shape:  {fc_array.shape}")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"  Output:    {output_file}")
    print()
    print("Next: re-run verify_and_discover.py to confirm Transdiagnostic .npz is valid.")
    print("=" * 78)


if __name__ == "__main__":
    main()
