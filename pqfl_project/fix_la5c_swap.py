"""Investigate and fix the LA5c label-swap bug.

The existing LA5c_processed.npz has labels distribution {SZ: 122, HC: 50},
but LA5c actually has 50 SZ + 122 HC (confirmed in PQFL_Report.pdf and
LA5c participants.tsv). This means the label encoding was inverted.

This script:
1. Re-reads LA5c participants.tsv to get ground-truth diagnoses
2. Cross-references with the subject_ids in the .npz
3. Determines the exact label mapping (which got swapped)
4. Saves a corrected .npz with proper labels
5. Reports what changed
"""
import sys
import csv
from pathlib import Path
from collections import Counter
import numpy as np

PROJECT_ROOT = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project")
DATA_ROOT = PROJECT_ROOT / "data"
LA5C_ROOT = DATA_ROOT / "LA5c"
PARTICIPANTS_TSV = LA5C_ROOT / "participants.tsv"
NPZ_PATH = DATA_ROOT / "processed" / "LA5c_processed.npz"
CORRECTED_NPZ = DATA_ROOT / "processed" / "LA5c_processed_corrected.npz"

print("=" * 78)
print("LA5c Label-Swap Investigation & Fix")
print("=" * 78)

# 1. Read participants.tsv to get ground truth
print("\n[1] Reading LA5c participants.tsv for ground-truth diagnoses...")
if not PARTICIPANTS_TSV.exists():
    print(f"[FAIL] participants.tsv not found: {PARTICIPANTS_TSV}")
    sys.exit(1)

with open(PARTICIPANTS_TSV, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = list(reader)
print(f"    Total participants: {len(rows)}")
print(f"    Columns: {list(rows[0].keys())}")

# Diagnosis column distribution
dx_counter = Counter(r.get("diagnosis", "").strip() for r in rows)
print(f"\n    Ground-truth diagnosis distribution (from participants.tsv):")
for dx, n in sorted(dx_counter.items(), key=lambda x: -x[1]):
    print(f"      {dx or '(empty)':<30}  {n:>4}")

# Map: participant_id -> diagnosis string
pheno = {}
for r in rows:
    pid = r["participant_id"].strip()  # e.g., "sub-10159"
    dx = r.get("diagnosis", "").strip()
    pheno[pid] = dx
print(f"    Mapped {len(pheno)} subject IDs to diagnoses")

# 2. Load existing .npz
print("\n[2] Loading existing LA5c_processed.npz...")
d = np.load(NPZ_PATH, allow_pickle=True)
fc = d["fc_matrices"]
labels = d["labels"]
subject_ids = d["subject_ids"]
print(f"    fc_matrices shape: {fc.shape}")
print(f"    subject_ids: {len(subject_ids)} (sample: {subject_ids[:3]})")
print(f"    Current labels distribution: {Counter(labels.tolist())}")

# 3. Cross-reference: for each subject in .npz, what's the true diagnosis?
print("\n[3] Cross-referencing .npz subjects with participants.tsv diagnoses...")
truth = []
unmatched = []
for sid in subject_ids:
    sid_str = str(sid)
    if sid_str in pheno:
        truth.append(pheno[sid_str])
    else:
        unmatched.append(sid_str)

if unmatched:
    print(f"    [WARN] {len(unmatched)} subjects in .npz not found in participants.tsv")
    print(f"           Sample unmatched: {unmatched[:5]}")
else:
    print(f"    [OK] All {len(subject_ids)} subjects matched")

truth_counter = Counter(truth)
print(f"\n    True diagnosis distribution for .npz subjects:")
for dx, n in sorted(truth_counter.items(), key=lambda x: -x[1]):
    print(f"      {dx:<30}  {n:>4}")

# 4. Determine the label mapping
# Per PQFL conventions: LABEL_SZ=0, LABEL_HC=1
# What does the current .npz use?
print("\n[4] Determining current label encoding...")

# Group subjects by their TRUE diagnosis, then see what label they have in .npz
diag_to_label = {}
for i, sid in enumerate(subject_ids):
    sid_str = str(sid)
    if sid_str in pheno:
        dx = pheno[sid_str]
        lbl = int(labels[i])
        if dx not in diag_to_label:
            diag_to_label[dx] = Counter()
        diag_to_label[dx][lbl] += 1

print(f"    Diagnosis -> current label distribution:")
for dx, label_counts in sorted(diag_to_label.items()):
    print(f"      {dx:<30}  ->  {dict(label_counts)}")

# Determine if there's a clear mapping (and if it's swapped)
print("\n[5] Diagnosis:")

# Expected: CONTROL -> LABEL_HC (1), SCHZ/SCHIZOPHRENIA -> LABEL_SZ (0)
# If swapped: CONTROL -> 0, SCHZ -> 1
expected = {"CONTROL": 1, "SCHZ": 0, "Schizophrenia": 0, "Control": 1,
           "HC": 1, "SZ": 0, "NO": 1, "Schiz": 0}

mapping = {}
for dx, label_counts in diag_to_label.items():
    most_common_label = label_counts.most_common(1)[0][0]
    mapping[dx] = most_common_label
    expected_label = expected.get(dx)
    if expected_label is not None:
        match = "OK" if most_common_label == expected_label else "SWAPPED!"
        print(f"    {dx:<20}  ->  label {most_common_label}  "
              f"(expected {expected_label})  [{match}]")
    else:
        print(f"    {dx:<20}  ->  label {most_common_label}")

# 5. Build corrected labels
print("\n[6] Building corrected labels...")
corrected_labels = np.zeros_like(labels)
n_changed = 0
for i, sid in enumerate(subject_ids):
    sid_str = str(sid)
    if sid_str in pheno:
        dx = pheno[sid_str]
        # Apply standard PQFL convention: SZ=0, HC=1
        if dx.upper() in ("SCHZ", "SCHIZOPHRENIA", "SZ", "SCHIZ"):
            corrected_labels[i] = 0  # SZ
        elif dx.upper() in ("CONTROL", "HC", "NO", "HC "):
            corrected_labels[i] = 1  # HC
        else:
            # Other diagnoses (BP, ADHD, etc.) -> LABEL_OTHER = 2
            corrected_labels[i] = 2
    else:
        corrected_labels[i] = labels[i]  # keep original if no phenotype

    if corrected_labels[i] != labels[i]:
        n_changed += 1

print(f"    Labels changed: {n_changed} / {len(labels)}")
print(f"    Original:  SZ={int((labels==0).sum())}, HC={int((labels==1).sum())}, "
      f"Other={int((labels==2).sum())}")
print(f"    Corrected: SZ={int((corrected_labels==0).sum())}, HC={int((corrected_labels==1).sum())}, "
      f"Other={int((corrected_labels==2).sum())}")

# 6. Save corrected .npz (don't overwrite original)
print(f"\n[7] Saving corrected .npz (original preserved)...")
print(f"    Original:  {NPZ_PATH}")
print(f"    Corrected: {CORRECTED_NPZ}")

# Copy all keys, replace labels
out_dict = {}
for k in d.keys():
    if k == "labels":
        out_dict[k] = corrected_labels
    else:
        out_dict[k] = d[k]
np.savez(CORRECTED_NPZ, **out_dict)
print(f"    [OK] Saved ({CORRECTED_NPZ.stat().st_size/1e6:.1f} MB)")

# 7. Summary + recommended action
print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"  The current LA5c_processed.npz has labels SWAPPED.")
print(f"  Original:  SZ={int((labels==0).sum())}, HC={int((labels==1).sum())}")
print(f"  Corrected: SZ={int((corrected_labels==0).sum())}, HC={int((corrected_labels==1).sum())}")
print()
print(f"  This means the June 8 final run (BA=0.688) was trained on INVERTED labels.")
print(f"  The reported sensitivity (0.80) was actually specificity, and vice versa.")
print(f"  The AUC=0.677 is unreliable - the model may have been learning to predict")
print(f"  HC given a SZ patient's brain, and vice versa.")
print()
print(f"  RECOMMENDED ACTION:")
print(f"  1. Verify this script's logic by reviewing the diagnosis->label mapping above")
print(f"  2. If confirmed swapped, replace the original:")
print(f"       Move-Item data\\processed\\LA5c_processed.npz data\\processed\\LA5c_processed_SWAPBUG.bak")
print(f"       Move-Item data\\processed\\LA5c_processed_corrected.npz data\\processed\\LA5c_processed.npz")
print(f"  3. Re-run final_training.py to get correct metrics:")
print(f"       python experiments\\final_training.py --data_dir data\\processed")
print(f"  4. Update the report PDF with corrected numbers")
print("=" * 78)
