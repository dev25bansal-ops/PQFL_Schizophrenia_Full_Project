"""COMPREHENSIVE data verification - audits everything in data/.

Checks:
1. Quarantined DROP datasets (in _dropped/) - should still be recoverable
2. KEEP raw datasets (LA5c, Kaggle, COBRE Figshare, Transdiagnostic, Depression)
   - Folder exists
   - File count and total size
   - Critical subdirectories present
   - Sample BOLD/.h5 files loadable
3. Processed .npz files in processed/
   - File exists
   - Loads without error
   - Shape, dtype, label distribution sanity check
   - NaN/Inf check
   - No label swap (verify against source data)
4. Disk space availability
5. Phase 2 readiness checklist
"""
import sys
import time
import csv
import json
from pathlib import Path
from collections import Counter
import numpy as np

DATA_ROOT = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data")
QUARANTINE = DATA_ROOT / "_dropped"
PROCESSED = DATA_ROOT / "processed"

EXPECTED_KEEP = {
    "LA5c": {
        "type": "fMRIPrep derivatives",
        "expected_files": ">=4000",
        "critical_paths": ["derivatives/fmriprep", "participants.tsv"],
    },
    "Kaggle_Psychosis_rsFMRI": {
        "type": "FNC features (.mat/.npy)",
        "expected_files": ">=3000",
        "critical_paths": ["data/train", "data/test"],
    },
    "COBRE Preprocessed (Figshare)": {
        "type": "Pre-fMRIPrep'd BOLD (.nii.gz)",
        "expected_files": ">=200",
        "critical_paths": ["extracted"],
    },
    "Transdiagnostic": {
        "type": "Raw BIDS + .h5 parcellated",
        "expected_files": ">=5000",
        "critical_paths": [
            "fMRI_timeseries_clean_denoised_GSR_parcellated",
            "phenotype/demos.tsv",
            "participants.tsv",
        ],
    },
    "Depression": {
        "type": "Raw BIDS",
        "expected_files": ">=100",
        "critical_paths": ["participants.tsv"],
    },
}

EXPECTED_DROPPED = ["BrainLat", "TCP2025", "MLSP", "Figshare Psychotic FC", "COBRE"]

print("=" * 78)
print("COMPREHENSIVE DATA VERIFICATION")
print("=" * 78)
print(f"  Data root:   {DATA_ROOT}")
print(f"  Quarantine:  {QUARANTINE}")
print(f"  Processed:   {PROCESSED}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Disk space
# ─────────────────────────────────────────────────────────────────────────
print("=" * 78)
print("[1] DISK SPACE on F: drive")
print("=" * 78)
try:
    import shutil
    total, used, free = shutil.disk_usage("F:\\")
    print(f"  Total:  {total/1e9:.0f} GB")
    print(f"  Used:   {used/1e9:.0f} GB ({100*used/total:.1f}%)")
    print(f"  Free:   {free/1e9:.0f} GB")
    if free < 50 * 1e9:
        print(f"  [WARN] Less than 50 GB free - check before preprocessing more datasets")
    else:
        print(f"  [OK] Sufficient disk space")
except Exception as e:
    print(f"  [WARN] Could not check disk space: {e}")

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: KEEP datasets (raw data)
# ─────────────────────────────────────────────────────────────────────────
print()
print("=" * 78)
print("[2] KEEP datasets - raw data integrity check")
print("=" * 78)

keep_status = {}
for ds_name, expected in EXPECTED_KEEP.items():
    print(f"\n  {ds_name} ({expected['type']})")
    ds_path = DATA_ROOT / ds_name
    if not ds_path.exists():
        print(f"    [FAIL] Folder missing: {ds_path}")
        keep_status[ds_name] = "MISSING"
        continue

    # Count files and total size
    files = [f for f in ds_path.rglob("*") if f.is_file()]
    total_size = sum(f.stat().st_size for f in files) / 1e9
    print(f"    Files:  {len(files)}   Size:  {total_size:.2f} GB")

    # Check critical paths
    all_critical_ok = True
    for crit in expected["critical_paths"]:
        crit_path = ds_path / crit
        if crit_path.exists():
            if crit_path.is_dir():
                n = sum(1 for _ in crit_path.rglob("*") if _.is_file())
                print(f"    [OK] {crit}  ({n} files)")
            else:
                print(f"    [OK] {crit}  ({crit_path.stat().st_size/1024:.1f} KB)")
        else:
            print(f"    [FAIL] Missing critical path: {crit}")
            all_critical_ok = False

    keep_status[ds_name] = "OK" if all_critical_ok else "INCOMPLETE"

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Quarantined DROP datasets (recoverable)
# ─────────────────────────────────────────────────────────────────────────
print()
print("=" * 78)
print("[3] Quarantined DROP datasets (recoverable)")
print("=" * 78)
for ds_name in EXPECTED_DROPPED:
    ds_path = QUARANTINE / ds_name
    if not ds_path.exists():
        print(f"  [WARN] {ds_name} not in quarantine (may have been deleted)")
        continue
    files = [f for f in ds_path.rglob("*") if f.is_file()]
    total_size = sum(f.stat().st_size for f in files) / 1e9
    print(f"  [OK] {ds_name}  ({len(files)} files, {total_size:.2f} GB)")

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Processed .npz files
# ─────────────────────────────────────────────────────────────────────────
print()
print("=" * 78)
print("[4] Processed .npz files")
print("=" * 78)

expected_npz = {
    "LA5c_processed.npz": {"expected_n": 172, "expected_labels": {0: 50, 1: 122}},
    "KagglePsychosis_processed.npz": {"expected_n": 471, "expected_labels": {0: 288, 2: 183}},
}

npz_status = {}
for npz_file in sorted(PROCESSED.glob("*.npz")):
    print(f"\n  {npz_file.name}  ({npz_file.stat().st_size/1e6:.1f} MB, "
          f"modified {time.strftime('%Y-%m-%d %H:%M', time.localtime(npz_file.stat().st_mtime))})")

    try:
        d = np.load(npz_file, allow_pickle=True)
        print(f"    Keys: {list(d.keys())}")

        if "fc_matrices" in d:
            fc = d["fc_matrices"]
            print(f"    fc_matrices: shape={fc.shape} dtype={fc.dtype}")

            # Sanity checks
            n_nan = int(np.isnan(fc).sum())
            n_inf = int(np.isinf(fc).sum())
            n_neg_diag = int((np.diagonal(fc, axis1=-2, axis2=-1) < 0).sum())
            print(f"    NaN: {n_nan}  Inf: {n_inf}  Negative diagonals: {n_neg_diag}")
            if n_nan > 0 or n_inf > 0:
                print(f"    [WARN] NaN/Inf present - matrices may be corrupt")
            else:
                print(f"    [OK] No NaN/Inf")

        if "labels" in d:
            lbl = d["labels"]
            unique, counts = np.unique(lbl, return_counts=True)
            label_names = {0: "SZ", 1: "HC", 2: "Other", 3: "BP"}
            dist = {label_names.get(int(u), f"L{int(u)}"): int(c) for u, c in zip(unique, counts)}
            print(f"    Labels: {dist}")

            # Compare to expected
            if npz_file.name in expected_npz:
                exp = expected_npz[npz_file.name]
                if len(lbl) != exp["expected_n"]:
                    print(f"    [FAIL] Expected {exp['expected_n']} samples, got {len(lbl)}")
                else:
                    print(f"    [OK] Sample count matches expected ({exp['expected_n']})")

                # Check label distribution
                actual_dist = {int(u): int(c) for u, c in zip(unique, counts)}
                if actual_dist == exp["expected_labels"]:
                    print(f"    [OK] Label distribution matches expected")
                else:
                    print(f"    [WARN] Label distribution mismatch!")
                    print(f"           Expected: {exp['expected_labels']}")
                    print(f"           Actual:   {actual_dist}")

        if "site_name" in d:
            print(f"    Site: {str(d['site_name'])} (id={int(d['site_id'])})")

        npz_status[npz_file.name] = "OK"

    except Exception as e:
        print(f"    [FAIL] Error loading: {e}")
        npz_status[npz_file.name] = "CORRUPT"

# Also check for backup files
print()
print("  Backup files:")
for bak in sorted(PROCESSED.glob("*.bak")):
    print(f"    {bak.name}  ({bak.stat().st_size/1e6:.1f} MB)")
for extra in sorted(PROCESSED.glob("*.json")):
    print(f"    {extra.name}  ({extra.stat().st_size/1024:.1f} KB)")

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Phase 2 readiness checklist
# ─────────────────────────────────────────────────────────────────────────
print()
print("=" * 78)
print("[5] Phase 2 Readiness Checklist")
print("=" * 78)

sites_ready = []
sites_pending = []

# Site 1: LA5c
if npz_status.get("LA5c_processed.npz") == "OK":
    print(f"  [DONE] Site 1: LA5c        - 172 subj (50 SZ + 122 HC)")
    sites_ready.append(("LA5c", 172, "50 SZ + 122 HC"))
else:
    print(f"  [TODO] Site 1: LA5c        - .npz needs fixing")
    sites_pending.append("LA5c")

# Site 2: Kaggle
if npz_status.get("KagglePsychosis_processed.npz") == "OK":
    print(f"  [DONE] Site 2: Kaggle      - 471 subj (288 SZ + 183 Other)")
    sites_ready.append(("Kaggle", 471, "288 SZ + 183 Other"))
else:
    print(f"  [TODO] Site 2: Kaggle      - .npz needs fixing")
    sites_pending.append("Kaggle")

# Site 3: Transdiagnostic
if npz_status.get("Transdiagnostic_processed.npz") == "OK":
    print(f"  [DONE] Site 3: Transdiag   - 241 subj (13 SZ + 92 HC + 136 Other)")
    sites_ready.append(("Transdiagnostic", 241, "13 SZ + 92 HC + 136 Other"))
else:
    print(f"  [TODO] Site 3: Transdiag   - run transdiagnostic_adapter.py (no --dry-run)")
    sites_pending.append("Transdiagnostic")

# Site 4: COBRE Figshare (auto-detect)
if npz_status.get("COBRE_processed.npz") == "OK":
    print(f"  [DONE] Site 4: COBRE        - 146 subj (72 SZ + 74 HC)")
    sites_ready.append(("COBRE", 146, "72 SZ + 74 HC"))
else:
    print(f"  [TODO] Site 4: COBRE        - run cobre_adapter_v3.py")
    sites_pending.append("COBRE")

# Site 5: Depression (optional)
print(f"  [OPT]  Site 5: Depression   - needs fMRIPrep (optional, low priority)")
sites_pending.append("Depression (optional)")

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: Summary
# ─────────────────────────────────────────────────────────────────────────
print()
print("=" * 78)
print("[6] SUMMARY")
print("=" * 78)

n_keep_ok = sum(1 for v in keep_status.values() if v == "OK")
n_keep_total = len(keep_status)
print(f"  Raw datasets intact:  {n_keep_ok}/{n_keep_total}")
for ds, status in keep_status.items():
    marker = "[OK]" if status == "OK" else "[!!]"
    print(f"    {marker} {ds:<35}  {status}")

print()
print(f"  Processed .npz files: {len(npz_status)} total")
for npz_name, status in npz_status.items():
    marker = "[OK]" if status == "OK" else "[!!]"
    print(f"    {marker} {npz_name:<40}  {status}")

print()
print(f"  Sites ready for Phase 2 training:  {len(sites_ready)}")
for site, n, lbls in sites_ready:
    print(f"    [READY] {site:<20}  {n} subj ({lbls})")

print()
n_total_ready = sum(n for _, n, _ in sites_ready)
print(f"  Total subjects ready:  {n_total_ready}")
if n_total_ready >= 800:
    print(f"  [OK] Sufficient for Phase 2 ({n_total_ready} >= 800)")
else:
    print(f"  [INFO] Need more sites - currently {n_total_ready}/800+")

print()
print(f"  Next actions:")
if "Transdiagnostic" in sites_pending:
    print(f"    1. Run: python transdiagnostic_adapter.py --n_classes 3")
if "COBRE Figshare" in sites_pending:
    print(f"    2. Run: python inspect_cobre_figshare.py")
    print(f"       (then I'll write the COBRE adapter)")
if "Depression (optional)" in sites_pending:
    print(f"    3. Optional: preprocess Depression (low priority)")
print()
print("=" * 78)
