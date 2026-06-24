"""Quarantine DROP datasets - moves them to data/_dropped/ (reversible)."""
from pathlib import Path
import shutil, sys

DATA_ROOT = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data")
QUARANTINE = DATA_ROOT / "_dropped"

# Datasets to quarantine (with reasons)
DROP_LIST = [
    ("BrainLat",                "Wrong patient population (dementia, 0 SZ patients)"),
    ("TCP2025",                 "Duplicate of Transdiagnostic (same dataset_description.json)"),
    ("MLSP",                    "Incompatible 38-dim FNC format (cannot reconstruct FC matrix)"),
    ("Figshare Psychotic FC",   "Structural features (thickness/FA/MD), not functional connectivity"),
    ("COBRE",                   "Raw XNAT - redundant with 'COBRE Preprocessed (Figshare)'"),
]

# Datasets to KEEP (for verification)
KEEP_LIST = [
    "LA5c",
    "Kaggle_Psychosis_rsFMRI",
    "COBRE Preprocessed (Figshare)",
    "Transdiagnostic",
    "Depression",
    "processed",  # output folder, not a dataset
]

print("=" * 78)
print("Dataset Cleanup - Quarantine DROP list")
print("=" * 78)
print(f"  Data root:    {DATA_ROOT}")
print(f"  Quarantine:   {QUARANTINE}")
print()

# Create quarantine folder
QUARANTINE.mkdir(parents=True, exist_ok=True)
print(f"[OK] Quarantine folder ready: {QUARANTINE}")
print()

# Show BEFORE state
print("=" * 78)
print("BEFORE - Current contents of data/")
print("=" * 78)
for p in sorted(DATA_ROOT.iterdir()):
    if p.name.startswith(".") or p.name == "_dropped":
        continue
    if p.is_dir():
        try:
            files = list(p.rglob("*"))
            n_files = sum(1 for f in files if f.is_file())
            total_gb = sum(f.stat().st_size for f in files if f.is_file()) / 1e9
            tag = "DROP" if p.name in [d[0] for d in DROP_LIST] else "KEEP"
            print(f"  [{tag}] {p.name:<40}  {n_files:>6} files  {total_gb:>7.2f} GB")
        except Exception as e:
            print(f"  [???] {p.name:<40}  (error: {e})")
    else:
        print(f"  [FILE] {p.name}")
print()

# Move each DROP dataset
print("=" * 78)
print("MOVING DROP datasets to quarantine")
print("=" * 78)
moved = []
skipped = []
failed = []

for name, reason in DROP_LIST:
    src = DATA_ROOT / name
    dst = QUARANTINE / name
    print(f"\n  {name}")
    print(f"    Reason: {reason}")

    if not src.exists():
        print(f"    [SKIP] Already moved (not in data/)")
        skipped.append(name)
        continue

    if dst.exists():
        print(f"    [SKIP] Already in quarantine: {dst}")
        skipped.append(name)
        continue

    try:
        # Use shutil.move for cross-device safety
        shutil.move(str(src), str(dst))
        print(f"    [OK] Moved to: {dst}")
        moved.append(name)
    except Exception as e:
        print(f"    [FAIL] {e}")
        failed.append((name, str(e)))

# Show AFTER state
print()
print("=" * 78)
print("AFTER - Contents of data/ (KEEP datasets only)")
print("=" * 78)
for p in sorted(DATA_ROOT.iterdir()):
    if p.name.startswith(".") or p.name == "_dropped":
        continue
    if p.is_dir():
        try:
            files = list(p.rglob("*"))
            n_files = sum(1 for f in files if f.is_file())
            total_gb = sum(f.stat().st_size for f in files if f.is_file()) / 1e9
            print(f"  [KEEP] {p.name:<40}  {n_files:>6} files  {total_gb:>7.2f} GB")
        except Exception as e:
            print(f"  [???] {p.name:<40}  (error: {e})")
    else:
        print(f"  [FILE] {p.name}")

# Show quarantine contents
print()
print("=" * 78)
print("QUARANTINE - Contents of data/_dropped/ (recoverable)")
print("=" * 78)
if QUARANTINE.exists():
    for p in sorted(QUARANTINE.iterdir()):
        if p.is_dir():
            try:
                files = list(p.rglob("*"))
                n_files = sum(1 for f in files if f.is_file())
                total_gb = sum(f.stat().st_size for f in files if f.is_file()) / 1e9
                print(f"  [DROP] {p.name:<40}  {n_files:>6} files  {total_gb:>7.2f} GB")
            except Exception as e:
                print(f"  [???] {p.name:<40}  (error: {e})")

# Summary
print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"  Moved to quarantine:  {len(moved)}  ({', '.join(moved) if moved else 'none'})")
print(f"  Already quarantined:  {len(skipped)}  ({', '.join(skipped) if skipped else 'none'})")
print(f"  Failed:               {len(failed)}  ({', '.join(f[0] for f in failed) if failed else 'none'})")
if failed:
    print(f"\n  Failures:")
    for name, err in failed:
        print(f"    {name}: {err}")

print()
print("Next steps:")
print("  1. Verify the data/ folder now contains only 5 KEEP datasets (shown above)")
print("  2. To restore any dataset: Move-Item data\\_dropped\\<name> data\\<name>")
print("  3. To permanently delete (after confirming Phase 2 works):")
print("     Remove-Item data\\_dropped -Recurse -Force")
print()
print("Now ready to work on the 5 KEEP datasets:")
print("  - LA5c (already preprocessed)")
print("  - Kaggle_Psychosis_rsFMRI (already preprocessed)")
print("  - COBRE Preprocessed (Figshare) - need new adapter")
print("  - Transdiagnostic - need adapter (use .h5 parcellated)")
print("  - Depression - optional, needs fMRIPrep")
print("=" * 78)
