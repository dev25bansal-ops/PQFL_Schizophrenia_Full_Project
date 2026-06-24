"""Inspect Transdiagnostic .h5 parcellated timeseries files.
Determines: atlas name, ROI count, file structure, sample data shape.
This is required before writing the Transdiagnostic adapter.
"""
import h5py
from pathlib import Path
from collections import Counter
import numpy as np

H5_DIR = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data\Transdiagnostic\fMRI_timeseries_clean_denoised_GSR_parcellated")

print("=" * 78)
print("[1] Folder structure - sample 3 subjects")
print("=" * 78)

subject_dirs = sorted([d for d in H5_DIR.iterdir() if d.is_dir() and d.name.startswith("NDAR_INV")])
print(f"Total subject folders: {len(subject_dirs)}")

# Show files in first 3 subject folders
for sd in subject_dirs[:3]:
    print(f"\n  {sd.name}/")
    for f in sorted(sd.iterdir()):
        if f.is_file():
            print(f"    {f.name}  ({f.stat().st_size/1024:.1f} KB)")

# Count files per subject
print()
print("=" * 78)
print("[2] Files per subject distribution")
print("=" * 78)
file_counts = Counter()
for sd in subject_dirs:
    n_files = sum(1 for f in sd.iterdir() if f.is_file())
    file_counts[n_files] += 1
for n, c in sorted(file_counts.items()):
    print(f"  {n} files:  {c} subjects")

# Collect all unique filenames across subjects
print()
print("=" * 78)
print("[3] Unique filenames across all subjects")
print("=" * 78)
all_filenames = Counter()
for sd in subject_dirs:
    for f in sd.iterdir():
        if f.is_file():
            all_filenames[f.name] += 1
print(f"Total unique filenames: {len(all_filenames)}")
for name, n in sorted(all_filenames.items(), key=lambda x: -x[1])[:20]:
    print(f"  {name}  ({n} subjects)")

# Inspect ONE .h5 file in detail
print()
print("=" * 78)
print("[4] Detailed inspection of first .h5 file")
print("=" * 78)

first_h5 = None
for sd in subject_dirs:
    h5_files = list(sd.glob("*.h5"))
    if h5_files:
        first_h5 = h5_files[0]
        break

if first_h5 is None:
    print("  [FAIL] No .h5 files found!")
    exit(1)

print(f"  File: {first_h5}")
print(f"  Size: {first_h5.stat().st_size/1024:.1f} KB")
print()

with h5py.File(first_h5, "r") as f:
    print(f"  Top-level keys: {list(f.keys())}")
    print(f"  Top-level attrs: {dict(f.attrs)}")
    print()

    def explore_group(group, prefix=""):
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                print(f"  {prefix}{key}/  (group, attrs={dict(item.attrs)})")
                explore_group(item, prefix + "  ")
            elif isinstance(item, h5py.Dataset):
                print(f"  {prefix}{key}:  shape={item.shape}  dtype={item.dtype}  attrs={dict(item.attrs)}")
                # Print first few values
                if len(item.shape) >= 2:
                    sample = item[:3, :5] if len(item.shape) == 2 else item[:3]
                    print(f"  {prefix}  sample[:3, :5]: {sample}")
                elif len(item.shape) == 1:
                    print(f"  {prefix}  sample[:5]: {item[:5]}")

    explore_group(f)

# Inspect a SECOND .h5 file to check consistency
print()
print("=" * 78)
print("[5] Compare with second subject's .h5 file")
print("=" * 78)

second_h5 = None
for sd in subject_dirs[1:]:
    h5_files = list(sd.glob("*.h5"))
    if h5_files:
        second_h5 = h5_files[0]
        break

if second_h5:
    print(f"  File: {second_h5}")
    with h5py.File(second_h5, "r") as f:
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Dataset):
                print(f"  {key}:  shape={item.shape}  dtype={item.dtype}")
            elif isinstance(item, h5py.Group):
                print(f"  {key}/  (group, keys={list(item.keys())})")

# Inspect ALL .h5 files in the FIRST subject folder
print()
print("=" * 78)
print("[6] All .h5 files in first subject folder")
print("=" * 78)
first_sub = subject_dirs[0]
print(f"  Subject: {first_sub.name}")
all_h5_in_first = sorted(first_sub.glob("*.h5"))
print(f"  Number of .h5 files: {len(all_h5_in_first)}")
for h5 in all_h5_in_first:
    print(f"\n  File: {h5.name}  ({h5.stat().st_size/1024:.1f} KB)")
    with h5py.File(h5, "r") as f:
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Dataset):
                print(f"    {key}:  shape={item.shape}  dtype={item.dtype}")
            elif isinstance(item, h5py.Group):
                print(f"    {key}/  (group)")

# Check if filenames indicate atlas
print()
print("=" * 78)
print("[7] Atlas inference from filenames")
print("=" * 78)
atlas_indicators = Counter()
for sd in subject_dirs[:50]:  # sample first 50
    for f in sd.iterdir():
        if f.is_file() and f.suffix == ".h5":
            name_lower = f.name.lower()
            for atlas in ["schaefer", "aal", "harvard", "ho", "power", "gordon",
                         "yeo", "dk", "desikan", "destrieux", "brodmann"]:
                if atlas in name_lower:
                    atlas_indicators[atlas] += 1
                    break
            else:
                atlas_indicators["(unknown)"] += 1

print(f"  Atlas indicators (from first 50 subjects):")
for atlas, n in sorted(atlas_indicators.items(), key=lambda x: -x[1]):
    print(f"    {atlas}:  {n}")

# Final recommendation
print()
print("=" * 78)
print("[8] ADAPTER DESIGN RECOMMENDATION")
print("=" * 78)
print("  Based on the inspection above, the Transdiagnostic adapter will:")
print("  1. Read demos.tsv (skip row 0 metadata, use row 1 as header)")
print("  2. Build subjectkey -> Primary_Dx mapping")
print("  3. Normalize Primary_Dx (case-insensitive, handle BP1/BPI aliases)")
print("  4. Map to PQFL labels based on 3-class or 4-class scheme")
print("  5. For each subject with .h5 on disk:")
print("     - Load the .h5 parcellated timeseries")
print("     - If ROI count = 100 (Schaefer): use directly")
print("     - If ROI count != 100: log warning, decide strategy")
print("  6. Compute Pearson correlation FC matrix")
print("  7. Regularize to SPD")
print("  8. Save Transdiagnostic_processed.npz")
print()
print("  Expected output: ~241 subjects")
print("    3-class: 13 SZ + 92 HC + 136 Other")
print("    4-class: 13 SZ + 92 HC + 25 BP + 111 Other")
print("=" * 78)
