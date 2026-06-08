#!/usr/bin/env python3
"""Check LA5c func/ directory contents to understand file structure."""
from pathlib import Path

fmriprep = Path("./data/LA5c/derivatives/fmriprep")

# Get first subject's func directory
subjects = sorted([d for d in fmriprep.iterdir() if d.is_dir() and d.name.startswith("sub-")])

if not subjects:
    print("No subjects found!")
    exit()

print("=" * 60)
print("  LA5c func/ DIRECTORY INSPECTION")
print("=" * 60)

for sub in subjects[:3]:  # Check first 3 subjects
    func_dir = sub / "func"
    anat_dir = sub / "anat"
    
    print(f"\n{sub.name}/")
    
    if func_dir.exists():
        print(f"  func/ ({len(list(func_dir.iterdir()))} files)")
        for f in sorted(func_dir.iterdir()):
            size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
            print(f"    {f.name}  ({size_mb:.1f} MB)")
    else:
        print(f"  func/ - DOES NOT EXIST")
    
    if anat_dir.exists():
        anat_files = list(anat_dir.iterdir())
        print(f"  anat/ ({len(anat_files)} files)")
        for f in sorted(anat_files)[:5]:
            size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
            print(f"    {f.name}  ({size_mb:.1f} MB)")

# Count total file types across all subjects
print(f"\n{'=' * 60}")
print("  FILE TYPE SUMMARY (all subjects)")
print(f"{'=' * 60}")

all_files = list(fmriprep.rglob("*"))
extensions = {}
for f in all_files:
    if f.is_file():
        ext = f.suffix
        if f.name.endswith(".nii.gz"):
            ext = ".nii.gz"
        extensions[ext] = extensions.get(ext, 0) + 1

for ext, count in sorted(extensions.items(), key=lambda x: -x[1]):
    print(f"  {ext:15s} {count:5d} files")

# Check for preproc BOLD files specifically
preproc_bolds = list(fmriprep.rglob("*task-rest*preproc*.nii.gz"))
brainmask_bolds = list(fmriprep.rglob("*task-rest*brainmask*.nii.gz"))
print(f"\n  Preprocessed BOLD:  {len(preproc_bolds)}")
print(f"  Brain mask BOLD:    {len(brainmask_bolds)}")
