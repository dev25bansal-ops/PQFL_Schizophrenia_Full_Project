#!/usr/bin/env python3
"""Diagnose what's in the data folder."""
from pathlib import Path

data = Path("./data")

print("=" * 50)
print("  DATA FOLDER DIAGNOSTIC")
print("=" * 50)

if not data.exists():
    print(f"  ./data does NOT exist!")
    print(f"  Creating it now...")
    data.mkdir(parents=True, exist_ok=True)
    exit()

# List all top-level items
print("\nTop-level contents:")
for item in sorted(data.iterdir()):
    if item.is_dir():
        sub_count = len(list(item.glob("sub-*")))
        size_gb = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / (1024**3)
        print(f"  [DIR]  {item.name}/  ({sub_count} sub-dirs, {size_gb:.1f} GB)")
    elif item.is_file():
        size_mb = item.stat().st_size / (1024**2)
        print(f"  [FILE] {item.name}  ({size_mb:.1f} MB)")

# Check for OLD wrong datasets
print("\n--- Checking for OLD wrong datasets ---")
wrong_datasets = {
    "ds000239": "WRONG COBRE (structural only, no fMRI)",
    "ds000247": "WRONG LA5c (MEG data, not fMRI)",
    "ds001838": "WRONG SRPBS (unrelated handedness study)",
    "ds005357": "WRONG TCP (Neurogame project)",
}

found_wrong = []
for ds_id, reason in wrong_datasets.items():
    ds_dir = data / ds_id
    if ds_dir.exists():
        size_gb = sum(f.stat().st_size for f in ds_dir.rglob("*") if f.is_file()) / (1024**3)
        print(f"  FOUND: {ds_id}/ ({size_gb:.1f} GB) - {reason}")
        found_wrong.append(ds_id)

# Check for CORRECT datasets
print("\n--- Checking for CORRECT datasets ---")
correct_datasets = {
    "LA5c": {"id": "ds000030", "check": "participants.tsv"},
    "TCP2025": {"id": "ds005237", "check": "participants.tsv"},
    "COBRE": {"id": "COINS", "check": "participants.tsv"},
    "SRPBS": {"id": "BICR", "check": "participants.tsv"},
}

found_correct = []
for site, info in correct_datasets.items():
    site_dir = data / site
    if site_dir.exists():
        has_parts = (site_dir / info["check"]).exists()
        bolds = list(site_dir.glob("sub-*/func/*task-rest*bold*.nii*"))
        size_gb = sum(f.stat().st_size for f in site_dir.rglob("*") if f.is_file()) / (1024**3)
        print(f"  FOUND: {site}/ ({size_gb:.1f} GB) - BOLD files: {len(bolds)}, {info['check']}: {'YES' if has_parts else 'NO'}")
        if has_parts and bolds:
            found_correct.append(site)
    else:
        print(f"  MISSING: {site}/ (needs download from {info['id']})")

# Recommendations
print("\n" + "=" * 50)
print("  RECOMMENDATIONS")
print("=" * 50)

if found_wrong:
    total_wrong_gb = sum(
        sum(f.stat().st_size for f in (data / ds).rglob("*") if f.is_file()) / (1024**3)
        for ds in found_wrong
    )
    print(f"\n  DELETE these WRONG datasets to free {total_wrong_gb:.1f} GB:")
    for ds_id in found_wrong:
        print(f"    rmdir /s /q data\\{ds_id}")

if found_correct:
    print(f"\n  READY for preprocessing: {', '.join(found_correct)}")
    print(f"  Run: python scripts/preprocess_real_data.py --data_dir ./data --compute_fdt")
else:
    print(f"\n  No correct datasets yet. Download with:")
    print(f"    python scripts/download_datasets.py --data_dir ./data --rest_only")

print()
