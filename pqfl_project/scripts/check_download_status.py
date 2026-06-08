#!/usr/bin/env python3
"""Quick check of downloaded dataset status."""
from pathlib import Path

data = Path("./data")

print("=" * 50)
print("  DOWNLOAD STATUS CHECK")
print("=" * 50)

nii_files = list(data.rglob("*.nii.gz"))
tsv_files = list(data.rglob("*.tsv"))
total_gb = sum(f.stat().st_size for f in nii_files) / (1024**3)

print(f"\nTotal BOLD files (.nii.gz): {len(nii_files)}")
print(f"Total TSV files (.tsv):     {len(tsv_files)}")
print(f"Total BOLD size:            {total_gb:.1f} GB")
print()

for site in ["LA5c", "TCP2025", "COBRE", "SRPBS", "MCIC", "BSNIP2", "FBIRN"]:
    site_dir = data / site
    if not site_dir.exists():
        print(f"  {site:12s} - NOT STARTED")
        continue

    subjects = list(site_dir.glob("sub-*"))
    bolds = list(site_dir.glob("sub-*/func/*task-rest*bold*.nii.gz"))
    bolds += list(site_dir.glob("sub-*/func/*task-rest*bold*.nii"))  # uncompressed
    parts = list(site_dir.glob("participants.tsv"))
    bold_gb = sum(f.stat().st_size for f in bolds) / (1024**3) if bolds else 0

    print(f"  {site:12s} - Subjects: {len(subjects)}, Rest BOLD: {len(bolds)} ({bold_gb:.1f} GB), participants.tsv: {'YES' if parts else 'NO'}")

print()
print("=" * 50)

# Check if ready for preprocessing
ready_sites = []
for site in ["LA5c", "TCP2025"]:
    site_dir = data / site
    if site_dir.exists():
        has_parts = list(site_dir.glob("participants.tsv"))
        has_bold = list(site_dir.glob("sub-*/func/*task-rest*bold*.nii*"))
        if has_parts and has_bold:
            ready_sites.append(site)

if ready_sites:
    print(f"  READY FOR PREPROCESSING: {', '.join(ready_sites)}")
    print(f"  Run: python scripts/preprocess_real_data.py --data_dir ./data --compute_fdt")
else:
    print("  No sites ready yet. Continue downloading!")
print("=" * 50)
