"""Decode Transdiagnostic Primary_Dx values from demos.tsv (row 1 is metadata)."""
import csv
from collections import Counter
from pathlib import Path

DEMOS = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data\Transdiagnostic\phenotype\demos.tsv")
PARTICIPANTS = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data\Transdiagnostic\participants.tsv")
H5_DIR = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data\Transdiagnostic\fMRI_timeseries_clean_denoised_GSR_parcellated")

# demos.tsv has metadata in row 0; real header in row 1; data from row 2 onwards
with open(DEMOS, encoding="utf-8", errors="replace") as f:
    reader = csv.reader(f)
    all_rows = list(reader)

print(f"Total rows in demos.tsv: {len(all_rows)}")
print(f"Row 0 (metadata): {all_rows[0][:5]}...")
print(f"Row 1 (header):   {all_rows[1][:10]}...")
print()

# Use row 1 as header, rows 2+ as data
header = all_rows[1]
data_rows = all_rows[2:]
print(f"Data rows: {len(data_rows)}")

# Find Primary_Dx column index
try:
    dx_idx = header.index("Primary_Dx")
    group_idx = header.index("Group")
    sk_idx = header.index("subjectkey")
    site_idx = header.index("Site")
except ValueError as e:
    print(f"[FAIL] Could not find column: {e}")
    print(f"Available columns: {header}")
    exit(1)

# Build subjectkey -> Primary_Dx, Group, Site mapping
sub_to_dx = {}
sub_to_group = {}
sub_to_site = {}
for row in data_rows:
    if len(row) <= max(dx_idx, group_idx, sk_idx, site_idx):
        continue
    sk = row[sk_idx].strip()
    dx = row[dx_idx].strip()
    group = row[group_idx].strip()
    site = row[site_idx].strip()
    if sk and sk.startswith("NDAR_INV"):
        sub_to_dx[sk] = dx
        sub_to_group[sk] = group
        sub_to_site[sk] = site

print(f"Subjects with Primary_Dx: {len(sub_to_dx)}")
print()

# Distribution of Primary_Dx
print("=" * 78)
print("[1] Primary_Dx value distribution (all subjects)")
print("=" * 78)
dx_counter = Counter(sub_to_dx.values())
for dx, n in sorted(dx_counter.items(), key=lambda x: -x[1]):
    print(f"  {dx:<15}  {n:>4}")

# Cross-tab Primary_Dx x Group
print()
print("=" * 78)
print("[2] Cross-tab: Primary_Dx x Group")
print("=" * 78)
cross = {}
for sk, dx in sub_to_dx.items():
    g = sub_to_group.get(sk, "UNKNOWN")
    if dx not in cross:
        cross[dx] = Counter()
    cross[dx][g] += 1

print(f"  {'Primary_Dx':<15}  {'Patient':<10}  {'GenPop':<10}  {'Other':<10}  Total")
print("  " + "-" * 65)
for dx in sorted(cross.keys()):
    by_g = cross[dx]
    total = sum(by_g.values())
    print(f"  {dx:<15}  {by_g.get('Patient', 0):<10}  {by_g.get('GenPop', 0):<10}  "
          f"{by_g.get('UNKNOWN', 0) + by_g.get('Other', 0):<10}  {total}")

# Cross-tab Primary_Dx x Site x Group
print()
print("=" * 78)
print("[3] Cross-tab: Primary_Dx x Site")
print("=" * 78)
cross_site = {}
for sk, dx in sub_to_dx.items():
    s = sub_to_site.get(sk, "?")
    if dx not in cross_site:
        cross_site[dx] = Counter()
    cross_site[dx][s] += 1

print(f"  {'Primary_Dx':<15}  {'Site 1':<10}  {'Site 2':<10}  Total")
print("  " + "-" * 50)
for dx in sorted(cross_site.keys()):
    by_s = cross_site[dx]
    total = sum(by_s.values())
    print(f"  {dx:<15}  {by_s.get('1', 0):<10}  {by_s.get('2', 0):<10}  {total}")

# Cross-reference with .h5 files on disk
print()
print("=" * 78)
print("[4] Primary_Dx distribution for subjects with .h5 files on disk")
print("=" * 78)
h5_subjects = set()
if H5_DIR.exists():
    for d in H5_DIR.iterdir():
        if d.is_dir() and d.name.startswith("NDAR_INV"):
            if any(d.glob("*.h5")):
                h5_subjects.add(d.name)

print(f"  Subjects with .h5 on disk: {len(h5_subjects)}")
print()

usable_counter = Counter()
for sk in h5_subjects:
    if sk in sub_to_dx:
        usable_counter[sub_to_dx[sk]] += 1

print(f"  Primary_Dx distribution (only subjects with .h5):")
for dx, n in sorted(usable_counter.items(), key=lambda x: -x[1]):
    print(f"    {dx:<15}  {n:>4}")
print(f"  TOTAL:  {sum(usable_counter.values())}")

# Recommend mapping based on observed values
print()
print("=" * 78)
print("[5] RECOMMENDED Primary_Dx -> PQFL label mapping")
print("=" * 78)
print("  Based on the values observed above, suggested mapping:")
print()
print("  Primary_Dx   ->  PQFL Label")
print("  --------------------------------------")
print("  SZ           ->  LABEL_SZ (0)        # Schizophrenia")
print("  SZA          ->  LABEL_SZ (0)        # Schizoaffective (DSM-5: SZ spectrum)")
print("  BP1          ->  LABEL_BP (3)        # Bipolar I")
print("  BP2          ->  LABEL_BP (3)        # Bipolar II (if present)")
print("  BP           ->  LABEL_BP (3)        # Bipolar NOS (if present)")
print("  MDD          ->  LABEL_OTHER (2)     # Major Depressive Disorder")
print("  GAD          ->  LABEL_OTHER (2)     # Generalized Anxiety Disorder")
print("  ADHD         ->  LABEL_OTHER (2)     # Attention-Deficit/Hyperactivity")
print("  999 or ''    ->  LABEL_HC (1)        # No diagnosis (HC / GenPop)")
print()
print("  Verify against the actual values in [1] before writing the adapter.")
print("=" * 78)
