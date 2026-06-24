"""BrainLat deep diagnostic - tells us exactly what's in there."""
from pathlib import Path
import csv, re, sys
from collections import Counter, defaultdict

ROOT = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data\BrainLat")
PHENO = ROOT / "MRI data" / "brainlat_demographic_mri.csv"
MRI_DIR = ROOT / "MRI data"

print("=" * 78)
print(f"BrainLat root: {ROOT}")
print("=" * 78)

print("\n[1] PHENOTYPE FILE")
print(f"    Path: {PHENO}")
print(f"    Exists: {PHENO.exists()}")
if not PHENO.exists():
    print("    FATAL: phenotype file not found")
    sys.exit(1)

with open(PHENO, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
print(f"    Columns: {list(rows[0].keys())}")
print(f"    Total subjects in CSV: {len(rows)}")

dx_col = "diagnosis"
dx_counter = Counter(r.get(dx_col, "").strip() for r in rows)
print(f"\n[2] DIAGNOSIS DISTRIBUTION ({dx_col} column)")
for dx, n in sorted(dx_counter.items(), key=lambda x: -x[1]):
    print(f"    {dx or '(empty)':<20}  {n:>4}  ({100*n/len(rows):.1f}%)")

def extract_site(mri_id):
    m = re.match(r"sub-([A-Z]+)\d+", mri_id)
    return m.group(1) if m else "???"

site_counter = Counter(extract_site(r["MRI_ID"]) for r in rows)
print(f"\n[3] SITE DISTRIBUTION (extracted from MRI_ID)")
for site, n in sorted(site_counter.items()):
    print(f"    {site:<6}  {n:>4}")

print(f"\n[4] CROSS-TAB: Site x Diagnosis")
cross = defaultdict(lambda: Counter())
for r in rows:
    s = extract_site(r["MRI_ID"])
    dx = r.get(dx_col, "").strip()
    cross[s][dx] += 1

all_dxs = sorted({dx for c in cross.values() for dx in c})
header = f"    {'Site':<6}  " + "  ".join(f"{d:>6}" for d in all_dxs) + "    Total"
print(header)
print("    " + "-" * (len(header) - 4))
for site in sorted(cross):
    cells = "  ".join(f"{cross[site].get(d,0):>6}" for d in all_dxs)
    total = sum(cross[site].values())
    print(f"    {site:<6}  {cells}  {total:>7}")

print(f"\n[5] SUBJECT FOLDERS ON DISK (MRI data/<SITE>/sub-*/)")
disk_counts = {}
disk_ids = {}
for site_dir in sorted(MRI_DIR.iterdir()):
    if not site_dir.is_dir() or not re.match(r"^[A-Z]{2,3}$", site_dir.name):
        continue
    subs = sorted([d.name for d in site_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    disk_counts[site_dir.name] = len(subs)
    disk_ids[site_dir.name] = set(subs)
    print(f"    {site_dir.name:<6}  {len(subs):>4} sub-* folders")

print(f"\n[6] BOLD AVAILABILITY")
total_with_bold = 0
total_no_bold = 0
bold_examples = []
no_bold_examples = []
case_mismatches = []

for site, subs in disk_ids.items():
    n_bold = 0
    for sub in sorted(subs):
        sub_dir = MRI_DIR / site / sub
        func_dir = sub_dir / "func"
        if not func_dir.exists():
            no_bold_examples.append(f"{site}/{sub}: no func/ dir")
            total_no_bold += 1
            continue
        bold_files = list(func_dir.glob("*bold*.nii.gz"))
        if bold_files:
            n_bold += 1
            total_with_bold += 1
            if len(bold_examples) < 3:
                bold_examples.append(f"{site}/{sub}: {bold_files[0].name}")
            expected = f"{sub}_task-rest_bold.nii.gz"
            actual = bold_files[0].name
            if expected != actual and expected.lower() == actual.lower():
                case_mismatches.append(f"{site}/{sub}: dir='{sub}' file='{actual[:len(sub)+5]}'")
        else:
            no_bold_examples.append(f"{site}/{sub}: no *bold*.nii.gz")
            total_no_bold += 1
    print(f"    {site:<6}  {n_bold:>3} / {len(subs):>3} subjects have BOLD")

print(f"\n    TOTAL: {total_with_bold} with BOLD, {total_no_bold} without")
if bold_examples:
    print(f"\n    Sample BOLD files found:")
    for e in bold_examples:
        print(f"      {e}")
if no_bold_examples[:5]:
    print(f"\n    Sample missing-BOLD (first 5):")
    for e in no_bold_examples[:5]:
        print(f"      {e}")
if case_mismatches[:5]:
    print(f"\n    CASE MISMATCHES (first 5) - adapter will fail on Linux:")
    for e in case_mismatches[:5]:
        print(f"      {e}")

print(f"\n[7] PHENOTYPE vs DISK MATCH")
pheno_ids = set()
for r in rows:
    sid = r["MRI_ID"]
    if not sid.startswith("sub-"):
        sid = f"sub-{sid}"
    pheno_ids.add(sid.upper())
all_disk_ids = set()
for site, subs in disk_ids.items():
    all_disk_ids.update(s.upper() for s in subs)
in_both = pheno_ids & all_disk_ids
in_pheno_only = pheno_ids - all_disk_ids
in_disk_only = all_disk_ids - pheno_ids
print(f"    In both pheno + disk:    {len(in_both):>4}")
print(f"    In pheno only (no disk): {len(in_pheno_only):>4}")
print(f"    On disk only (no pheno): {len(in_disk_only):>4}")
if in_pheno_only:
    print(f"    Sample pheno-only (first 10): {sorted(in_pheno_only)[:10]}")
if in_disk_only:
    print(f"    Sample disk-only (first 10): {sorted(in_disk_only)[:10]}")

print(f"\n[8] VERDICT")
mappable_dxs = {"HC", "Control", "Healthy", "CN", "SZ", "Schizophrenia", "SCZ"}
total_useful = sum(1 for r in rows if r.get(dx_col, "").strip() in mappable_dxs)
sz_count = dx_counter.get("SZ", 0) + dx_counter.get("Schizophrenia", 0) + dx_counter.get("SCZ", 0)
print(f"    Subjects with mappable diagnosis:   {total_useful}")
print(f"    Subjects with BOLD on disk:         {total_with_bold}")
print(f"    Schizophrenia patients (SZ/SCZ):    {sz_count}")
print(f"    CN (Cognitively Normal) subjects:   {dx_counter.get('CN', 0)}")
print(f"    AD+FTD+PD+MS (Other):               {sum(dx_counter.get(d,0) for d in ('AD','FTD','bvFTD','PD','MS'))}")
print()
if sz_count == 0:
    print("    --> BrainLat has NO schizophrenia patients.")
    print("        Recommendation: keep as HC (CN) + Other (AD/FTD/PD/MS) augmentation site")
    print("        CN -> LABEL_HC, AD/FTD/PD/MS -> LABEL_OTHER, no SZ contributions")
else:
    print(f"    --> BrainLat has {sz_count} SZ patients - full federation possible")
print("=" * 78)
