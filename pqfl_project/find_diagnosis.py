"""Find the REAL Transdiagnostic diagnosis file - tmb_dsm was a red herring
(Digit Symbol Matching test, not DSM diagnosis).

Strategy:
1. Check demos.tsv with proper parsing (might have diagnosis column)
2. Check notes.tsv for documentation
3. For every TSV in phenotype/, scan for any column with values like
   "SZ", "Schizophrenia", "BP", "Bipolar", "MDD", "Depression", etc.
4. Cross-reference PANSS (SZ), YMRS (BP), MADRS (MDD) with Group
5. Build inferred diagnosis for each subject
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

TD = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data\Transdiagnostic")
PHENO = TD / "phenotype"

# Diagnosis-like keywords to search for in column values
DIAG_KEYWORDS = {
    "schizo", "schiz", "scz", "sz", "psychosis",
    "bipolar", "mania", "manic",
    "depress", "mdd",
    "anxiety", " gad",
    "adhd",
    "control", "healthy", "hc",
}

print("=" * 78)
print("[1] demos.tsv - first 5 rows (raw)")
print("=" * 78)
demos_path = PHENO / "demos.tsv"
with open(demos_path, encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if i >= 5: break
        print(line.rstrip()[:300])

print()
print("=" * 78)
print("[2] demos.tsv - parsed as CSV with header")
print("=" * 78)
with open(demos_path, encoding="utf-8", errors="replace") as f:
    reader = csv.reader(f)
    rows = list(reader)
print(f"Total rows: {len(rows)}")
if len(rows) >= 2:
    print(f"Row 0 (header?): {rows[0][:15]}...")
    print(f"Row 1: {rows[1][:15]}...")
    print(f"Row 2: {rows[2][:15]}...")
    print(f"Row 3: {rows[3][:15]}...")

print()
print("=" * 78)
print("[3] notes.tsv - FULL CONTENT")
print("=" * 78)
notes_path = PHENO / "notes.tsv"
with open(notes_path, encoding="utf-8", errors="replace") as f:
    for line in f:
        print(line.rstrip())

# Read participants.tsv for ground truth
participants_path = TD / "participants.tsv"
with open(participants_path, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    participants = list(reader)

sk_to_group = {}
for r in participants:
    pid = r["participant_id"].strip()
    group = r.get("Group", "").strip()
    if pid.startswith("sub-NDARINV"):
        sk = pid.replace("sub-NDARINV", "NDAR_INV")
        sk_to_group[sk] = group

# 4. Scan every TSV for diagnosis-like columns
print()
print("=" * 78)
print("[4] Scanning ALL TSVs for diagnosis-like column values")
print("=" * 78)

candidate_files = []
for tsv in sorted(PHENO.glob("*.tsv")):
    if tsv.name.endswith("_definitions.tsv"):
        continue
    try:
        with open(tsv, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows_tsv = list(reader)
        if not rows_tsv:
            continue
        for col in reader.fieldnames:
            values = [str(r.get(col, "")).strip().lower() for r in rows_tsv]
            # Check if any value matches a diagnosis keyword
            matches = [v for v in values if any(kw in v for kw in DIAG_KEYWORDS)]
            if matches and len(matches) >= 5:  # at least 5 subjects with non-empty values
                unique_vals = Counter(values)
                top_vals = unique_vals.most_common(10)
                # Filter out mostly-empty columns
                non_empty = sum(1 for v in values if v and v != "n/a")
                if non_empty >= 10:
                    candidate_files.append({
                        "file": tsv.name,
                        "column": col,
                        "non_empty": non_empty,
                        "top_values": top_vals[:8],
                    })
    except Exception as e:
        print(f"  [WARN] Could not parse {tsv.name}: {e}")

if candidate_files:
    print(f"\nFound {len(candidate_files)} candidate (file, column) pairs with diagnosis-like values:\n")
    for c in candidate_files:
        print(f"  {c['file']}  column='{c['column']}'  ({c['non_empty']} non-empty)")
        for val, n in c["top_values"]:
            print(f"    '{val[:50]}':  {n}")
        print()
else:
    print("No diagnosis-like columns found in any TSV.")

# 5. Strategy: use PANSS, YMRS, MADRS presence as diagnosis proxy
print()
print("=" * 78)
print("[5] Diagnosis inference from symptom severity scales")
print("=" * 78)
print("  PANSS (panss01.tsv) = SZ severity (only SZ patients should have it)")
print("  YMRS  (ymrs01.tsv)  = Mania severity (only BP patients should have it)")
print("  MADRS (madrs01.tsv) = Depression severity (MDD + BP-depressive patients)")
print()

# Load presence of each scale per subject
def load_subjects_with_scale(scale_file):
    path = PHENO / scale_file
    if not path.exists():
        return set()
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {r["subjectkey"].strip() for r in reader if r.get("subjectkey")}

panss_subs = load_subjects_with_scale("panss01.tsv")
ymrs_subs = load_subjects_with_scale("ymrs01.tsv")
madrs_subs = load_subjects_with_scale("madrs01.tsv")
all_pheno_subs = sk_to_group.keys()

print(f"  Subjects in PANSS file: {len(panss_subs)}")
print(f"  Subjects in YMRS file:  {len(ymrs_subs)}")
print(f"  Subjects in MADRS file: {len(madrs_subs)}")
print(f"  Subjects in participants.tsv: {len(all_pheno_subs)}")

# Cross-tab: which subjects have PANSS/YMRS/MADRS by Group
print()
print("  Cross-tab: scale presence by Group")
print(f"  {'Scale':<8}  {'Patient':<10}  {'GenPop':<10}  {'UNKNOWN':<10}")
print("  " + "-" * 50)
for scale_name, subs in [("PANSS", panss_subs), ("YMRS", ymrs_subs), ("MADRS", madrs_subs)]:
    by_group = Counter()
    for s in subs:
        g = sk_to_group.get(s, "UNKNOWN")
        by_group[g] += 1
    print(f"  {scale_name:<8}  {by_group.get('Patient', 0):<10}  "
          f"{by_group.get('GenPop', 0):<10}  {by_group.get('UNKNOWN', 0):<10}")

# 6. Build inferred diagnosis per subject
print()
print("=" * 78)
print("[6] Inferred diagnosis per subject (using severity scale presence)")
print("=" * 78)
inferred = {}
all_subjects = set()
all_subjects.update(panss_subs, ymrs_subs, madrs_subs, all_pheno_subs)

for sk in all_subjects:
    group = sk_to_group.get(sk, "UNKNOWN")
    has_panss = sk in panss_subs
    has_ymrs = sk in ymrs_subs
    has_madrs = sk in madrs_subs

    if group == "GenPop":
        inferred[sk] = "HC"
    elif has_panss:
        inferred[sk] = "SZ"
    elif has_ymrs and not has_madrs:
        inferred[sk] = "BP"
    elif has_madrs and not has_ymrs:
        inferred[sk] = "MDD"
    elif has_ymrs and has_madrs:
        inferred[sk] = "BP"  # BP-depressive
    elif group == "Patient":
        inferred[sk] = "Other_Patient"  # patient without any of these scales
    else:
        inferred[sk] = "UNKNOWN"

inferred_counter = Counter(inferred.values())
print(f"\n  Inferred diagnosis distribution:")
for dx, n in sorted(inferred_counter.items(), key=lambda x: -x[1]):
    print(f"    {dx:<20}  {n:>4}")

# 7. Cross-reference with .h5 parcellated files on disk
print()
print("=" * 78)
print("[7] Cross-reference inferred diagnosis with .h5 files on disk")
print("=" * 78)
h5_dir = TD / "fMRI_timeseries_clean_denoised_GSR_parcellated"
if h5_dir.exists():
    h5_subjects = set()
    for d in h5_dir.iterdir():
        if d.is_dir() and d.name.startswith("NDAR_INV"):
            # Each subject folder has multiple .h5 files (different runs/atlas)
            if any(d.glob("*.h5")):
                h5_subjects.add(d.name)
    print(f"  Subjects with .h5 parcellated files on disk: {len(h5_subjects)}")
    print(f"  Sample h5 subject folders: {sorted(h5_subjects)[:5]}")

    # Cross-reference
    print()
    print(f"  Inferred diagnosis x .h5 availability:")
    final_counts = Counter()
    for sk, dx in inferred.items():
        if sk in h5_subjects:
            final_counts[dx] += 1
    for dx, n in sorted(final_counts.items(), key=lambda x: -x[1]):
        print(f"    {dx:<20}  {n:>4}  (with .h5 on disk)")
    print(f"  TOTAL usable:  {sum(final_counts.values())}")

# 8. Recommend label mapping
print()
print("=" * 78)
print("[8] RECOMMENDED LABEL MAPPING for Transdiagnostic adapter")
print("=" * 78)
print("  Based on the inferred diagnosis above, the adapter should:")
print("    inferred == 'HC'                  -> LABEL_HC (1)")
print("    inferred == 'SZ'                  -> LABEL_SZ (0)")
print("    inferred == 'BP'                  -> LABEL_BP (3)   [for 4-class]")
print("                                            or LABEL_OTHER (2) [for 3-class]")
print("    inferred == 'MDD'                 -> LABEL_OTHER (2)")
print("    inferred == 'Other_Patient'       -> LABEL_OTHER (2)")
print("    inferred == 'UNKNOWN'             -> skip subject")
print("=" * 78)
