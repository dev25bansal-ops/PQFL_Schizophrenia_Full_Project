"""Two parallel checks:
1. Verify the KagglePsychosis_processed.npz is valid
2. Find the Transdiagnostic DSM phenotype file (needed to label SZ/BD/MDD)
"""
from pathlib import Path
import sys
import numpy as np

DATA_ROOT = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data")
PROCESSED = DATA_ROOT / "processed"

print("=" * 78)
print("CHECK 1: Verify processed .npz files")
print("=" * 78)
print()

for npz in sorted(PROCESSED.glob("*.npz")):
    print(f"  {npz.name}  ({npz.stat().st_size/1e6:.1f} MB, "
          f"modified {npz.stat().st_mtime:.0f})")
    try:
        d = np.load(npz, allow_pickle=True)
        print(f"    Keys: {list(d.keys())}")
        if "fc_matrices" in d:
            fc = d["fc_matrices"]
            print(f"    fc_matrices shape: {fc.shape}  dtype: {fc.dtype}")
        if "labels" in d:
            lbl = d["labels"]
            unique, counts = np.unique(lbl, return_counts=True)
            label_names = {0: "SZ", 1: "HC", 2: "Other", 3: "BP"}
            dist = {label_names.get(int(u), f"L{int(u)}"): int(c) for u, c in zip(unique, counts)}
            print(f"    labels distribution: {dist}")
        if "site_id" in d:
            print(f"    site_id: {int(d['site_id'])}")
        if "site_name" in d:
            print(f"    site_name: {str(d['site_name'])}")
        if "subject_ids" in d:
            print(f"    subject_ids: {len(d['subject_ids'])} (sample: {d['subject_ids'][:3]})")
        if "fdt_features" in d:
            fdt = d["fdt_features"]
            print(f"    fdt_features shape: {fdt.shape}")
        # Quick sanity check: are FC matrices finite?
        if "fc_matrices" in d:
            fc = d["fc_matrices"]
            n_nan = int(np.isnan(fc).sum())
            n_inf = int(np.isinf(fc).sum())
            print(f"    NaN count: {n_nan}  Inf count: {n_inf}  "
                  f"({'OK' if n_nan == 0 and n_inf == 0 else 'PROBLEM!'})")
        print(f"    [OK] Valid npz")
    except Exception as e:
        print(f"    [FAIL] Error loading: {e}")
    print()

# ─────────────────────────────────────────────────────────────────────────
# CHECK 2: Find Transdiagnostic DSM phenotype file
# ─────────────────────────────────────────────────────────────────────────
print("=" * 78)
print("CHECK 2: Find Transdiagnostic DSM phenotype file")
print("=" * 78)
print()

TD_ROOT = DATA_ROOT / "Transdiagnostic"
PHENO_DIR = TD_ROOT / "phenotype"

print(f"  Transdiagnostic root: {TD_ROOT}")
print(f"  Phenotype directory:  {PHENO_DIR}")
print(f"  Phenotype exists:     {PHENO_DIR.exists()}")
print()

# Search for DSM-related files anywhere in Transdiagnostic
print("  Searching for files matching 'dsm', 'diag', 'dx'...")
matches = []
for pattern in ["*dsm*", "*diag*", "*dx*", "*patient*"]:
    for p in TD_ROOT.rglob(pattern):
        if p.is_file():
            matches.append(p)

# Also list all TSV/CSV files in phenotype/ (in case DSM file has non-obvious name)
if PHENO_DIR.exists():
    print(f"\n  All TSV/CSV files in phenotype/ ({sum(1 for _ in PHENO_DIR.glob('*.tsv'))} tsv, "
          f"{sum(1 for _ in PHENO_DIR.glob('*.csv'))} csv):")
    for p in sorted(PHENO_DIR.iterdir()):
        if p.is_file() and p.suffix in (".tsv", ".csv"):
            size = p.stat().st_size / 1024
            # For small files, show header line
            header = ""
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    header = f.readline().strip()[:150]
            except Exception:
                pass
            print(f"    {p.name:<45}  {size:>7.1f} KB")
            if header:
                print(f"        header: {header}")

# Show DSM/diag matches with their headers
if matches:
    print(f"\n  Files matching dsm/diag/dx/patient ({len(matches)} found):")
    for p in matches[:20]:
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                header = f.readline().strip()[:200]
            size = p.stat().st_size / 1024
            print(f"    {p.relative_to(TD_ROOT)}  ({size:.1f} KB)")
            print(f"      header: {header}")
            # Show first data row
            with open(p, encoding="utf-8", errors="replace") as f:
                f.readline()  # skip header
                row1 = f.readline().strip()[:200]
            print(f"      row 1:  {row1}")
            print()
        except Exception as e:
            print(f"    {p.relative_to(TD_ROOT)}  (error: {e})")
else:
    print("  [WARN] No DSM/diag/dx files found by name pattern.")

# Check participants.tsv for Group column (already known from inventory)
print()
print("  participants.tsv Group distribution (key file - has Patient/Control):")
participants_tsv = TD_ROOT / "participants.tsv"
if participants_tsv.exists():
    import csv
    with open(participants_tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        print(f"    Total rows: {len(rows)}")
        print(f"    Columns: {list(rows[0].keys()) if rows else 'N/A'}")
        from collections import Counter
        group_counts = Counter(r.get("Group", "") for r in rows)
        site_counts = Counter(r.get("Site", "") for r in rows)
        print(f"    Group distribution: {dict(group_counts)}")
        print(f"    Site distribution:  {dict(site_counts)}")
else:
    print(f"    [FAIL] participants.tsv not found at {participants_tsv}")

print()
print("=" * 78)
print("VERDICT")
print("=" * 78)
print("  If CHECK 1 shows both .npz files valid with reasonable label")
print("    distributions, the existing preprocessing is confirmed.")
print("  For CHECK 2, we're looking for a file with DSM codes (295.x = SZ,")
print("    296.x = BP, 296.x = MDD) to split 'Patient' into SZ/BD/MDD.")
print("  If no DSM file exists, we'll fall back to a simpler scheme:")
print("    Patient -> LABEL_OTHER, Control -> LABEL_HC (2-class subset).")
print("=" * 78)
