"""Inspect COBRE Figshare preprocessed derivatives."""
from pathlib import Path
from collections import Counter
import json

COBRE_FIG = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data\COBRE Preprocessed (Figshare)")

print("=" * 78)
print("[1] Top-level files")
print("=" * 78)
for f in sorted(COBRE_FIG.iterdir()):
    if f.is_file():
        print(f"  {f.name}  ({f.stat().st_size/1024:.1f} KB)")
    elif f.is_dir():
        n_files = sum(1 for _ in f.rglob("*") if _.is_file())
        print(f"  [DIR] {f.name}  ({n_files} files)")

print()
print("=" * 78)
print("[2] extracted/ directory contents")
print("=" * 78)
extracted = COBRE_FIG / "extracted"
if extracted.exists():
    for f in sorted(extracted.iterdir())[:30]:
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size/1024:.1f} KB)")
        elif f.is_dir():
            n_files = sum(1 for _ in f.rglob("*") if _.is_file())
            print(f"  [DIR] {f.name}  ({n_files} files)")

# Look for phenotypic data
print()
print("=" * 78)
print("[3] Phenotype file content")
print("=" * 78)
for candidate_name in [
    "phenotypic_data.csv",
    "phenotypic_data.tsv",
    "participants.tsv",
    "keys_phenotypic_data.json",
    "keys_confounds.json",
    "list_files.json",
    "README.md",
]:
    candidate = extracted / candidate_name
    if candidate.exists():
        print(f"\n--- {candidate.name} ({candidate.stat().st_size/1024:.1f} KB) ---")
        if candidate.suffix == ".json":
            try:
                with open(candidate, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for k, v in list(data.items())[:25]:
                        print(f"  {k}: {v}")
                else:
                    print(f"  Type: {type(data).__name__}, length: {len(data)}")
                    if isinstance(data, list):
                        for item in data[:5]:
                            print(f"  {item}")
            except Exception as e:
                print(f"  [WARN] JSON parse failed: {e}")
                with open(candidate, encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f):
                        if i >= 10: break
                        print(f"  {line.rstrip()[:200]}")
        elif candidate.suffix == ".md":
            with open(candidate, encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if i >= 60: break
                    print(f"  {line.rstrip()}")
        else:
            with open(candidate, encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if i >= 10: break
                    print(f"  {line.rstrip()[:200]}")

# Find BOLD .nii.gz files
print()
print("=" * 78)
print("[4] Find BOLD .nii.gz files")
print("=" * 78)
bold_files = list(extracted.rglob("*bold*.nii.gz"))
print(f"  Total BOLD files found: {len(bold_files)}")
print(f"  Sample BOLD files (first 15):")
for f in bold_files[:15]:
    rel = f.relative_to(extracted)
    print(f"  {rel}  ({f.stat().st_size/1e6:.1f} MB)")

# Find T1w .nii.gz files
print()
print("=" * 78)
print("[5] Find T1w .nii.gz files")
print("=" * 78)
t1w_files = list(extracted.rglob("*T1w*.nii.gz"))
print(f"  Total T1w files found: {len(t1w_files)}")
print(f"  Sample T1w files (first 10):")
for f in t1w_files[:10]:
    rel = f.relative_to(extracted)
    print(f"  {rel}  ({f.stat().st_size/1e6:.1f} MB)")

# File extension distribution
print()
print("=" * 78)
print("[6] File extension distribution")
print("=" * 78)
ext_count = Counter()
for f in extracted.rglob("*"):
    if f.is_file():
        # Get full extension (e.g., .nii.gz, .tar.gz)
        name = f.name.lower()
        if name.endswith(".nii.gz"):
            ext = ".nii.gz"
        elif name.endswith(".tar.gz"):
            ext = ".tar.gz"
        else:
            ext = f.suffix.lower()
        ext_count[ext] += 1
for ext, n in sorted(ext_count.items(), key=lambda x: -x[1])[:15]:
    print(f"  {ext:<15}  {n}")

# Find unique directory structures
print()
print("=" * 78)
print("[7] Top-level directory structure (depth 4)")
print("=" * 78)
seen_dirs = set()
for d in extracted.rglob("*"):
    if d.is_dir():
        rel = d.relative_to(extracted)
        depth = len(rel.parts)
        if depth <= 4 and depth >= 1:
            key = str(rel)
            if key not in seen_dirs:
                seen_dirs.add(key)
                indent = "  " * depth
                n_files_in_dir = sum(1 for _ in d.iterdir() if _.is_file())
                n_subdirs = sum(1 for _ in d.iterdir() if _.is_dir())
                print(f"  {indent}{rel.name}/  ({n_files_in_dir} files, {n_subdirs} subdirs)")
                if len(seen_dirs) >= 50:
                    break

# Find confounds files
print()
print("=" * 78)
print("[8] Find confounds (.tsv) and metadata (.json) files")
print("=" * 78)
tsv_files = list(extracted.rglob("*.tsv"))[:10]
json_files = list(extracted.rglob("*.json"))[:10]
print(f"  TSV files (showing first 10 of {len(list(extracted.rglob('*.tsv')))}):")
for f in tsv_files:
    rel = f.relative_to(extracted)
    print(f"  {rel}  ({f.stat().st_size/1024:.1f} KB)")
print(f"\n  JSON files (showing first 10 of {len(list(extracted.rglob('*.json')))}):")
for f in json_files:
    rel = f.relative_to(extracted)
    print(f"  {rel}  ({f.stat().st_size/1024:.1f} KB)")

# Check if there's a sub-XXX folder structure
print()
print("=" * 78)
print("[9] Subject folder structure analysis")
print("=" * 78)
sub_dirs = sorted([d for d in extracted.rglob("*") if d.is_dir() and d.name.startswith("sub-")])[:10]
print(f"  Total sub-* folders: {len(list(extracted.rglob('sub-*')))}")
if sub_dirs:
    print(f"  Sample subject folders:")
    for sd in sub_dirs[:5]:
        rel = sd.relative_to(extracted)
        print(f"\n  {rel}/")
        for child in sorted(sd.iterdir())[:10]:
            if child.is_file():
                print(f"    {child.name}  ({child.stat().st_size/1e6:.1f} MB)")
            elif child.is_dir():
                n = sum(1 for _ in child.iterdir())
                print(f"    [DIR] {child.name}/  ({n} items)")
else:
    print(f"  No sub-* folders found - data is in a different layout")
    print(f"  Listing ALL directories at depth 1-2:")
    for d in sorted(extracted.rglob("*")):
        if d.is_dir():
            rel = d.relative_to(extracted)
            depth = len(rel.parts)
            if depth <= 2:
                n = sum(1 for _ in d.iterdir() if _.is_file())
                print(f"    {rel}/  ({n} files)")

print()
print("=" * 78)
print("[10] ADAPTER DESIGN NOTES")
print("=" * 78)
print("  Based on the inspection above, the COBRE adapter will:")
print("  1. Locate phenotypic_data CSV (likely extracted/phenotypic_data.csv)")
print("  2. Parse Subject Type column (Control / Patient) and Diagnosis column")
print("  3. Map Patient+Schizophrenia -> LABEL_SZ, Control -> LABEL_HC")
print("  4. Find BOLD .nii.gz files (resting state)")
print("  5. Apply Schaefer 100-ROI parcellation using nilearn")
print("  6. Compute Pearson correlation FC matrix")
print("  7. Regularize to SPD")
print("  8. Save COBRE_processed.npz")
print("=" * 78)
