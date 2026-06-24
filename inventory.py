import os, csv, json
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / "data_inventory.txt"
lines = []

def w(s=""):
    lines.append(s)

w(f"DATA INVENTORY - {__import__('datetime').datetime.now()}")
w(f"Root: {ROOT}")
w()
w("=== TOP-LEVEL ENTRIES ===")
for p in sorted(ROOT.iterdir()):
    if p.name.startswith("."): continue
    if p.is_dir():
        try:
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9
            w(f"  DIR   {p.name:<30}  {n:>8} files  {sz:>8.2f} GB")
        except Exception as e:
            w(f"  DIR   {p.name:<30}  (error: {e})")
    else:
        w(f"  FILE  {p.name:<30}  {p.stat().st_size/1e6:>8.2f} MB")
w()

for ds in sorted(ROOT.iterdir()):
    if not ds.is_dir() or ds.name in ("processed",) or ds.name.startswith("."):
        continue
    w("=" * 80)
    w(f"DATASET: {ds.name}")
    w("=" * 80)
    files = list(ds.rglob("*"))
    files = [f for f in files if f.is_file()]
    total_gb = sum(f.stat().st_size for f in files) / 1e9
    w(f"Total files: {len(files)}   Total size: {total_gb:.2f} GB")
    w()
    w("--- File extensions (top 10) ---")
    ext_count = {}
    for f in files:
        ext_count[f.suffix.lower()] = ext_count.get(f.suffix.lower(), 0) + 1
    for ext, c in sorted(ext_count.items(), key=lambda x: -x[1])[:10]:
        w(f"  {ext or '(no ext)':<12}  {c} files")
    w()
    w("--- Directory tree (depth 3) ---")
    seen = 0
    for d in sorted(ds.rglob("*")):
        if not d.is_dir(): continue
        rel = d.relative_to(ds)
        depth = len(rel.parts)
        if depth > 3: continue
        if seen > 80: break
        w("  " + "  " * (depth-1) + str(rel).replace("\\", "/"))
        seen += 1
    w()
    w("--- CSV/TSV/JSON/TXT/YAML files (first 20, with first 3 lines) ---")
    text_exts = {".csv", ".tsv", ".json", ".txt", ".md", ".yaml", ".yml", ".log", ".ini"}
    text_files = [f for f in files if f.suffix.lower() in text_exts][:20]
    for f in text_files:
        rel = f.relative_to(ROOT)
        w(f"  >>> {rel}  ({f.stat().st_size/1024:.1f} KB)")
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                for i, ln in enumerate(fh):
                    if i >= 3: break
                    w(f"    {ln.rstrip()[:200]}")
        except Exception as e:
            w(f"    (read error: {e})")
        w()
    w("--- Sample sub-* folder (first one found) ---")
    sample = next((f for f in files if "/sub-" in str(f).replace("\\","/") or "\\sub-" in str(f)), None)
    if sample:
        sub_dir = sample.parent
        while sub_dir.name and not sub_dir.name.startswith("sub-"):
            sub_dir = sub_dir.parent
        w(f"  {sub_dir.relative_to(ROOT)}")
        for f in list(sub_dir.rglob("*"))[:25]:
            if f.is_file():
                w(f"    ./{f.relative_to(sub_dir).as_posix()}   {f.stat().st_size/1e6:.1f} MB")
    else:
        w("  (no sub-* directories)")
    w()

w("=" * 80)
w("PROCESSED NPZ FILES")
w("=" * 80)
proc = ROOT / "processed"
if proc.exists():
    for f in sorted(proc.glob("*.npz")):
        w(f"  {f.name}   {f.stat().st_size/1e6:.1f} MB   {__import__('datetime').datetime.fromtimestamp(f.stat().st_mtime)}")
else:
    w("  (no processed/ folder yet)")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"\nInventory saved to: {OUT}")
print(f"Total lines: {len(lines)}")
print("\n=== PASTE THE CONTENTS BELOW BACK TO THE CHAT ===\n")
print("\n".join(lines))
