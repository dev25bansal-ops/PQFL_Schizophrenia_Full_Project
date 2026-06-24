"""BrainLat BOLD re-download v2 - fixed."""
import argparse, csv, os, re, sys, time, gzip
from pathlib import Path
from collections import defaultdict

BRAINLAT_ROOT = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data\BrainLat")
MANIFEST_PATH = BRAINLAT_ROOT / "manifest.csv"
PHENO_PATH = BRAINLAT_ROOT / "MRI data" / "brainlat_demographic_mri.csv"
MRI_DIR = BRAINLAT_ROOT / "MRI data"
DEFAULT_SITES = ["AR", "CLB", "COB", "COA"]
PSITES = ["PCA", "PIB", "PMA", "PSL"]
MIN_BOLD_SIZE_BYTES = 5 * 1024 * 1024

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(MANIFEST_PATH))
    p.add_argument("--brainlat-root", default=str(BRAINLAT_ROOT))
    p.add_argument("--sites", nargs="+", default=DEFAULT_SITES)
    p.add_argument("--include-psites", action="store_true")
    p.add_argument("--include-json", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max", type=int, default=None)
    p.add_argument("--verify-only", action="store_true")
    return p.parse_args()

def extract_subject_id(filename):
    name = filename.lstrip("-")
    m = re.match(r"sub-?([A-Z]+\d+)", name, re.IGNORECASE)
    if not m: return None, None, None, None
    subject_id = m.group(1)
    site_match = re.match(r"([A-Z]+)", subject_id)
    site_code = site_match.group(1) if site_match else None
    lower = name.lower()
    if "bold" not in lower or "rest" not in lower:
        return subject_id, site_code, None, None
    if lower.endswith(".json"): file_type = "json"
    elif lower.endswith(".gz"): file_type = "gz"
    else: file_type = "other"
    return subject_id, site_code, "bold", file_type

def load_phenotype_subjects():
    if not PHENO_PATH.exists():
        print(f"[WARN] Phenotype file not found: {PHENO_PATH}")
        return set()
    subjects = set()
    with open(PHENO_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row["MRI_ID"].strip()
            if not sid.startswith("sub-"): sid = f"sub-{sid}"
            subjects.add(sid.upper())
    return subjects

def find_existing_bold(subject_id, site_code):
    for case_sub in [f"sub-{subject_id}", f"sub-{subject_id.lower()}", f"sub-{subject_id.upper()}"]:
        sub_dir = MRI_DIR / site_code / case_sub
        if not sub_dir.exists(): continue
        func_dir = sub_dir / "func"
        if not func_dir.exists(): continue
        for f in func_dir.iterdir():
            if f.is_file() and "bold" in f.name.lower() and f.name.lower().endswith(".nii.gz"):
                if f.stat().st_size >= MIN_BOLD_SIZE_BYTES: return f
    return None

def target_bids_path(subject_id, site_code):
    func_dir = MRI_DIR / site_code / f"sub-{subject_id}" / "func"
    func_dir.mkdir(parents=True, exist_ok=True)
    return func_dir / f"sub-{subject_id}_task-rest_bold.nii.gz"

def read_manifest(manifest_path):
    entries = []
    with open(manifest_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            syn_id = (row.get("ID") or row.get("id") or "").strip().strip('"')
            name = (row.get("name") or row.get("Name") or "").strip().strip('"')
            entries.append({"syn_id": syn_id, "name": name, "size": 0})
    return entries, fieldnames

def verify_nifti(path):
    if not path.exists(): return False, "file does not exist"
    size = path.stat().st_size
    if size < MIN_BOLD_SIZE_BYTES: return False, f"file too small ({size} bytes)"
    try:
        with gzip.open(path, "rb") as f:
            full_header = f.read(348)
            if len(full_header) >= 348:
                magic = full_header[344:348]
                if magic in (b"n+1\x00", b"ni1\x00"):
                    return True, f"valid NIfTI-1, {size/1e6:.1f} MB"
                header_size = int.from_bytes(full_header[:4], "little")
                if header_size == 348:
                    return True, f"valid NIfTI-1 (no magic), {size/1e6:.1f} MB"
        return False, "not a recognized NIfTI format"
    except Exception as e:
        return False, f"gunzip failed: {e}"

def synapse_login():
    try:
        import synapseclient
    except ImportError:
        print("ERROR: synapseclient not installed. Run: pip install synapseclient")
        sys.exit(1)
    syn = synapseclient.Synapse()
    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if token:
        try:
            syn.login(authToken=token, silent=True)
            print(f"[OK] Logged in via SYNAPSE_AUTH_TOKEN env var")
            return syn
        except Exception as e:
            print(f"[WARN] Token login failed: {e}")
    try:
        syn.login(rememberMe=True, silent=True)
        print(f"[OK] Logged in interactively")
        return syn
    except Exception as e:
        print(f"[ERROR] Synapse login failed: {e}")
        sys.exit(1)

def main():
    args = parse_args()
    sites = list(args.sites)
    if args.include_psites: sites = sites + PSITES

    print("=" * 78)
    print("BrainLat BOLD Re-download (v2)")
    print("=" * 78)
    print(f"  BrainLat root:  {args.brainlat_root}")
    print(f"  Manifest:       {args.manifest}")
    print(f"  Phenotype:      {PHENO_PATH}")
    print(f"  Sites filter:   {sites}")
    print(f"  Include .json:  {args.include_json}")
    print(f"  Dry run:        {args.dry_run}")
    print(f"  Verify only:    {args.verify_only}")
    print()

    print("[1/5] Loading phenotype subjects...")
    pheno_subjects = load_phenotype_subjects()
    print(f"      Subjects with phenotype: {len(pheno_subjects)}")

    print("\n[2/5] Reading Synapse manifest...")
    entries, fieldnames = read_manifest(Path(args.manifest))
    print(f"      Manifest columns: {fieldnames}")
    print(f"      Total entries: {len(entries)}")

    print("\n[3/5] Filtering to BOLD .gz files (skipping .json sidecars)...")
    wanted = []
    skipped_no_pheno = skipped_wrong_site = skipped_json = skipped_other = 0
    for e in entries:
        sub_id, site, modality, file_type = extract_subject_id(e["name"])
        if modality != "bold": continue
        if file_type == "json" and not args.include_json:
            skipped_json += 1; continue
        if file_type not in ("gz", "nii_gz"):
            skipped_other += 1; continue
        if site not in sites:
            skipped_wrong_site += 1; continue
        sid_check = f"sub-{sub_id}".upper()
        if sid_check not in pheno_subjects:
            skipped_no_pheno += 1; continue
        wanted.append({**e, "subject_id": sub_id, "site": site})

    print(f"      Skipped (.json sidecars):      {skipped_json}")
    print(f"      Skipped (wrong site):          {skipped_wrong_site}")
    print(f"      Skipped (no phenotype):        {skipped_no_pheno}")
    print(f"      Skipped (other file type):     {skipped_other}")
    print(f"      WANTED .gz BOLD files:         {len(wanted)}")

    by_site = defaultdict(list)
    for e in wanted: by_site[e["site"]].append(e)
    for site in sorted(by_site):
        print(f"        {site:<6}  {len(by_site[site]):>4} .gz files")

    if not args.include_psites:
        print(f"\n      [INFO] P-sites ({', '.join(PSITES)}) are SKIPPED by default.")
        print(f"             Use --include-psites to download anyway (no phenotype).")

    print("\n[4/5] Checking disk for existing BOLD files...")
    missing = []
    existing = 0
    for e in wanted:
        existing_path = find_existing_bold(e["subject_id"], e["site"])
        if existing_path and not args.force: existing += 1
        else: missing.append(e)
    print(f"      Already on disk (will skip): {existing}")
    print(f"      Missing (need to download):  {len(missing)}")

    if args.max:
        missing = missing[:args.max]
        print(f"      Limited to first {len(missing)} (--max flag)")

    if args.verify_only:
        print("\n[5/5] VERIFY MODE: checking existing BOLD files...")
        good = bad = 0
        bad_files = []
        for e in wanted:
            existing_path = find_existing_bold(e["subject_id"], e["site"])
            if not existing_path: continue
            ok, msg = verify_nifti(existing_path)
            if ok: good += 1
            else:
                bad += 1
                bad_files.append((existing_path, msg))
        print(f"      Valid NIfTI:    {good}")
        print(f"      Invalid/corrupt: {bad}")
        if bad_files:
            print(f"\n      Corrupt files:")
            for p, msg in bad_files[:20]:
                print(f"        {p}  ->  {msg}")
        return

    if not missing:
        print("\n[DONE] Nothing to download.")
        return

    if args.dry_run:
        print(f"\n[5/5] DRY RUN - would download {len(missing)} files:")
        for e in missing[:30]:
            target = target_bids_path(e["subject_id"], e["site"])
            print(f"      {e['syn_id']:<15}  {e['site']}/sub-{e['subject_id']}/  ->  {target.name}")
            print(f"              manifest name: {e['name']}")
        if len(missing) > 30:
            print(f"      ... and {len(missing)-30} more")
        est_mb = len(missing) * 21
        print(f"\n      Estimated total: ~{est_mb} MB (~{est_mb/1024:.1f} GB)")
        print(f"      (Assuming ~21 MB per BOLD file)")
        print("\nRun without --dry-run to actually download.")
        return

    print(f"\n[5/5] Downloading {len(missing)} missing BOLD .gz files...")
    syn = synapse_login()
    success = 0
    failed = []
    total_bytes = 0
    start_time = time.time()

    for i, e in enumerate(missing, 1):
        target = target_bids_path(e["subject_id"], e["site"])
        elapsed = time.time() - start_time
        rate = total_bytes / 1e6 / elapsed if elapsed > 0 else 0
        print(f"\n  [{i}/{len(missing)}] {e['syn_id']}  site={e['site']}  sub={e['subject_id']}  (elapsed {elapsed:.0f}s, {rate:.1f} MB/s avg)")
        try:
            file_entity = syn.get(entity=e["syn_id"], downloadLocation=str(target.parent), ifcollision="overwrite.local")
            downloaded_path = Path(file_entity.path)
            if downloaded_path != target:
                if downloaded_path.exists(): downloaded_path.rename(target)
                elif hasattr(file_entity, "cacheDir") and file_entity.cacheDir:
                    cached = Path(file_entity.cacheDir)
                    if cached.exists(): cached.rename(target)
            if not target.exists(): raise RuntimeError(f"File not at expected path: {target}")
            actual_size = target.stat().st_size
            if actual_size < MIN_BOLD_SIZE_BYTES: raise RuntimeError(f"File too small: {actual_size} bytes")
            ok, msg = verify_nifti(target)
            if not ok: print(f"    [WARN] Downloaded but not valid NIfTI: {msg}")
            success += 1
            total_bytes += actual_size
            print(f"    [OK] {target.name}  ({actual_size/1e6:.1f} MB)  - {msg}")
        except KeyboardInterrupt:
            print(f"\n[ABORTED] {success} downloaded, {len(missing)-i+1} remaining.")
            break
        except Exception as ex:
            print(f"    [FAIL] {ex}")
            failed.append({"entry": e, "error": str(ex)})

    print("\n" + "=" * 78)
    print("DOWNLOAD SUMMARY")
    print("=" * 78)
    print(f"  Successful:   {success}")
    print(f"  Failed:       {len(failed)}")
    print(f"  Total bytes:  {total_bytes/1e9:.2f} GB")
    print(f"  Time:         {(time.time()-start_time)/60:.1f} min")
    if failed:
        retry_csv = BRAINLAT_ROOT / "failed_downloads.csv"
        with open(retry_csv, "w", encoding="utf-8", newline="") as out:
            w = csv.writer(out)
            w.writerow(["synapse_id", "site", "subject_id", "manifest_name", "error"])
            for f in failed:
                e = f["entry"]
                w.writerow([e["syn_id"], e["site"], e["subject_id"], e["name"], f["error"]])
        print(f"\nFailed list saved to: {retry_csv}")

if __name__ == "__main__":
    main()
