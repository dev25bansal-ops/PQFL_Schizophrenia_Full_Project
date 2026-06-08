#!/usr/bin/env python3
"""Pre-flight diagnostic: Check BOLD file dimensions before preprocessing.

This script scans all BOLD files in the data directory and reports:
  - Image dimensions (3D vs 4D)
  - Number of volumes (timepoints) for 4D images
  - Repetition time (TR) from NIfTI header
  - Whether the file is MNI-space or T1w-space
  - Whether the file is a preproc BOLD or a brainmask (3D)

Usage:
    python scripts/diagnose_bold_dimensions.py --data_dir ./data --site LA5c
    python scripts/diagnose_bold_dimensions.py --data_dir ./data --site LA5c --limit 5
    python scripts/diagnose_bold_dimensions.py --data_dir ./data
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def diagnose_site(site_name: str, site_dir: Path, limit: int = 0):
    """Diagnose BOLD file dimensions for a single site."""
    import nibabel as nib

    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC: {site_name}")
    print(f"  Directory: {site_dir}")
    print(f"{'='*70}")

    # Find subject directories
    subject_dirs = sorted([d for d in site_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    fmriprep_dir = site_dir / "derivatives" / "fmriprep"
    if not subject_dirs and fmriprep_dir.exists():
        subject_dirs = sorted([d for d in fmriprep_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")])
        print(f"  Using fMRIPrep derivatives: {fmriprep_dir}")

    if not subject_dirs:
        print(f"  ERROR: No subject directories found!")
        return

    print(f"  Found {len(subject_dirs)} subjects")

    # Check participants.tsv
    participants_tsv = site_dir / "participants.tsv"
    if participants_tsv.exists():
        print(f"  participants.tsv: FOUND")
    else:
        print(f"  participants.tsv: MISSING")

    # Check for confounds
    confound_count = 0
    bold_count = 0

    stats = {
        "total_bold_files": 0,
        "mni_preproc": 0,
        "mni_mask": 0,
        "t1w_preproc": 0,
        "t1w_mask": 0,
        "other_bold": 0,
        "volumes_per_file": [],
        "tr_values": [],
        "confound_tsvs": 0,
        "issues": [],
    }

    subjects_to_check = subject_dirs[:limit] if limit > 0 else subject_dirs

    for subj_dir in subjects_to_check:
        func_dir = subj_dir / "func"
        if not func_dir.exists():
            stats["issues"].append(f"{subj_dir.name}: No func/ directory")
            continue

        # Find all BOLD-related files
        bold_files = sorted(func_dir.glob("*bold*.nii*")) + sorted(func_dir.glob("*_preproc.nii*"))
        bold_files = list(set(bold_files))  # Remove duplicates

        confound_files = list(func_dir.glob("*desc-confounds_timeseries.tsv"))
        stats["confound_tsvs"] += len(confound_files)

        for bf in bold_files:
            stats["total_bold_files"] += 1
            name = bf.name

            # Classify file type
            if "space-MNI" in name and "preproc" in name:
                stats["mni_preproc"] += 1
                file_type = "MNI-PREPROC"
            elif "space-MNI" in name and "brainmask" in name:
                stats["mni_mask"] += 1
                file_type = "MNI-MASK  "
            elif "space-T1w" in name and "preproc" in name:
                stats["t1w_preproc"] += 1
                file_type = "T1W-PREPROC"
            elif "space-T1w" in name and "brainmask" in name:
                stats["t1w_mask"] += 1
                file_type = "T1W-MASK  "
            else:
                stats["other_bold"] += 1
                file_type = "OTHER     "

            # Load and check dimensions
            try:
                img = nib.load(str(bf))
                shape = img.shape
                ndim = len(shape)

                if ndim >= 4:
                    n_vols = shape[3]
                    stats["volumes_per_file"].append((subj_dir.name, n_vols, file_type))

                    # Get TR
                    zooms = img.header.get_zooms()
                    tr = zooms[-1] if len(zooms) >= 4 else None
                    if tr is not None and isinstance(tr, (int, float)) and 0.5 < tr < 5.0:
                        stats["tr_values"].append(tr)

                    # Flag short runs
                    if n_vols < 50:
                        stats["issues"].append(
                            f"{subj_dir.name}: Short BOLD run ({n_vols} vols) in {name}"
                        )
                elif ndim == 3:
                    # This is a mask file, not a time series
                    pass
                else:
                    stats["issues"].append(
                        f"{subj_dir.name}: Unexpected dimensions {shape} for {name}"
                    )

            except Exception as e:
                stats["issues"].append(f"{subj_dir.name}: Failed to load {name}: {e}")

    # Print summary
    print(f"\n  --- File Classification ---")
    print(f"  MNI preproc BOLD (4D): {stats['mni_preproc']}")
    print(f"  MNI brain mask (3D):   {stats['mni_mask']}")
    print(f"  T1w preproc BOLD (4D): {stats['t1w_preproc']}")
    print(f"  T1w brain mask (3D):   {stats['t1w_mask']}")
    print(f"  Other BOLD files:      {stats['other_bold']}")
    print(f"  Total BOLD files:      {stats['total_bold_files']}")
    print(f"  Confound TSV files:    {stats['confound_tsvs']}")

    # Volume statistics
    if stats["volumes_per_file"]:
        mni_vols = [v for s, v, t in stats["volumes_per_file"] if "MNI" in t]
        t1w_vols = [v for s, v, t in stats["volumes_per_file"] if "T1W" in t]

        if mni_vols:
            print(f"\n  --- MNI Preproc Volume Statistics ---")
            print(f"  Subjects with MNI data: {len(mni_vols)}")
            print(f"  Min volumes: {min(mni_vols)}")
            print(f"  Max volumes: {max(mni_vols)}")
            print(f"  Mean volumes: {sum(mni_vols)/len(mni_vols):.1f}")
            short_count = sum(1 for v in mni_vols if v < 50)
            if short_count:
                print(f"  WARNING: {short_count} subjects with < 50 volumes")

        if t1w_vols:
            print(f"\n  --- T1w Preproc Volume Statistics ---")
            print(f"  Subjects with T1w data: {len(t1w_vols)}")
            print(f"  Min volumes: {min(t1w_vols)}")
            print(f"  Max volumes: {max(t1w_vols)}")
            print(f"  Mean volumes: {sum(t1w_vols)/len(t1w_vols):.1f}")

    # TR statistics
    if stats["tr_values"]:
        unique_trs = set(stats["tr_values"])
        print(f"\n  --- Repetition Time (TR) ---")
        print(f"  Unique TRs: {unique_trs}")
        print(f"  Most common TR: {max(set(stats['tr_values']), key=stats['tr_values'].count)}")

    # Issues
    if stats["issues"]:
        print(f"\n  --- Issues ({len(stats['issues'])}) ---")
        for issue in stats["issues"][:20]:
            print(f"  ⚠ {issue}")
        if len(stats["issues"]) > 20:
            print(f"  ... and {len(stats['issues']) - 20} more issues")

    # Recommendations
    print(f"\n  --- Recommendations ---")
    if stats["mni_preproc"] > 0:
        print(f"  ✓ Will use MNI preproc BOLD files (recommended)")
        print(f"  ✓ Pipeline will select: *space-MNI152NLin2009cAsym_preproc.nii.gz")
    elif stats["t1w_preproc"] > 0:
        print(f"  ⚠ Only T1w-space preproc found - will use T1w (not ideal for cross-site)")
    if stats["confound_tsvs"] == 0:
        print(f"  ⚠ No confound TSV files found - will skip confound regression")
        print(f"    This is OK for fMRIPrep data, but FD scrubbing will also be skipped")
    if stats["mni_preproc"] > 0 and mni_vols and min(mni_vols) >= 50:
        print(f"  ✓ All MNI BOLD runs have >= 50 volumes - bandpass filtering should work")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Pre-flight diagnostic: Check BOLD file dimensions"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing downloaded datasets")
    parser.add_argument("--site", type=str, default=None,
                        help="Specific site to check (e.g., LA5c)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of subjects to check (0=all)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.site:
        site_dir = data_dir / args.site
        if site_dir.exists():
            diagnose_site(args.site, site_dir, limit=args.limit)
        else:
            print(f"ERROR: Directory not found: {site_dir}")
    else:
        # Check all sites
        for site_dir in sorted(data_dir.iterdir()):
            if site_dir.is_dir() and site_dir.name != "processed":
                diagnose_site(site_dir.name, site_dir, limit=args.limit)


if __name__ == "__main__":
    main()
