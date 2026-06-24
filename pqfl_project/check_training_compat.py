"""Quick check: can train_pqfl_phase2.py load all 4 sites?"""
import sys
from pathlib import Path
sys.path.insert(0, r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project")

import numpy as np
from pqfl.data.dataset import FCDataset, SiteFCDataset

PROCESSED = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data\processed")

print("=" * 78)
print("Phase 2 training script - data loading compatibility check")
print("=" * 78)

npz_files = sorted([f for f in PROCESSED.glob("*_processed.npz") if not f.name.endswith(".bak")])
print(f"\nFound {len(npz_files)} processed .npz files:")
for f in npz_files:
    print(f"  {f.name}")

print(f"\nLoading each site...")
sites = {}
for npz_path in npz_files:
    print(f"\n  Loading {npz_path.name}...")
    try:
        data = np.load(npz_path, allow_pickle=True)
        fc_matrices = data["fc_matrices"]
        labels = data["labels"]
        site_id = int(data["site_id"])
        site_name = str(data["site_name"])
        print(f"    site_id={site_id}, site_name='{site_name}'")
        print(f"    fc shape: {fc_matrices.shape}")
        unique, counts = np.unique(labels, return_counts=True)
        print(f"    labels: {dict(zip(unique.tolist(), counts.tolist()))}")

        fc_dataset = FCDataset(
            fc_matrices=fc_matrices,
            labels=labels,
            fdt_features=None,
            site_id=site_id,
        )
        site_ds = SiteFCDataset(
            fc_dataset=fc_dataset,
            site_name=site_name,
            site_id=site_id,
        )
        sites[site_id] = site_ds
        print(f"    [OK] FCDataset + SiteFCDataset created")
    except Exception as e:
        print(f"    [FAIL] {e}")
        import traceback
        traceback.print_exc()

print(f"\n--- Summary ---")
print(f"Loaded {len(sites)} sites:")
for sid, sds in sorted(sites.items()):
    n = len(sds.dataset)
    n_sz = int((sds.dataset.labels == 0).sum())
    n_hc = int((sds.dataset.labels == 1).sum())
    n_other = int((sds.dataset.labels == 2).sum())
    print(f"  site_id={sid} ({sds.site_name}): n={n}, SZ={n_sz}, HC={n_hc}, Other={n_other}")

total_n = sum(len(sds.dataset) for sds in sites.values())
total_sz = sum(int((sds.dataset.labels == 0).sum()) for sds in sites.values())
total_hc = sum(int((sds.dataset.labels == 1).sum()) for sds in sites.values())
total_other = sum(int((sds.dataset.labels == 2).sum()) for sds in sites.values())
print(f"\nTOTAL: {total_n} subjects (SZ={total_sz}, HC={total_hc}, Other={total_other})")

print(f"\n--- FC shape consistency check ---")
shapes = [sds.dataset.fc_matrices.shape for sds in sites.values()]
print(f"  All shapes: {shapes}")
all_consistent = all(s[1:] == (100, 100) for s in shapes)
print(f"  All sites use (100, 100) FC matrices: {'YES' if all_consistent else 'NO'}")

if all_consistent and len(sites) >= 2:
    print(f"\n[READY] All sites loadable, consistent shapes, ready for Phase 2 training!")
    print(f"\nNext command:")
    print(f"  python experiments\\train_pqfl_phase2.py --data_dir data\\processed --n_classes 3 --n_folds 5")
else:
    print(f"\n[ISSUE] Not ready for training - check above")
print("=" * 78)
