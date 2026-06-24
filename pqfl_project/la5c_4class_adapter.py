"""LA5c 4-class: re-preprocess to include BIPOLAR (49) + ADHD (43) subjects."""
import sys, time, csv
from pathlib import Path
from collections import Counter
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img
from nilearn.datasets import fetch_atlas_schaefer_2018

sys.path.insert(0, r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project")

LA5C = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data\LA5c")
FMRIPREP = LA5C / "derivatives" / "fmriprep"
PARTICIPANTS = LA5C / "participants.tsv"
OUTPUT = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data\processed\LA5c_4class.npz")

DX_MAP = {"CONTROL": 1, "SCHZ": 0, "BIPOLAR": 3, "ADHD": 2}
N_ROIS = 100

# Load participants
pheno = {}
with open(PARTICIPANTS, encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        pheno[row["participant_id"].strip()] = row.get("diagnosis", "").strip()
print(f"Loaded {len(pheno)} participants")
print(f"Diagnosis distribution: {Counter(pheno.values())}")

# Find subjects with BOLD
subjects = []
for sub_id, dx in pheno.items():
    if dx not in DX_MAP: continue
    func_dir = FMRIPREP / sub_id / "func"
    if not func_dir.exists(): continue
    bolds = list(func_dir.glob("*task-rest*bold*preproc*.nii.gz")) or list(func_dir.glob("*task-rest*bold*.nii.gz"))
    if bolds:
        subjects.append((sub_id, dx, bolds[0]))
print(f"\nSubjects with BOLD + valid dx: {len(subjects)}")
print(f"Label distribution: {Counter(DX_MAP[dx] for _, dx, _ in subjects)}")

# Load atlas
print("\nLoading Schaefer 100 atlas...")
atlas = fetch_atlas_schaefer_2018(n_rois=N_ROIS, yeo_networks=7, resolution_mm=1)
atlas_img = nib.load(atlas.maps)
print(f"Atlas shape: {atlas_img.shape}")

# Process
print(f"\nProcessing {len(subjects)} subjects...")
fc_list, lbl_list, sid_list = [], [], []
failed = []
t0 = time.time()

for i, (sub_id, dx, bold_path) in enumerate(subjects, 1):
    if i % 20 == 0 or i == 1:
        el = time.time() - t0
        rate = i/el if el>0 else 0
        eta = (len(subjects)-i)/rate if rate>0 else 0
        print(f"  [{i}/{len(subjects)}] {sub_id}  (elapsed {el:.0f}s, ETA {eta:.0f}s)")
    try:
        bold_img = nib.load(str(bold_path))
        atlas_r = resample_to_img(atlas_img, bold_img, interpolation="nearest", force_resample=True)
        atlas_data = atlas_r.get_fdata().astype(int)
        bold_data = bold_img.get_fdata()
        labels_in_atlas = [l for l in np.unique(atlas_data) if l > 0]
        T = bold_data.shape[3]
        ts = np.zeros((T, N_ROIS))
        for j, lbl in enumerate(labels_in_atlas[:N_ROIS]):
            mask = (atlas_data == lbl)
            if mask.sum() > 0:
                ts[:, j] = bold_data[mask, :].mean(axis=0)
        # Z-score
        m = ts.mean(axis=0, keepdims=True)
        s = ts.std(axis=0, keepdims=True, ddof=1); s[s==0]=1
        ts = (ts - m) / s
        # FC
        fc = np.corrcoef(ts.T)
        fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
        fc = fc + 1e-3*np.eye(fc.shape[0])
        fc = (fc + fc.T) / 2
        if fc.shape != (N_ROIS, N_ROIS):
            failed.append((sub_id, f"shape {fc.shape}")); continue
        fc_list.append(fc); lbl_list.append(DX_MAP[dx]); sid_list.append(sub_id)
    except Exception as e:
        failed.append((sub_id, str(e)[:80]))

el = time.time() - t0
print(f"\nProcessed: {len(fc_list)} in {el:.1f}s, failed: {len(failed)}")
if failed:
    for s, e in failed[:5]: print(f"  {s}: {e}")

fc_arr = np.array(fc_list, dtype=np.float64)
lbl_arr = np.array(lbl_list, dtype=np.int64)
sid_arr = np.array(sid_list)

print(f"\nFC shape: {fc_arr.shape}")
print(f"NaN: {int(np.isnan(fc_arr).sum())}, Inf: {int(np.isinf(fc_arr).sum())}")
print(f"Final labels: {Counter(lbl_arr.tolist())}")

np.savez(OUTPUT, fc_matrices=fc_arr, labels=lbl_arr, subject_ids=sid_arr,
    site_id=3, site_name="LA5c", n_classes=4, n_rois=N_ROIS, n_samples=len(fc_arr),
    preprocessing_date=time.strftime("%Y-%m-%d %H:%M:%S"),
    atlas="Schaefer 100", source="LA5c fMRIPrep",
    label_scheme="SZ=0, HC=1, Other=2, BP=3")
print(f"\n[OK] Saved: {OUTPUT}")
print(f"     Size:  {OUTPUT.stat().st_size/1e6:.1f} MB")
