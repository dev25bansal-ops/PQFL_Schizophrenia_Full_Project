"""Kaggle 4-class: remap Other -> BP for the 183 BP subjects in Kaggle."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project")
from pqfl.data.dataset_adapters import load_kaggle_psychosis

DATA_ROOT = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data\Kaggle_Psychosis_rsFMRI")
OUTPUT = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data\processed\KagglePsychosis_4class.npz")

print("Loading Kaggle (existing adapter, returns BP as Other=2)...")
site_data = load_kaggle_psychosis(DATA_ROOT)
print(f"  Loaded {len(site_data['labels'])} subjects")
print(f"  Original labels: {np.unique(site_data['labels'], return_counts=True)}")

# Remap Other (2) -> BP (3)
labels = site_data['labels'].copy()
labels[labels == 2] = 3
print(f"\nRemapped labels: {np.unique(labels, return_counts=True)}")
print(f"  0 (SZ): {int((labels==0).sum())}")
print(f"  3 (BP): {int((labels==3).sum())}")

import time
np.savez(OUTPUT,
    fc_matrices=site_data['fc_matrices'],
    labels=labels,
    subject_ids=site_data['subject_ids'],
    site_id=site_data['site_id'],
    site_name=site_data['site_name'],
    n_classes=4, n_rois=100, n_samples=len(labels),
    preprocessing_date=time.strftime("%Y-%m-%d %H:%M:%S"),
    atlas="105 ICN -> 100", source="Kaggle Psychosis",
    label_scheme="SZ=0, BP=3 (4-class)")
print(f"\n[OK] Saved: {OUTPUT}")
print(f"     Size:  {OUTPUT.stat().st_size/1e6:.1f} MB")
