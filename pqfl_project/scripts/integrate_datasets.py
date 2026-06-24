#!/usr/bin/env python3
"""Dataset integration script for federated learning.

This script integrates processed .npz files from multiple sites into a
unified multi-site dataset ready for federated training. It handles:

1. Loading individual site .npz files
2. Validating data quality (SPD matrices, consistent ROI count, labels)
3. Computing ComBat harmonization to remove site effects (optional)
4. Creating federated site partitions with proper metadata
5. Saving a unified multi-site .npz file for training

Usage:
    # Integrate all processed datasets
    python scripts/integrate_datasets.py --data_dir ./data/processed

    # Integrate specific sites only
    python scripts/integrate_datasets.py --data_dir ./data/processed --sites LA5c TCP2025 SPINS

    # Apply ComBat harmonization to remove site effects
    python scripts/integrate_datasets.py --data_dir ./data/processed --combat

    # Validate without saving
    python scripts/integrate_datasets.py --data_dir ./data/processed --validate_only
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
from scipy import linalg as la

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Site role definitions for federated learning
SITE_ROLES = {
    "COBRE": "training",
    "FBIRN": "training",
    "MCIC": "training",
    "LA5c": "training",
    "SRPBS": "training",
    "SPINS": "training",
    "BSNIP2": "validation",
    "TCP2025": "validation",
}


def validate_site_data(site_name, data, expected_n_rois=100):
    """Validate a site's processed data."""
    warnings = []
    is_valid = True

    fc = data.get("fc_matrices")
    if fc is None:
        warnings.append("No fc_matrices found")
        return False, warnings

    if fc.ndim != 3:
        warnings.append(f"FC matrices not 3D: shape={fc.shape}")
        is_valid = False

    n_samples, n_rois1, n_rois2 = fc.shape
    if n_rois1 != n_rois2:
        warnings.append(f"FC matrices not square: {n_rois1}x{n_rois2}")
        is_valid = False

    if n_rois1 != expected_n_rois:
        warnings.append(f"ROI count mismatch: expected {expected_n_rois}, got {n_rois1}")
        is_valid = False

    if n_samples > 0:
        sample_fc = fc[0]
        sym_err = np.max(np.abs(sample_fc - sample_fc.T))
        if sym_err > 1e-6:
            warnings.append(f"FC not symmetric (max error={sym_err:.2e}), will be symmetrized")

        eigvals = la.eigvalsh(sample_fc)
        min_eig = np.min(eigvals)
        if min_eig < 0:
            warnings.append(f"FC not positive definite (min eigenvalue={min_eig:.2e}), will be regularized")

    labels = data.get("labels")
    if labels is None:
        warnings.append("No labels found")
        is_valid = False
    else:
        unique_labels = np.unique(labels)
        if not np.all(np.isin(unique_labels, [0, 1])):
            warnings.append(f"Labels not binary: {unique_labels}")
            is_valid = False

    fdt = data.get("fdt_features")
    if fdt is not None:
        if fdt.shape[0] != n_samples:
            warnings.append(f"FDT count mismatch: {fdt.shape[0]} vs {n_samples} samples")
            is_valid = False

    if labels is not None:
        n_sz = int(labels.sum())
        n_hc = int((1 - labels).sum())
        if n_sz < 5:
            warnings.append(f"Very few SZ subjects: {n_sz}")
        sz_ratio = n_sz / len(labels) if len(labels) > 0 else 0
        if sz_ratio < 0.1 or sz_ratio > 0.9:
            warnings.append(f"Severe class imbalance: SZ ratio={sz_ratio:.2f}")

    return is_valid, warnings


def regularize_spd(fc, lambda_reg=1e-3):
    """Ensure all FC matrices are symmetric positive definite."""
    n_samples, n_rois, _ = fc.shape
    n_regularized = 0

    for i in range(n_samples):
        fc[i] = 0.5 * (fc[i] + fc[i].T)
        eigvals = la.eigvalsh(fc[i])
        min_eig = np.min(eigvals)
        if min_eig < lambda_reg:
            fc[i] += (abs(min_eig) + lambda_reg) * np.eye(n_rois)
            n_regularized += 1

    if n_regularized > 0:
        logger.info(f"  Regularized {n_regularized}/{n_samples} matrices for SPD")

    return fc


def apply_combat_harmonization(fc_matrices, site_ids, labels, pathological=True):
    """Apply ComBat harmonization to remove site effects from FC matrices."""
    from pqfl.riemannian.spd_utils import compute_geodesic_mean, log_map

    n_samples, n_rois, _ = fc_matrices.shape
    unique_sites = np.unique(site_ids)

    logger.info(f"  Applying ComBat harmonization ({len(unique_sites)} sites, {n_samples} samples)")

    if pathological:
        geo_mean = compute_geodesic_mean(fc_matrices)
        logger.info(f"  Geodesic mean computed")

        tangent_vectors = np.zeros((n_samples, n_rois * (n_rois + 1) // 2))
        for i in range(n_samples):
            tv = log_map(fc_matrices[i], geo_mean)
            upper_tri_idx = np.triu_indices(n_rois)
            tangent_vectors[i] = tv[upper_tri_idx]

        harmonized_tangent = _combat_vectorized(tangent_vectors, site_ids, labels)

        harmonized_fc = np.zeros_like(fc_matrices)
        for i in range(n_samples):
            tv_mat = np.zeros((n_rois, n_rois))
            upper_tri_idx = np.triu_indices(n_rois)
            tv_mat[upper_tri_idx] = harmonized_tangent[i]
            tv_mat = tv_mat + tv_mat.T - np.diag(np.diag(tv_mat))

            from pqfl.riemannian.spd_utils import exp_map
            harmonized_fc[i] = exp_map(tv_mat, geo_mean)
    else:
        fc_vectors = fc_matrices.reshape(n_samples, -1)
        harmonized_vectors = _combat_vectorized(fc_vectors, site_ids, labels)
        harmonized_fc = harmonized_vectors.reshape(n_samples, n_rois, n_rois)

    harmonized_fc = regularize_spd(harmonized_fc)
    logger.info(f"  ComBat harmonization complete")
    return harmonized_fc


def _combat_vectorized(data, site_ids, labels):
    """Apply parametric ComBat to vectorized features."""
    unique_sites = np.unique(site_ids)
    n_sites = len(unique_sites)
    n_samples, n_features = data.shape

    X = np.column_stack([np.ones(n_samples), labels])
    n_covariates = X.shape[1]

    Z = np.zeros((n_samples, n_sites))
    for i, site in enumerate(unique_sites):
        Z[site_ids == site, i] = 1

    harmonized = data.copy()

    for j in range(n_features):
        y = data[:, j]
        XZ = np.column_stack([X, Z])
        try:
            coeffs = np.linalg.lstsq(XZ, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        beta = coeffs[:n_covariates]
        gamma = coeffs[n_covariates:]

        residuals = y - XZ @ coeffs
        sigma2 = np.var(residuals)

        if sigma2 < 1e-10:
            continue

        y_biological = X @ beta
        y_site_adjusted = y_biological + residuals
        harmonized[:, j] = y_site_adjusted

    return harmonized


def integrate_datasets(data_dir, sites=None, apply_combat=False, validate_only=False, expected_n_rois=100):
    """Integrate multiple site datasets for federated learning."""
    npz_files = sorted(data_dir.glob("*_processed.npz"))

    if not npz_files:
        logger.error(f"No *_processed.npz files found in {data_dir}")
        return None

    logger.info(f"Found {len(npz_files)} processed dataset files:")

    site_data = {}
    site_names = []
    all_warnings = []

    for npz_file in npz_files:
        site_name = npz_file.stem.replace("_processed", "")

        if sites is not None and site_name not in sites:
            logger.info(f"  Skip {site_name}: not in requested sites")
            continue

        logger.info(f"  Loading {site_name} from {npz_file.name}...")
        data = dict(np.load(npz_file, allow_pickle=True))

        is_valid, warnings = validate_site_data(site_name, data, expected_n_rois)
        for w in warnings:
            logger.warning(f"  {site_name}: {w}")
            all_warnings.append(f"{site_name}: {w}")

        if not is_valid:
            logger.error(f"  SKIP {site_name}: validation failed")
            continue

        site_data[site_name] = data
        site_names.append(site_name)

        n_sz = int(data["labels"].sum())
        n_hc = int((1 - data["labels"]).sum())
        logger.info(f"  {site_name}: {len(data['labels'])} subjects ({n_sz} SZ / {n_hc} HC)")

    if not site_data:
        logger.error("No valid datasets to integrate!")
        return None

    if validate_only:
        logger.info("Validation only mode -- not saving.")
        return {"validated_sites": list(site_data.keys()), "warnings": all_warnings}

    # Stack all data
    all_fc = []
    all_labels = []
    all_site_ids = []
    all_fdt = []
    all_subject_ids = []
    site_id_map = {}

    for idx, (site_name, data) in enumerate(sorted(site_data.items())):
        fc = data["fc_matrices"]
        labels = data["labels"]
        n_samples = len(labels)

        fc = regularize_spd(fc)

        all_fc.append(fc)
        all_labels.append(labels)
        all_site_ids.append(np.full(n_samples, idx, dtype=np.int32))
        site_id_map[site_name] = idx

        fdt = data.get("fdt_features")
        if fdt is not None:
            if fdt.shape[0] == n_samples:
                all_fdt.append(fdt)
            else:
                logger.warning(f"  {site_name}: FDT count mismatch, padding with zeros")
                all_fdt.append(np.zeros((n_samples, fdt.shape[1] if fdt.ndim > 1 else 20)))

        subj_ids = data.get("subject_ids")
        if subj_ids is not None:
            all_subject_ids.append(subj_ids)
        else:
            all_subject_ids.append(np.array([f"{site_name}_{i}" for i in range(n_samples)]))

    fc_matrices = np.concatenate(all_fc, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    site_ids = np.concatenate(all_site_ids, axis=0)
    subject_ids = np.concatenate(all_subject_ids, axis=0)
    fdt_features = np.concatenate(all_fdt, axis=0) if all_fdt else None

    total_n = len(labels)
    total_sz = int(labels.sum())
    total_hc = total_n - total_sz

    logger.info(f"\n  Integrated dataset:")
    logger.info(f"    Total samples: {total_n}")
    logger.info(f"    Total SZ: {total_sz}")
    logger.info(f"    Total HC: {total_hc}")
    logger.info(f"    SZ ratio: {total_sz/total_n:.2f}")
    logger.info(f"    Sites: {site_names}")

    if apply_combat and len(site_names) > 1:
        fc_matrices = apply_combat_harmonization(fc_matrices, site_ids, labels)

    output_path = data_dir / "multisite_federated.npz"

    save_dict = {
        "fc_matrices": fc_matrices,
        "labels": labels,
        "site_ids": site_ids,
        "site_names": np.array(site_names),
        "site_id_map": json.dumps(site_id_map),
        "subject_ids": subject_ids,
        "n_rois": expected_n_rois,
        "n_samples": total_n,
        "n_sz": total_sz,
        "n_hc": total_hc,
        "n_sites": len(site_names),
        "combat_applied": apply_combat,
        "integration_timestamp": datetime.now().isoformat(),
    }

    if fdt_features is not None:
        save_dict["fdt_features"] = fdt_features

    site_roles = {name: SITE_ROLES.get(name, "training") for name in site_names}
    save_dict["site_roles"] = json.dumps(site_roles)

    site_stats = {}
    for site_name, data in site_data.items():
        n_sz = int(data["labels"].sum())
        n_hc = int((1 - data["labels"]).sum())
        site_stats[site_name] = {
            "n_sz": n_sz,
            "n_hc": n_hc,
            "n_total": n_sz + n_hc,
            "sz_ratio": n_sz / (n_sz + n_hc),
            "role": SITE_ROLES.get(site_name, "training"),
        }
    save_dict["site_statistics"] = json.dumps(site_stats)

    np.savez_compressed(output_path, **save_dict)
    logger.info(f"\n  Saved integrated dataset to: {output_path}")

    # Print summary
    print(f"\n{'='*74}")
    print(f"  MULTI-SITE FEDERATED DATASET INTEGRATION COMPLETE")
    print(f"{'='*74}\n")

    print(f"  {'Site':12s} | {'Role':10s} | {'ID':>3s} | {'SZ':>4s} | {'HC':>4s} | {'Total':>5s} | {'SZ%':>5s}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*3}-+-{'-'*4}-+-{'-'*4}-+-{'-'*5}-+-{'-'*5}")

    for site_name in sorted(site_names):
        stats = site_stats[site_name]
        sid = site_id_map[site_name]
        print(f"  {site_name:12s} | {stats['role']:10s} | {sid:3d} | {stats['n_sz']:4d} | {stats['n_hc']:4d} | {stats['n_total']:5d} | {stats['sz_ratio']:.1%}")

    print(f"  {'TOTAL':12s} | {'':10s} | {'':3s} | {total_sz:4d} | {total_hc:4d} | {total_n:5d} | {total_sz/total_n:.1%}")

    print(f"\n  ComBat harmonization: {'Applied' if apply_combat else 'Not applied'}")
    print(f"  FDT features: {'Available' if fdt_features is not None else 'Not available'}")
    print(f"  Output: {output_path}")

    return {
        "output_path": str(output_path),
        "n_sites": len(site_names),
        "n_samples": total_n,
        "n_sz": total_sz,
        "n_hc": total_hc,
        "combat_applied": apply_combat,
        "site_statistics": site_stats,
        "warnings": all_warnings,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Integrate multi-site datasets for federated learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing *_processed.npz files")
    parser.add_argument("--sites", nargs="+", default=None,
                        help="Specific sites to integrate (default: all available)")
    parser.add_argument("--combat", action="store_true",
                        help="Apply ComBat harmonization to remove site effects")
    parser.add_argument("--validate_only", action="store_true",
                        help="Only validate data, don't save integrated file")
    parser.add_argument("--n_rois", type=int, default=100,
                        help="Expected number of ROIs (default: 100)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Directory not found: {data_dir}")
        return

    result = integrate_datasets(
        data_dir=data_dir,
        sites=args.sites,
        apply_combat=args.combat,
        validate_only=args.validate_only,
        expected_n_rois=args.n_rois,
    )

    if result is not None:
        print(f"""
========================================
NEXT STEPS:
========================================

1. Verify integrated dataset:
   python -c "import numpy as np; d=np.load('{data_dir}/multisite_federated.npz'); print('FC:', d['fc_matrices'].shape, 'Labels:', d['labels'].shape, 'Sites:', d['site_ids'].shape)"

2. Train federated model:
   python experiments/final_training.py --data_dir {data_dir} --n_rois {args.n_rois}
""")


if __name__ == "__main__":
    main()
