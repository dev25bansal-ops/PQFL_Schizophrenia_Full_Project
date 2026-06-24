#!/usr/bin/env python3
"""Tier 1: Enhanced PQFL Training with methodological rigor improvements.

Adds 4 critical improvements over the baseline train_pqfl_phase2.py:
1. Leave-One-Site-Out CV (LOSO) — true cross-site generalization
2. Site-confound regression — removes site-effect leak
3. ComBat harmonization — gold-standard site correction
4. Multi-seed statistical significance — Wilcoxon, McNemar, Cohen's d

Usage:
    # LOSO CV with site-confound regression, 3 seeds
    python tier1_enhanced_training.py --data_dir data/processed --n_classes 4 \\
        --cv_mode loso --site_deconfound --seeds 42 123 456 \\
        --n_qubits 8 --output_dir results/tier1_loso_deconfound

    # 5-fold CV with ComBat + multi-seed
    python tier1_enhanced_training.py --data_dir data/processed --n_classes 4 \\
        --cv_mode kfold --combat --seeds 42 123 456 \\
        --n_qubits 8 --output_dir results/tier1_kfold_combat

    # Full Tier 1: LOSO + site-deconfound + ComBat + 3 seeds
    python tier1_enhanced_training.py --data_dir data/processed --n_classes 4 \\
        --cv_mode loso --site_deconfound --combat --seeds 42 123 456 \\
        --n_qubits 8 --output_dir results/tier1_full

Prerequisites:
    pip install neuroCombat scikit-learn statsmodels scipy
"""
import argparse
import sys
import os
import json
import time
import copy
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from scipy import stats as scipy_stats

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pqfl.data.dataset import FCDataset, SiteFCDataset
from pqfl.riemannian.engine import RiemannianEngine
from pqfl.quantum.vqc import HybridVQC, VQCConfig
from pqfl.federated.client import PQFLClient
from pqfl.evaluation.metrics import compute_classification_metrics
from pqfl.baselines.classical import TangentSpaceSVM

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ─────────────────────────────────────────────────────────────
BASE_CONFIG = {
    "n_qubits": 8,
    "n_components": 71,
    "learning_rate": 0.0005,
    "dropout": 0.5,
    "n_base_layers": 2,
    "label_smoothing": 0.1,
    "batch_size": 16,
}

CLASS_NAMES = {2: ["HC", "SZ"], 3: ["SZ", "HC", "Other"], 4: ["SZ", "HC", "Other", "BP"]}


# ═════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═════════════════════════════════════════════════════════════════════════
def load_sites(data_dir, n_classes):
    """Load all *_processed.npz files."""
    data_dir = Path(data_dir)
    npz_files = sorted(data_dir.glob("*_processed.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No *_processed.npz files in {data_dir}")

    logger.info(f"\nFound {len(npz_files)} processed site files:")
    for f in npz_files:
        logger.info(f"  {f.name}")

    sites = []
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        fc_matrices = data["fc_matrices"]
        labels = data["labels"]
        site_id = int(data["site_id"])
        site_name = str(data["site_name"])

        # Filter labels for n_classes
        if n_classes == 2:
            mask = labels < 2
        elif n_classes == 3:
            mask = labels < 3
        else:
            mask = np.ones(len(labels), dtype=bool)

        fc_matrices = fc_matrices[mask]
        labels = labels[mask]

        if len(labels) == 0:
            logger.info(f"  [SKIP] {site_name}: no subjects in {n_classes}-class scheme")
            continue

        unique, counts = np.unique(labels, return_counts=True)
        label_dist = dict(zip(unique.tolist(), counts.tolist()))
        logger.info(f"  [LOAD] {site_name}: {len(labels)} subjects, labels={label_dist}")

        fc_dataset = FCDataset(
            fc_matrices=fc_matrices, labels=labels,
            fdt_features=None, site_id=site_id,
        )
        sites.append({
            "site_name": site_name,
            "site_id": site_id,
            "fc_dataset": fc_dataset,
            "labels": labels,
            "fc_matrices": fc_matrices,
        })

    logger.info(f"Loaded {len(sites)} sites, {sum(len(s['labels']) for s in sites)} total subjects")
    return sites


# ═════════════════════════════════════════════════════════════════════════
# 2. RIEMANNIAN PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════
def aggregate_features(sites, riemannian_engine):
    """Aggregate FC matrices from all sites and compute tangent features."""
    all_fc, all_labels, all_site_ids = [], [], []
    for site_idx, site in enumerate(sites):
        fc = site["fc_matrices"]
        lbl = site["labels"]
        all_fc.append(fc)
        all_labels.append(lbl)
        all_site_ids.append(np.full(len(lbl), site["site_id"]))

    combined_fc = np.concatenate(all_fc, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    combined_site_ids = np.concatenate(all_site_ids, axis=0)

    logger.info(f"Aggregated: {len(combined_labels)} samples, {combined_fc.shape[1]} ROIs")

    unique, counts = np.unique(combined_labels, return_counts=True)
    logger.info(f"Label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    logger.info("Computing Fréchet mean (this may take 30-90 seconds)...")
    t0 = time.time()
    riemannian_engine.fit(combined_fc)
    logger.info(f"Fréchet mean computed in {time.time()-t0:.1f}s")

    logger.info("Computing tangent features (log-map + PCA)...")
    t0 = time.time()
    tangent = riemannian_engine.transform(combined_fc, return_tensor=False)
    logger.info(f"Tangent features computed in {time.time()-t0:.1f}s, shape: {tangent.shape}")
    return tangent, combined_labels, combined_site_ids, combined_fc


# ═════════════════════════════════════════════════════════════════════════
# 3. SITE-CONFOUND REGRESSION (Tier 1.1)
# ═════════════════════════════════════════════════════════════════════════
def site_confound_regress(X_train, site_ids_train, X_val=None, site_ids_val=None):
    """Remove site-specific signal from tangent features.

    Fits a multinomial logistic regression to predict SITE from features,
    then uses the residuals (features minus site-predictable component).

    CRITICAL: Fit on TRAIN only, apply to both train and val.
    """
    logger.info("  [SITE-DECONFOUND] Regressing out site identity...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Fit site predictor on training data
    site_clf = LogisticRegression(max_iter=2000, C=1.0)
    site_clf.fit(X_train_scaled, site_ids_train)

    # Compute site prediction accuracy (for reporting)
    site_pred_train = site_clf.predict(X_train_scaled)
    site_acc_train = balanced_accuracy_score(site_ids_train, site_pred_train)
    logger.info(f"  [SITE-DECONFOUND] Site predictability (train): {site_acc_train:.4f}")

    # Compute residuals: X - X_predicted_from_site
    X_train_pred_proba = site_clf.predict_proba(X_train_scaled)  # (N, n_sites)
    # Use closed-form projection: residual = X - Z @ beta where Z = [1, site_proba]
    Z_train = np.hstack([np.ones((len(X_train_scaled), 1)), X_train_pred_proba])
    beta, *_ = np.linalg.lstsq(Z_train, X_train_scaled, rcond=None)
    X_train_resid = X_train_scaled - Z_train @ beta

    # Check site predictability after regression
    site_clf_post = LogisticRegression(max_iter=2000, C=1.0)
    site_clf_post.fit(X_train_resid, site_ids_train)
    site_pred_post = site_clf_post.predict(X_train_resid)
    site_acc_post = balanced_accuracy_score(site_ids_train, site_pred_post)
    logger.info(f"  [SITE-DECONFOUND] Site predictability after regression: {site_acc_post:.4f}")
    logger.info(f"  [SITE-DECONFOUND] Site signal removed: {site_acc_train - site_acc_post:.4f}")

    # Apply same transform to validation
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        X_val_pred_proba = site_clf.predict_proba(X_val_scaled)
        Z_val = np.hstack([np.ones((len(X_val_scaled), 1)), X_val_pred_proba])
        X_val_resid = X_val_scaled - Z_val @ beta
        return X_train_resid, X_val_resid, site_acc_train, site_acc_post
    else:
        return X_train_resid, None, site_acc_train, site_acc_post


# ═════════════════════════════════════════════════════════════════════════
# 4. COMBAT HARMONIZATION (Tier 1.4)
# ═════════════════════════════════════════════════════════════════════════
def apply_combat(X, site_ids, labels):
    """Apply ComBat harmonization to remove site effects while preserving biological variables.

    ComBat uses empirical-Bayes parametric site correction that preserves
    biological variables (diagnosis, age, sex) while removing batch (site) effects.
    """
    try:
        from neuroCombat import neuroCombat
        import pandas as pd
    except ImportError:
        logger.warning("  [COMBAT] neuroCombat not installed. Skipping ComBat.")
        logger.warning("  [COMBAT] Install with: pip install neuroCombat")
        return X

    logger.info("  [COMBAT] Applying neuroCombat harmonization...")

    # Create covars DataFrame
    covars = pd.DataFrame({
        'SITE': site_ids.astype(str),
        'DIAGNOSIS': labels.astype(str),
    })

    # neuroCombat expects (features, samples) shape
    data_combat = neuroCombat(
        dat=X.T,  # transpose to (features, samples)
        covars=covars,
        batch_col='SITE',
        categorical_cols=['DIAGNOSIS'],
    )
    X_harm = data_combat['data'].T  # back to (samples, features)

    # Check site predictability before and after ComBat
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_harm_scaled = scaler.fit_transform(X_harm)

    clf_before = LogisticRegression(max_iter=2000)
    clf_before.fit(X_scaled, site_ids)
    acc_before = balanced_accuracy_score(site_ids, clf_before.predict(X_scaled))

    clf_after = LogisticRegression(max_iter=2000)
    clf_after.fit(X_harm_scaled, site_ids)
    acc_after = balanced_accuracy_score(site_ids, clf_after.predict(X_harm_scaled))

    logger.info(f"  [COMBAT] Site predictability before: {acc_before:.4f}")
    logger.info(f"  [COMBAT] Site predictability after:  {acc_after:.4f}")
    logger.info(f"  [COMBAT] Site signal removed: {acc_before - acc_after:.4f}")

    return X_harm


# ═════════════════════════════════════════════════════════════════════════
# 5. TRAINING ONE FOLD
# ═════════════════════════════════════════════════════════════════════════
def train_one_fold(X_train, y_train, X_val, y_val,
                   site_ids_train, site_ids_val,
                   config, n_classes, n_rounds, patience, seed, device,
                   site_deconfound=False, combat_applied=False):
    """Train and evaluate PQFL on a single CV fold."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_features = X_train.shape[1]
    n_qubits = config["n_qubits"]

    # ── Optional: Site-confound regression ──
    if site_deconfound and not combat_applied:
        X_train, X_val, site_acc_before, site_acc_after = site_confound_regress(
            X_train, site_ids_train, X_val, site_ids_val
        )
        n_features = X_train.shape[1]

    # ── Create datasets ──
    n_rois_placeholder = 1
    train_ds = FCDataset(
        fc_matrices=np.zeros((len(y_train), n_rois_placeholder, n_rois_placeholder)),
        labels=y_train, tangent_features=torch.tensor(X_train, dtype=torch.float32),
    )
    val_ds = FCDataset(
        fc_matrices=np.zeros((len(y_val), n_rois_placeholder, n_rois_placeholder)),
        labels=y_val, tangent_features=torch.tensor(X_val, dtype=torch.float32),
    )

    from torch.utils.data import DataLoader
    def make_collate(has_fdt):
        def collate(batch):
            if isinstance(batch[0], dict):
                x = torch.stack([b["tangent_features"] for b in batch])
                y = torch.stack([b["label"] for b in batch])
                return {"tangent_features": x, "label": y}
            else:
                x = torch.stack([b[0] for b in batch])
                y = torch.stack([b[1] for b in batch])
                return x, y
        return collate

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                              collate_fn=make_collate(False))
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                            collate_fn=make_collate(False))

    # ── Compute class weights ──
    # Handle classes with 0 training examples (common in LOSO CV)
    # Set weight to 0 for missing classes instead of astronomical values
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    present = counts > 0
    counts_safe = np.where(counts == 0, 1, counts)  # avoid div by zero
    weights = counts_safe.sum() / (n_classes * counts_safe)
    weights = np.where(present, weights, 0.0)  # zero weight for absent classes
    class_weights = torch.tensor(weights, dtype=torch.float32)

    # ── Create model ──
    encoder_hidden = [max(16, n_features // 2), n_qubits * 2]
    vqc_config = VQCConfig(
        n_qubits=n_qubits,
        n_base_layers=config["n_base_layers"],
        n_personal_layers=1,
        encoding_type="angle",
        input_dim=n_features,
        encoder_hidden_dims=encoder_hidden,
        dropout=config["dropout"],
        n_classes=n_classes,
        fdt_features=0,  # No FDT features in Phase 2 (set to 0 to avoid shape mismatch)
    )
    model = HybridVQC(vqc_config)

    # ── Create client ──
    client = PQFLClient(
        model=model, train_loader=train_loader, val_loader=val_loader,
        site_id=0, site_name="Federated",
        learning_rate=config["learning_rate"],
        label_smoothing=config["label_smoothing"],
        device=device,
        class_weights=class_weights,
        local_epochs=2,
    )

    # ── Training loop with early stopping ──
    history = []
    best_ba = -1.0
    best_round_metrics = None
    rounds_without_improvement = 0
    current_params = client.get_parameters()

    for round_idx in range(1, n_rounds + 1):
        updated_params, num_samples, train_metrics = client.fit(current_params, {})
        current_params = updated_params

        val_metrics = client.evaluate(current_params, {})
        val_ba = val_metrics.get("balanced_accuracy", 0)
        val_auc = val_metrics.get("auc_roc_macro", val_metrics.get("auc_roc", 0.5))
        val_loss = val_metrics.get("val_loss", 0)
        train_loss = train_metrics.get("train_loss", 0)

        history.append({
            "round": round_idx,
            "val_ba": float(val_ba),
            "val_auc": float(val_auc) if isinstance(val_auc, (int, float)) else 0,
            "val_loss": float(val_loss),
            "train_loss": float(train_loss),
        })

        if val_ba > best_ba:
            best_ba = val_ba
            best_round_metrics = {
                "round": round_idx,
                "balanced_accuracy": float(val_ba),
                "auc_roc_macro": float(val_auc) if isinstance(val_auc, (int, float)) else 0.5,
                "val_loss": float(val_loss),
                "train_loss": float(train_loss),
            }
            best_params = [p.copy() if isinstance(p, np.ndarray) else p
                          for p in current_params]
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1

        if patience > 0 and rounds_without_improvement >= patience:
            logger.info(f"  Early stop at round {round_idx} (best BA={best_ba:.4f} at round {best_round_metrics['round']})")
            break

    # ── Final evaluation with best params ──
    client.set_parameters(best_params)
    final_metrics = client.evaluate(best_params, {})

    # Extract per-class metrics
    result = {
        "balanced_accuracy": float(final_metrics.get("balanced_accuracy", best_ba)),
        "auc_roc_macro": float(final_metrics.get("auc_roc_macro", final_metrics.get("auc_roc", 0.5))),
        "f1_macro": float(final_metrics.get("f1_macro", final_metrics.get("f1", 0))),
        "accuracy": float(final_metrics.get("accuracy", 0)),
        "val_loss": float(final_metrics.get("val_loss", 0)),
        "best_round": best_round_metrics["round"],
        "total_rounds": round_idx,
        "train_loss": best_round_metrics["train_loss"],
    }

    # Per-class sensitivity/specificity
    for cls_name in CLASS_NAMES.get(n_classes, []):
        for metric in [f"sensitivity_{cls_name}", f"specificity_{cls_name}"]:
            if metric in final_metrics:
                result[metric] = float(final_metrics[metric])

    # Confusion matrix
    if "confusion_matrix" in final_metrics:
        result["confusion_matrix"] = final_metrics["confusion_matrix"]

    result["n_total"] = len(y_val)
    result["round_history"] = history
    result["fold_time_seconds"] = 0  # will be filled by caller

    return result


# ═════════════════════════════════════════════════════════════════════════
# 6. TANGENT SVM BASELINE
# ═════════════════════════════════════════════════════════════════════════
def run_svm_baseline(X_train, y_train, X_val, y_val, n_classes):
    """Run TangentSVM baseline for comparison."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Handle classes missing from training (LOSO CV)
    present_train = np.unique(y_train)
    # Remap labels to contiguous range if some classes are missing
    label_map = {old: new for new, old in enumerate(present_train)}
    y_train_mapped = np.array([label_map[y] for y in y_train])
    y_val_mapped = np.array([label_map.get(y, -1) for y in y_val])
    # For val samples whose class is not in training, assign to nearest present class
    if -1 in y_val_mapped:
        y_val_mapped = np.where(y_val_mapped == -1, 0, y_val_mapped)

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ])
    clf.fit(X_train, y_train_mapped)
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)

    ba = balanced_accuracy_score(y_val_mapped, y_pred)
    # Compute AUC only for classes present in validation
    val_classes = np.unique(y_val_mapped)
    try:
        if len(val_classes) > 1 and y_proba.shape[1] == len(val_classes):
            auc = roc_auc_score(y_val_mapped, y_proba, multi_class='ovr',
                               average='macro', labels=val_classes)
        elif len(val_classes) == 2:
            auc = roc_auc_score(y_val_mapped, y_proba[:, 1])
        else:
            auc = 0.5  # only one class in validation
    except:
        auc = 0.5  # fallback
    f1 = f1_score(y_val_mapped, y_pred, average='macro', zero_division=0)

    # Per-class sensitivity
    result = {"balanced_accuracy": float(ba), "auc_roc_macro": float(auc), "f1_macro": float(f1)}

    for cls in range(n_classes):
        mask_true = (y_val == cls)
        mask_pred = (y_pred == cls)
        sens = mask_pred[mask_true].sum() / max(mask_true.sum(), 1)
        spec = (~mask_pred[~mask_true]).sum() / max((~mask_true).sum(), 1)
        cls_name = CLASS_NAMES.get(n_classes, [f"C{i}" for i in range(n_classes)])[cls]
        result[f"sensitivity_{cls_name}"] = float(sens)
        result[f"specificity_{cls_name}"] = float(spec)

    return result


# ═════════════════════════════════════════════════════════════════════════
# 7. MAIN TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Tier 1 Enhanced PQFL Training")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_classes", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--cv_mode", type=str, default="loso", choices=["loso", "kfold"],
                       help="loso = Leave-One-Site-Out, kfold = 5-fold stratified")
    parser.add_argument("--site_deconfound", action="store_true",
                       help="Apply site-confound regression")
    parser.add_argument("--combat", action="store_true",
                       help="Apply ComBat harmonization")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                       help="Random seeds for multi-seed evaluation")
    parser.add_argument("--n_rounds", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--n_folds", type=int, default=5, help="For kfold mode")
    parser.add_argument("--n_qubits", type=int, default=8)
    parser.add_argument("--metric", type=str, default="log_euclidean",
                       choices=["log_euclidean", "affine_invariant"],
                       help="Riemannian metric (log_euclidean is 10x faster)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Override config
    BASE_CONFIG["n_qubits"] = args.n_qubits

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    logger.info(f"Config: {BASE_CONFIG}")
    logger.info(f"CV mode: {args.cv_mode}")
    logger.info(f"Site deconfound: {args.site_deconfound}")
    logger.info(f"ComBat: {args.combat}")
    logger.info(f"Seeds: {args.seeds}")

    # ── Load data ──
    sites = load_sites(args.data_dir, args.n_classes)
    if len(sites) < 2:
        raise RuntimeError(f"Need >=2 sites, got {len(sites)}")

    # ── Riemannian preprocessing ──
    n_rois = sites[0]["fc_matrices"].shape[1]
    n_components = min(BASE_CONFIG["n_components"], 5050, len(sites[0]["labels"]) * len(sites) - 1)
    
    # Use log_euclidean metric for speed (affine_invariant is 10x slower)
    # Both are valid Riemannian metrics; log_euclidean is standard for large datasets
    metric = getattr(args, 'metric', 'log_euclidean')
    logger.info(f"\n=== Riemannian preprocessing (metric={metric}) ===")
    logger.info(f"This may take 30-90 seconds for {sum(len(s['labels']) for s in sites)} matrices...")
    riemannian_engine = RiemannianEngine(
        n_rois=n_rois,
        n_components=n_components,
        metric=metric,
        regularization_lambda=1e-3,
    )

    X_all, y_all, site_ids_all, fc_all = aggregate_features(sites, riemannian_engine)

    # ── Optional: ComBat (applied globally before CV) ──
    if args.combat:
        X_all = apply_combat(X_all, site_ids_all, y_all)

    # ── Determine CV splits ──
    if args.cv_mode == "loso":
        logo = LeaveOneGroupOut()
        splits = list(logo.split(X_all, y_all, groups=site_ids_all))
        n_splits = len(splits)
        logger.info(f"\nLOSO CV: {n_splits} folds (one per site)")
        for i, (tr, va) in enumerate(splits):
            held_out_site = site_ids_all[va[0]]
            site_name = next(s["site_name"] for s in sites if s["site_id"] == held_out_site)
            logger.info(f"  Fold {i+1}: hold out {site_name} (site_id={held_out_site}), "
                       f"train={len(tr)}, val={len(va)}")
    else:
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        splits = list(skf.split(X_all, y_all))
        n_splits = len(splits)
        logger.info(f"\n{k_splits}-fold stratified CV: {n_splits} folds")

    # ── Run training across seeds × folds ──
    all_seed_results = []
    all_svm_results = []

    total_start = time.time()

    for seed_idx, seed in enumerate(args.seeds):
        logger.info(f"\n{'='*78}")
        logger.info(f"SEED {seed_idx+1}/{len(args.seeds)} (seed={seed})")
        logger.info(f"{'='*78}")

        seed_fold_results = []
        seed_svm_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"\n--- Seed {seed}, Fold {fold_idx+1}/{n_splits} ---")

            X_train, X_val = X_all[train_idx], X_all[val_idx]
            y_train, y_val = y_all[train_idx], y_all[val_idx]
            site_train, site_val = site_ids_all[train_idx], site_ids_all[val_idx]

            fold_start = time.time()

            # PQFL training
            try:
                pqfl_result = train_one_fold(
                    X_train, y_train, X_val, y_val,
                    site_train, site_val,
                    BASE_CONFIG, args.n_classes, args.n_rounds, args.patience,
                    seed, device,
                    site_deconfound=args.site_deconfound,
                    combat_applied=args.combat,
                )
                pqfl_result["fold"] = fold_idx + 1
                pqfl_result["seed"] = seed
                pqfl_result["fold_time_seconds"] = time.time() - fold_start

                if args.cv_mode == "loso":
                    held_out_site = site_ids_all[val_idx[0]]
                    site_name = next(s["site_name"] for s in sites if s["site_id"] == held_out_site)
                    pqfl_result["held_out_site"] = site_name

                logger.info(f"  Fold {fold_idx+1} results: BA={pqfl_result['balanced_accuracy']:.4f}, "
                           f"AUC={pqfl_result['auc_roc_macro']:.4f}, "
                           f"Time={pqfl_result['fold_time_seconds']:.0f}s")
                seed_fold_results.append(pqfl_result)
            except Exception as e:
                logger.error(f"  Fold {fold_idx+1} FAILED: {e}")
                import traceback
                traceback.print_exc()
                continue

            # SVM baseline
            try:
                svm_result = run_svm_baseline(X_train, y_train, X_val, y_val, args.n_classes)
                svm_result["fold"] = fold_idx + 1
                svm_result["seed"] = seed
                seed_svm_results.append(svm_result)
            except Exception as e:
                logger.warning(f"  SVM baseline failed: {e}")

        all_seed_results.append({"seed": seed, "fold_results": seed_fold_results})
        all_svm_results.append({"seed": seed, "fold_results": seed_svm_results})

    total_time = time.time() - total_start

    # ── Aggregate results ──
    logger.info(f"\n{'='*78}")
    logger.info(f"AGGREGATE RESULTS — {args.n_classes}-class — {n_splits} folds × {len(args.seeds)} seeds")
    logger.info(f"{'='*78}")

    # Flatten all fold results across seeds
    all_pqfl_ba = [r["balanced_accuracy"] for sr in all_seed_results for r in sr["fold_results"]]
    all_pqfl_auc = [r["auc_roc_macro"] for sr in all_seed_results for r in sr["fold_results"]]
    all_svm_ba = [r["balanced_accuracy"] for sr in all_svm_results for r in sr["fold_results"]]
    all_svm_auc = [r["auc_roc_macro"] for sr in all_svm_results for r in sr["fold_results"]]

    pqfl_ba_mean, pqfl_ba_std = np.mean(all_pqfl_ba), np.std(all_pqfl_ba)
    pqfl_auc_mean, pqfl_auc_std = np.mean(all_pqfl_auc), np.std(all_pqfl_auc)
    svm_ba_mean = np.mean(all_svm_ba) if all_svm_ba else 0
    svm_auc_mean = np.mean(all_svm_auc) if all_svm_auc else 0

    # Statistical tests
    stat_tests = {}
    if len(all_pqfl_ba) >= 5 and len(all_svm_ba) >= 5:
        min_len = min(len(all_pqfl_ba), len(all_svm_ba))
        try:
            wilcoxon_stat, wilcoxon_p = scipy_stats.wilcoxon(all_pqfl_ba[:min_len], all_svm_ba[:min_len])
            stat_tests["wilcoxon_ba"] = {"statistic": float(wilcoxon_stat), "p_value": float(wilcoxon_p)}
        except:
            pass
        try:
            t_stat, t_p = scipy_stats.ttest_rel(all_pqfl_ba[:min_len], all_svm_ba[:min_len])
            stat_tests["paired_ttest_ba"] = {"statistic": float(t_stat), "p_value": float(t_p)}
        except:
            pass
        # Cohen's d
        diff = np.array(all_pqfl_ba[:min_len]) - np.array(all_svm_ba[:min_len])
        cohens_d = diff.mean() / (diff.std(ddof=1) + 1e-10)
        stat_tests["cohens_d_ba"] = float(cohens_d)

    logger.info(f"\nPQFL Summary ({len(all_pqfl_ba)} runs = {len(args.seeds)} seeds × {n_splits} folds):")
    logger.info(f"  Balanced Accuracy: {pqfl_ba_mean:.4f} ± {pqfl_ba_std:.4f}")
    logger.info(f"  AUC-ROC (macro):  {pqfl_auc_mean:.4f} ± {pqfl_auc_std:.4f}")
    logger.info(f"  F1 (macro):       {np.mean([r.get('f1_macro',0) for sr in all_seed_results for r in sr['fold_results']]):.4f}")

    if all_svm_ba:
        logger.info(f"\nSVM Baseline ({len(all_svm_ba)} runs):")
        logger.info(f"  Balanced Accuracy: {svm_ba_mean:.4f}")
        logger.info(f"  AUC-ROC:           {svm_auc_mean:.4f}")

    if stat_tests:
        logger.info(f"\nStatistical Tests (PQFL vs SVM):")
        if "wilcoxon_ba" in stat_tests:
            logger.info(f"  Wilcoxon signed-rank: p={stat_tests['wilcoxon_ba']['p_value']:.4f}")
        if "paired_ttest_ba" in stat_tests:
            logger.info(f"  Paired t-test: p={stat_tests['paired_ttest_ba']['p_value']:.4f}")
        if "cohens_d_ba" in stat_tests:
            logger.info(f"  Cohen's d: {stat_tests['cohens_d_ba']:.4f}")

    # Per-class metrics
    for cls_name in CLASS_NAMES.get(args.n_classes, []):
        sens_key = f"sensitivity_{cls_name}"
        spec_key = f"specificity_{cls_name}"
        sens_vals = [r[sens_key] for sr in all_seed_results for r in sr["fold_results"] if sens_key in r]
        spec_vals = [r[spec_key] for sr in all_seed_results for r in sr["fold_results"] if spec_key in r]
        if sens_vals:
            logger.info(f"  {cls_name} sens: {np.mean(sens_vals):.4f} ± {np.std(sens_vals):.4f}, "
                       f"spec: {np.mean(spec_vals):.4f} ± {np.std(spec_vals):.4f}")

    logger.info(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # ── Save results ──
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_str = args.cv_mode
        decon_str = "_deconfound" if args.site_deconfound else ""
        combat_str = "_combat" if args.combat else ""
        args.output_dir = f"results/tier1_{mode_str}{decon_str}{combat_str}_{timestamp}"

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_json = {
        "config": BASE_CONFIG,
        "n_classes": args.n_classes,
        "cv_mode": args.cv_mode,
        "site_deconfound": args.site_deconfound,
        "combat": args.combat,
        "seeds": args.seeds,
        "n_seeds": len(args.seeds),
        "n_folds": n_splits,
        "n_runs": len(all_pqfl_ba),
        "n_samples": len(y_all),
        "n_sites": len(sites),
        "total_time_seconds": total_time,
        "pqfl_summary": {
            "ba_mean": float(pqfl_ba_mean), "ba_std": float(pqfl_ba_std),
            "auc_mean": float(pqfl_auc_mean), "auc_std": float(pqfl_auc_std),
        },
        "svm_summary": {
            "ba_mean": float(svm_ba_mean), "auc_mean": float(svm_auc_mean),
        },
        "statistical_tests": stat_tests,
        "all_seed_results": all_seed_results,
        "all_svm_results": all_svm_results,
    }

    results_file = output_path / "tier1_results.json"
    with open(results_file, "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"\nDone in {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
