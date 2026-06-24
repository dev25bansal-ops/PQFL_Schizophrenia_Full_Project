#!/usr/bin/env python3
"""PQFL Phase 2: Multi-class federated training on the expanded 16-site federation.

This script extends the June 8 final_training.py to support:
  - 3-class classification (SZ / HC / Other) — primary experiment
  - 4-class classification (SZ / HC / Other / BP) — secondary experiment
  - Class-weighted loss to handle imbalanced multi-class distributions
  - New datasets: BrainLat (9 sites), Transdiagnostic, Depression, Kaggle Psychosis, MLSP
  - Backward-compatible with the original 8 datasets (COBRE, LA5c, TCP2025, etc.)

Usage:
    # Primary 3-class experiment (recommended first run)
    python experiments/train_pqfl_phase2.py \\
        --data_dir F:\\PQFL_Schizophrenia_Full_Project\\pqfl_project\\data\\processed \\
        --n_classes 3 \\
        --n_rounds 50 \\
        --patience 10

    # Secondary 4-class experiment (after 3-class works)
    python experiments/train_pqfl_phase2.py \\
        --data_dir F:\\PQFL_Schizophrenia_Full_Project\\pqfl_project\\data\\processed \\
        --n_classes 4 \\
        --n_rounds 50 \\
        --patience 10

    # Quick smoke test (5 rounds, no early stopping)
    python experiments/train_pqfl_phase2.py --data_dir ... --n_classes 3 --n_rounds 5 --patience 0 --quick

Output:
    results/<timestamp>_phase2_<n_classes>class/
        ├── phase2_results.json    (full per-fold metrics + summary)
        ├── training_curves.png    (per-fold BA/AUC over rounds)
        ├── confusion_matrix.png   (3×3 or 4×4 normalized CM)
        └── site_summary.json      (per-site subject counts + label dist)
"""

import argparse
import sys
import os
import json
import copy
import time
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, confusion_matrix,
    f1_score, accuracy_score,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pqfl.data.dataset import FCDataset
from pqfl.riemannian.engine import RiemannianEngine
from pqfl.quantum.vqc import HybridVQC, VQCConfig
from pqfl.federated.client import PQFLClient
from pqfl.evaluation.metrics import compute_classification_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Optimal config (from June 5 sweep + multi-class extensions)
# ─────────────────────────────────────────────────────────────────────────────
BASE_CONFIG = {
    "n_qubits": 6,           # Sweep-optimal
    "n_components": 71,      # Sweep-optimal (auto-reduced if data has fewer features)
    "learning_rate": 0.0005,
    "dropout": 0.5,
    "n_base_layers": 2,
    "label_smoothing": 0.1,
    "batch_size": 16,
}

# Class names per scheme (for metric reporting + figures)
CLASS_NAMES = {
    2: ["HC", "SZ"],
    3: ["SZ", "HC", "Other"],
    4: ["SZ", "HC", "Other", "BP"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="PQFL Phase 2 multi-class training")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing *_processed.npz files")
    parser.add_argument("--n_classes", type=int, default=3, choices=[2, 3, 4],
                       help="Number of output classes (3 = primary, 4 = secondary)")
    parser.add_argument("--n_rounds", type=int, default=50,
                       help="Max federated rounds per fold")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience (0 = disabled)")
    parser.add_argument("--n_folds", type=int, default=5,
                       help="Number of stratified CV folds")
    parser.add_argument("--n_qubits", type=int, default=None,
                       help="Override n_qubits (default 6)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output dir (default: results/<timestamp>_phase2_<n>c)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: 5 rounds, no early stopping, 3 folds")
    parser.add_argument("--device", type=str, default=None,
                       help="Device: cuda, cpu, or auto (default)")
    return parser.parse_args()


def discover_processed_sites(data_dir: Path) -> list:
    """Find all *_processed.npz files in data_dir."""
    files = sorted(data_dir.glob("*_processed.npz"))
    if not files:
        raise FileNotFoundError(f"No *_processed.npz files in {data_dir}")
    return files


def load_all_sites(npz_files: list, n_classes: int) -> list:
    """Load all site .npz files into a list of dicts.

    Filters out sites that have no subjects matching the n_classes scheme.
    """
    sites = []
    total_n = 0
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        fc = data["fc_matrices"]
        labels = data["labels"].astype(np.int64)
        site_id = int(data["site_id"])
        site_name = str(data["site_name"])

        # Filter labels for the chosen scheme
        valid_labels = set(range(n_classes))
        mask = np.array([l in valid_labels for l in labels])
        if not mask.any():
            logger.info(f"  [SKIP] {site_name}: no subjects in {n_classes}-class scheme")
            continue

        fc = fc[mask]
        labels = labels[mask]
        subject_ids = data["subject_ids"][mask] if "subject_ids" in data else None

        # Skip sites with < 5 subjects or only 1 class
        if len(labels) < 5:
            logger.info(f"  [SKIP] {site_name}: only {len(labels)} subjects in scheme")
            continue
        if len(np.unique(labels)) < 2 and n_classes > 2:
            logger.info(f"  [SKIP] {site_name}: only 1 class present in scheme "
                       f"({dict(Counter(labels))})")
            continue

        sites.append({
            "fc_matrices": fc,
            "labels": labels,
            "site_id": site_id,
            "site_name": site_name,
            "subject_ids": subject_ids,
            "npz_path": str(npz_path),
        })
        total_n += len(labels)
        logger.info(f"  [LOAD] {site_name}: {len(labels)} subjects, "
                   f"labels={dict(Counter(labels))}")

    logger.info(f"Loaded {len(sites)} sites, {total_n} total subjects")
    return sites


def aggregate_features(sites: list, riemannian_engine) -> tuple:
    """Aggregate FC matrices across all sites and project to tangent space.

    Returns:
        X_all: (n_total, n_features) tangent features
        y_all: (n_total,) labels
        site_ids_all: (n_total,) site indices
        fdt_all: None (FDT extraction not in base RiemannianEngine API)
    """
    all_fc = []
    all_labels = []
    all_site_ids = []

    for site_idx, site in enumerate(sites):
        all_fc.append(site["fc_matrices"])
        all_labels.append(site["labels"])
        all_site_ids.append(np.full(len(site["labels"]), site_idx))

    all_fc = np.concatenate(all_fc, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_site_ids = np.concatenate(all_site_ids, axis=0)

    logger.info(f"Aggregated: {all_fc.shape[0]} samples, {all_fc.shape[1]} ROIs")
    logger.info(f"Label distribution: {dict(Counter(all_labels))}")

    # Riemannian preprocessing: SPD regularization + Fréchet mean + tangent projection
    # Returns tangent features (n_samples, n_components)
    X_all = riemannian_engine.fit_transform(all_fc)
    if isinstance(X_all, torch.Tensor):
        X_all = X_all.cpu().numpy()
    # Ensure 2D
    if X_all.ndim == 1:
        X_all = X_all.reshape(1, -1)

    # FDT features are not exposed by the base RiemannianEngine; return None
    # (the VQC handles None gracefully via fdt_dim=0)
    fdt_all = None

    logger.info(f"Tangent features: {X_all.shape}")
    return X_all, all_labels, all_site_ids, fdt_all


def compute_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for balanced training."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1, counts)  # avoid div by zero
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_fold(
    X_train, y_train, X_val, y_val,
    fdt_train, fdt_val,
    site_ids_train, site_ids_val,
    config, n_classes, n_rounds, patience, seed, device,
):
    """Train and evaluate PQFL on a single CV fold.

    Uses PQFLClient's fit() + evaluate() directly with simple early stopping.
    """
    n_features = X_train.shape[1]
    fdt_dim = fdt_train.shape[1] if fdt_train is not None and fdt_train.shape[1] > 0 else 0

    # Limit n_components to available features
    logger.info(f"  Fold: n_features={n_features}, fdt_dim={fdt_dim}, n_classes={n_classes}")

    # Create datasets (FC matrices are dummy since we use tangent features directly)
    n_rois_placeholder = 1
    train_ds = FCDataset(
        fc_matrices=np.zeros((len(y_train), n_rois_placeholder, n_rois_placeholder)),
        labels=y_train, tangent_features=X_train, fdt_features=fdt_train,
    )
    val_ds = FCDataset(
        fc_matrices=np.zeros((len(y_val), n_rois_placeholder, n_rois_placeholder)),
        labels=y_val, tangent_features=X_val, fdt_features=fdt_val,
    )

    # Create model with n_classes
    encoder_hidden = [max(16, n_features // 2), config["n_qubits"] * 2]
    vqc_config = VQCConfig(
        n_qubits=config["n_qubits"],
        n_base_layers=config["n_base_layers"],
        n_personal_layers=1,
        encoding_type="angle",
        entanglement="functional",
        input_dim=n_features,
        encoder_hidden_dims=encoder_hidden,
        fdt_features=fdt_dim,
        classifier_hidden_dims=[16],
        dropout=config["dropout"],
        use_dual_register=False,
        n_classes=n_classes,  # NEW: multi-class output
    )

    model = HybridVQC(vqc_config)

    # Compute class weights for this fold's training set
    class_weights = compute_class_weights(y_train, n_classes).to(device)
    logger.info(f"  Class weights: {class_weights.tolist()}")

    # Dataloaders
    has_fdt = fdt_train is not None and fdt_train.shape[1] > 0

    def make_collate(include_fdt):
        def collate_fn(batch):
            result = {
                "tangent_features": torch.stack([b["tangent_features"] for b in batch]),
                "label": torch.stack([b["label"] for b in batch]),
            }
            if include_fdt and "fdt_features" in batch[0]:
                result["fdt_features"] = torch.stack([b["fdt_features"] for b in batch])
            return result
        return collate_fn

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        collate_fn=make_collate(has_fdt),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        collate_fn=make_collate(has_fdt),
    )

    # Create client with class-weighted loss
    # PATCHED v2: pass site_id and site_name so logs show "Federated" instead of "Unknown"
    client = PQFLClient(
        model=model, train_loader=train_loader, val_loader=val_loader,
        site_id=0,
        site_name="Federated",
        learning_rate=config["learning_rate"],
        label_smoothing=config["label_smoothing"],
        device=device,
        class_weights=class_weights,  # NEW: weighted CE loss
        local_epochs=2,  # match sweep-optimal from June 5
    )

    # Training loop with early stopping
    history = []
    best_ba = -1.0
    best_round_metrics = None
    rounds_without_improvement = 0

    # Get initial shared parameters (FedPer: only shared params are exchanged)
    current_params = client.get_parameters()

    for round_idx in range(1, n_rounds + 1):
        # Local training step
        updated_params, n_samples, train_metrics = client.fit(
            current_params,
            config={"learning_rate": config["learning_rate"], "local_epochs": 2},
        )

        # Update shared parameters
        current_params = updated_params

        # Evaluate
        eval_metrics = client.evaluate(current_params)
        eval_metrics["round"] = round_idx
        eval_metrics["train_loss"] = train_metrics.get("train_loss", 0.0)

        # Rename for clarity in history
        round_record = {
            "round": round_idx,
            "val_ba": eval_metrics.get("balanced_accuracy", 0.0),
            "val_auc": eval_metrics.get("auc_roc_macro",
                                       eval_metrics.get("auc_roc", 0.0)),
            "val_loss": eval_metrics.get("val_loss", 0.0),
            "train_loss": eval_metrics.get("train_loss", 0.0),
            "val_acc": eval_metrics.get("accuracy", 0.0),
        }
        history.append(round_record)

        current_ba = round_record["val_ba"]
        if current_ba > best_ba:
            best_ba = current_ba
            best_round_metrics = eval_metrics
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1

        # Early stopping
        if patience > 0 and rounds_without_improvement >= patience:
            logger.info(f"  Early stop at round {round_idx} (best BA={best_ba:.4f} "
                       f"at round {round_idx - rounds_without_improvement})")
            break

        if round_idx % 5 == 0 or round_idx == 1:
            logger.info(f"  Round {round_idx}: BA={current_ba:.4f}, "
                       f"AUC={round_record['val_auc']:.4f}, "
                       f"loss={round_record['val_loss']:.4f}")

    # Use best round's metrics as final result for this fold
    if best_round_metrics is None:
        best_round_metrics = eval_metrics

    # Build final metrics dict in standard format
    final_metrics = compute_classification_metrics(
        y_val,
        np.array(best_round_metrics.get("y_pred", [])),
        best_round_metrics.get("y_prob"),
        n_classes=n_classes,
        class_names=CLASS_NAMES[n_classes],
    ) if "y_pred" in best_round_metrics else dict(best_round_metrics)

    # Fallback: use the eval metrics directly (they're already in the right format
    # from compute_classification_metrics inside client.evaluate)
    if "balanced_accuracy" not in final_metrics:
        final_metrics = dict(best_round_metrics)

    final_metrics["val_loss"] = history[-1].get("val_loss", 0.0) if history else 0.0
    final_metrics["best_round"] = int(np.argmax([h["val_ba"] for h in history]) + 1) if history else 0
    final_metrics["total_rounds"] = len(history)
    final_metrics["round_history"] = history

    return final_metrics


def run_baselines(X_train, y_train, X_val, y_val, n_classes, seed=42):
    """Run classical baselines (Tangent SVM + Riemannian LR) for comparison."""
    baselines = {}

    # Sklearn doesn't have native Riemannian LR, but TangentSpaceSVM does the job
    try:
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        # SVM with RBF kernel + class weights
        svm = SVC(kernel="rbf", probability=True, random_state=seed,
                  class_weight="balanced", C=1.0)
        svm.fit(X_train_s, y_train)
        y_pred = svm.predict(X_val_s)
        y_prob = svm.predict_proba(X_val_s)

        baselines["TangentSVM"] = compute_classification_metrics(
            y_val, y_pred, y_prob, n_classes=n_classes,
            class_names=CLASS_NAMES[n_classes],
        )

        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=seed,
                                class_weight="balanced", multi_class="multinomial")
        lr.fit(X_train_s, y_train)
        y_pred = lr.predict(X_val_s)
        y_prob = lr.predict_proba(X_val_s)

        baselines["LogisticRegression"] = compute_classification_metrics(
            y_val, y_pred, y_prob, n_classes=n_classes,
            class_names=CLASS_NAMES[n_classes],
        )

    except Exception as e:
        logger.warning(f"Baseline computation failed: {e}")

    return baselines


def main():
    args = parse_args()

    # Quick mode overrides
    if args.quick:
        args.n_rounds = 5
        args.patience = 0
        args.n_folds = 3

    if args.n_qubits is not None:
        BASE_CONFIG["n_qubits"] = args.n_qubits

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(f"results/{timestamp}_phase2_{args.n_classes}class")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover and load sites
    data_dir = Path(args.data_dir)
    npz_files = discover_processed_sites(data_dir)
    logger.info(f"\nFound {len(npz_files)} processed site files:")
    for f in npz_files:
        logger.info(f"  {f.name}")

    logger.info(f"\nLoading sites for {args.n_classes}-class scheme...")
    sites = load_all_sites(npz_files, args.n_classes)
    if len(sites) < 2:
        raise RuntimeError(f"Need ≥2 sites, got {len(sites)}")

    # Save site summary
    site_summary = [{
        "site_name": s["site_name"],
        "site_id": s["site_id"],
        "n_subjects": len(s["labels"]),
        "label_distribution": {int(k): int(v) for k, v in Counter(s["labels"]).items()},
        "npz_path": s["npz_path"],
    } for s in sites]
    with open(out_dir / "site_summary.json", "w") as f:
        json.dump(site_summary, f, indent=2)

    # Riemannian preprocessing
    logger.info("\n=== Riemannian preprocessing ===")
    riemannian_engine = RiemannianEngine(
        metric="affine_invariant",
        regularization_lambda=1e-3,
        n_components=BASE_CONFIG["n_components"],
    )
    X_all, y_all, site_ids_all, fdt_all = aggregate_features(sites, riemannian_engine)

    # 5-fold stratified CV
    logger.info(f"\n=== {args.n_folds}-fold Stratified CV ===")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    fold_results = []
    baseline_results_per_fold = []

    total_start = time.time()
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx+1}/{args.n_folds}")
        logger.info(f"{'='*60}")
        logger.info(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
        logger.info(f"  Train labels: {dict(Counter(y_all[train_idx]))}")
        logger.info(f"  Val labels:   {dict(Counter(y_all[val_idx]))}")

        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]
        fdt_train = fdt_all[train_idx] if fdt_all is not None else None
        fdt_val = fdt_all[val_idx] if fdt_all is not None else None
        site_ids_train = site_ids_all[train_idx]
        site_ids_val = site_ids_all[val_idx]

        fold_start = time.time()

        # PQFL training
        fold_metrics = train_one_fold(
            X_train, y_train, X_val, y_val,
            fdt_train, fdt_val,
            site_ids_train, site_ids_val,
            BASE_CONFIG, args.n_classes, args.n_rounds, args.patience,
            args.seed, device,
        )
        fold_metrics["fold"] = fold_idx + 1
        fold_metrics["fold_time_seconds"] = time.time() - fold_start
        fold_results.append(fold_metrics)

        logger.info(f"\n  Fold {fold_idx+1} results:")
        logger.info(f"    BA:  {fold_metrics['balanced_accuracy']:.4f}")
        if "auc_roc_macro" in fold_metrics:
            logger.info(f"    AUC: {fold_metrics['auc_roc_macro']:.4f}")
        if "f1_macro" in fold_metrics:
            logger.info(f"    F1:  {fold_metrics['f1_macro']:.4f}")
        logger.info(f"    Time: {fold_metrics['fold_time_seconds']:.1f}s, "
                   f"best round: {fold_metrics['best_round']}")

        # Baselines (every fold for proper comparison)
        baselines = run_baselines(X_train, y_train, X_val, y_val,
                                  args.n_classes, args.seed)
        baseline_results_per_fold.append(baselines)

    total_time = time.time() - total_start

    # Aggregate across folds
    logger.info(f"\n{'='*60}")
    logger.info(f"AGGREGATE RESULTS — {args.n_classes}-class — {args.n_folds} folds")
    logger.info(f"{'='*60}")

    ba_scores = [f["balanced_accuracy"] for f in fold_results]
    summary = {
        "ba_mean": float(np.mean(ba_scores)),
        "ba_std": float(np.std(ba_scores)),
    }

    if "auc_roc_macro" in fold_results[0]:
        auc_scores = [f["auc_roc_macro"] for f in fold_results]
        summary["auc_mean"] = float(np.mean(auc_scores))
        summary["auc_std"] = float(np.std(auc_scores))

    if "f1_macro" in fold_results[0]:
        f1_scores = [f["f1_macro"] for f in fold_results]
        summary["f1_mean"] = float(np.mean(f1_scores))
        summary["f1_std"] = float(np.std(f1_scores))

    # Per-class sensitivity/specificity
    for cls_name in CLASS_NAMES[args.n_classes]:
        sens_key = f"sensitivity_{cls_name}"
        spec_key = f"specificity_{cls_name}"
        if sens_key in fold_results[0]:
            summary[f"sens_{cls_name}_mean"] = float(np.mean([f[sens_key] for f in fold_results]))
            summary[f"spec_{cls_name}_mean"] = float(np.mean([f[spec_key] for f in fold_results]))

    logger.info(f"\nPQFL Summary:")
    logger.info(f"  Balanced Accuracy: {summary['ba_mean']:.4f} ± {summary['ba_std']:.4f}")
    if "auc_mean" in summary:
        logger.info(f"  AUC-ROC (macro):  {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    if "f1_mean" in summary:
        logger.info(f"  F1 (macro):       {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")

    # Baseline aggregate
    baseline_summary = {}
    for model_name in baseline_results_per_fold[0].keys():
        bas = {k: [b[model_name][k] for b in baseline_results_per_fold
                    if k in b[model_name]]
               for k in baseline_results_per_fold[0][model_name]
               if isinstance(baseline_results_per_fold[0][model_name].get(k), (int, float))}
        baseline_summary[model_name] = {
            k: float(np.mean(v)) for k, v in bas.items() if v
        }

    logger.info(f"\nBaseline Summary:")
    for name, b in baseline_summary.items():
        logger.info(f"  {name}: BA={b.get('balanced_accuracy', 0):.4f}, "
                   f"AUC={b.get('auc_roc_macro', 0):.4f}")

    # Save final results
    final_results = {
        "config": BASE_CONFIG,
        "n_classes": args.n_classes,
        "class_names": CLASS_NAMES[args.n_classes],
        "n_folds": args.n_folds,
        "n_rounds": args.n_rounds,
        "patience": args.patience,
        "seed": args.seed,
        "total_time_seconds": total_time,
        "pqfl_fold_results": fold_results,
        "pqfl_summary": summary,
        "baseline_results": baseline_summary,
        "baseline_fold_results": baseline_results_per_fold,
        "n_samples": len(y_all),
        "n_sites": len(sites),
        "label_distribution": {int(k): int(v) for k, v in Counter(y_all).items()},
        "fold_histories": [f.get("round_history", []) for f in fold_results],
    }

    results_path = out_dir / "phase2_results.json"
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_path}")

    # Generate plots
    try:
        _plot_training_curves(fold_results, out_dir, args.n_classes)
        _plot_confusion_matrix(fold_results, out_dir, args.n_classes)
    except Exception as e:
        logger.warning(f"Plot generation failed: {e}")

    logger.info(f"\nDone in {total_time:.1f}s ({total_time/60:.1f} min)")
    return final_results


def _plot_training_curves(fold_results, out_dir, n_classes):
    """Plot per-fold training curves (BA over rounds)."""
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    try:
        fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    except Exception:
        pass
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for fold_idx, fold in enumerate(fold_results):
        history = fold.get("round_history", [])
        if not history:
            continue
        rounds = [h["round"] for h in history]
        bas = [h.get("val_ba", 0) for h in history]
        aucs = [h.get("val_auc", 0) for h in history]

        axes[0].plot(rounds, bas, marker='o', label=f"Fold {fold_idx+1}", alpha=0.7)
        axes[1].plot(rounds, aucs, marker='s', label=f"Fold {fold_idx+1}", alpha=0.7)

    axes[0].set_xlabel("Federated Round")
    axes[0].set_ylabel("Balanced Accuracy")
    axes[0].set_title(f"PQFL {n_classes}-class: Validation BA per Fold")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Federated Round")
    axes[1].set_ylabel("AUC-ROC (macro)")
    axes[1].set_title(f"PQFL {n_classes}-class: Validation AUC per Fold")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    out_path = out_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches=None)
    plt.close()
    logger.info(f"Saved: {out_path}")


def _plot_confusion_matrix(fold_results, out_dir, n_classes):
    """Plot aggregated confusion matrix across all folds."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    class_names = CLASS_NAMES[n_classes]

    # Aggregate predictions across folds (we don't have raw preds saved, so reconstruct from CMs)
    # Each fold's CM is in metrics["confusion_matrix"]
    total_cm = np.zeros((n_classes, n_classes), dtype=int)
    for fold in fold_results:
        cm = fold.get("confusion_matrix")
        if cm is None:
            continue
        cm = np.array(cm)
        if cm.shape == (n_classes, n_classes):
            total_cm += cm.astype(int)

    # Normalize
    cm_norm = total_cm.astype(float) / total_cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"PQFL {n_classes}-class: Confusion Matrix (counts)")

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title(f"PQFL {n_classes}-class: Confusion Matrix (normalized)")

    out_path = out_dir / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches=None)
    plt.close()
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
