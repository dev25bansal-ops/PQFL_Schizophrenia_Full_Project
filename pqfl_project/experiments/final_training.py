#!/usr/bin/env python3
"""Final training with optimal hyperparameters from sweep.

Best config from hyperparameter sweep (Config 11):
  - n_qubits=6, n_components=71, learning_rate=0.0005
  - dropout=0.5, n_base_layers=2, label_smoothing=0.1, batch_size=16
  - Sweep result: BA=0.7128±0.0648, AUC=0.7121±0.0477

Usage:
    # Full final training with 5-fold CV
    python experiments/final_training.py --data_dir data/processed

    # Quick validation
    python experiments/final_training.py --data_dir data/processed --quick

    # Custom rounds
    python experiments/final_training.py --data_dir data/processed --n_rounds 60 --patience 10
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

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pqfl.data.dataset import FCDataset, SiteFCDataset, MultiSiteDataset
from pqfl.riemannian.engine import RiemannianEngine
from pqfl.quantum.vqc import HybridVQC, VQCConfig
from pqfl.federated.client import PQFLClient
from pqfl.evaluation.metrics import compute_classification_metrics
from pqfl.baselines.classical import TangentSpaceSVM, RiemannianLogisticRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ====== Optimal Configuration from Sweep ======
OPTIMAL_CONFIG = {
    "n_qubits": 6,
    "n_components": 71,
    "learning_rate": 0.0005,
    "dropout": 0.5,
    "n_base_layers": 2,
    "label_smoothing": 0.1,
    "batch_size": 16,
}


def load_and_preprocess(data_dir, n_components, n_rois=100):
    """Load data and run Riemannian preprocessing."""
    data_dir = Path(data_dir)
    npz_files = sorted(data_dir.glob("*_processed.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No *_processed.npz files found in {data_dir}")

    sites = {}
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        fc_matrices = data["fc_matrices"]
        labels = data["labels"]
        site_id = int(data["site_id"])
        site_name = str(data["site_name"])
        fdt_features = data["fdt_features"] if "fdt_features" in data else None

        fc_dataset = FCDataset(
            fc_matrices=fc_matrices, labels=labels,
            fdt_features=fdt_features, site_id=site_id,
        )
        sites[site_id] = SiteFCDataset(
            fc_dataset=fc_dataset, site_name=site_name, site_id=site_id,
        )

    n_rois_actual = list(sites.values())[0].dataset.n_rois

    # Collect all FC matrices and labels
    all_fc, all_labels = [], []
    all_fdt = []
    for site_id, site_ds in sites.items():
        fc, lbl = site_ds.dataset.get_data_for_riemannian()
        all_fc.append(fc)
        all_labels.append(lbl)
        if site_ds.dataset.fdt_features is not None:
            all_fdt.append(site_ds.dataset.fdt_features)
        else:
            all_fdt.append(np.zeros((len(lbl), 0)))

    combined_fc = np.concatenate(all_fc, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    fdt_features = np.concatenate(all_fdt, axis=0) if any(f.size > 0 for f in all_fdt) else None

    # Fit Riemannian engine
    engine = RiemannianEngine(n_rois=n_rois_actual, n_components=n_components)
    engine.fit(combined_fc)
    tangent = engine.transform(combined_fc, return_tensor=False)

    logger.info(
        f"Data loaded: {len(combined_labels)} samples, "
        f"SZ={int(combined_labels.sum())}/HC={int((1-combined_labels).sum())}, "
        f"{n_components} PCA components, "
        f"{engine.tangent_pca.total_explained_variance:.2%} variance"
    )

    return tangent, combined_labels, fdt_features, n_rois_actual, engine


def train_fold(X_train, X_val, y_train, y_val, fdt_train, fdt_val, fdt_dim,
               config, n_rounds, patience, seed, device):
    """Train and evaluate PQFL on a single fold with detailed tracking."""
    n_features = X_train.shape[1]
    n_rois_placeholder = 1

    # Create datasets
    train_ds = FCDataset(
        fc_matrices=np.zeros((len(y_train), n_rois_placeholder, n_rois_placeholder)),
        labels=y_train, tangent_features=X_train, fdt_features=fdt_train,
    )
    val_ds = FCDataset(
        fc_matrices=np.zeros((len(y_val), n_rois_placeholder, n_rois_placeholder)),
        labels=y_val, tangent_features=X_val, fdt_features=fdt_val,
    )

    # Create model
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
    )

    model = HybridVQC(vqc_config)

    # Create dataloaders
    from torch.utils.data import DataLoader

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

    # Create client
    client = PQFLClient(
        model=model, train_loader=train_loader, val_loader=val_loader,
        site_id=0, site_name="cv_fold",
        local_epochs=2, learning_rate=config["learning_rate"],
        label_smoothing=config["label_smoothing"], device=device,
    )

    # Training loop with early stopping and history tracking
    best_ba = 0.0
    best_auc = 0.5
    best_state = None
    best_round = 0
    rounds_without_improvement = 0
    params = client.get_parameters()
    round_history = []

    for round_num in range(n_rounds):
        params, n_samples, metrics = client.fit(params)

        val_ba = metrics.get("balanced_accuracy", 0)
        val_auc = metrics.get("auc_roc", 0.5)
        train_loss = metrics.get("train_loss", 0)

        round_history.append({
            "round": round_num + 1,
            "val_ba": float(val_ba),
            "val_auc": float(val_auc),
            "train_loss": float(train_loss),
        })

        if val_ba > best_ba:
            best_ba = val_ba
            best_auc = val_auc
            best_state = copy.deepcopy(client.model.state_dict())
            best_round = round_num + 1
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1

        if rounds_without_improvement >= patience:
            break

    # Restore best model and evaluate
    if best_state is not None:
        client.model.load_state_dict(best_state)

    final_metrics = client.evaluate()
    final_metrics["best_round"] = best_round
    final_metrics["total_rounds"] = round_num + 1
    final_metrics["round_history"] = round_history

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="PQFL Final Training with Optimal Config")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_rounds", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--quick", action="store_true", help="Quick mode: 3 folds, 20 rounds")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./final_results")
    # Override optimal config (rarely needed)
    parser.add_argument("--n_qubits", type=int, default=None)
    parser.add_argument("--n_components", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--n_base_layers", type=int, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    # Apply optimal config with optional overrides
    config = dict(OPTIMAL_CONFIG)
    if args.n_qubits is not None: config["n_qubits"] = args.n_qubits
    if args.n_components is not None: config["n_components"] = args.n_components
    if args.learning_rate is not None: config["learning_rate"] = args.learning_rate
    if args.dropout is not None: config["dropout"] = args.dropout
    if args.n_base_layers is not None: config["n_base_layers"] = args.n_base_layers
    if args.label_smoothing is not None: config["label_smoothing"] = args.label_smoothing
    if args.batch_size is not None: config["batch_size"] = args.batch_size

    if args.quick:
        args.n_folds = 3
        args.n_rounds = 20
        args.patience = 5

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PQFL FINAL TRAINING — Optimal Configuration from Sweep")
    logger.info("=" * 70)
    logger.info(f"Config: {config}")
    logger.info(f"K={args.n_folds} folds, max {args.n_rounds} rounds, patience={args.patience}")
    logger.info(f"Seed={args.seed}, Device={args.device}")
    logger.info(f"Output: {output_dir}")

    # Load and preprocess
    start_time = time.time()
    tangent, labels, fdt_features, n_rois, engine = load_and_preprocess(
        args.data_dir, config["n_components"], 100
    )
    fdt_dim = fdt_features.shape[1] if fdt_features is not None else 0
    logger.info(f"Riemannian preprocessing done in {time.time()-start_time:.1f}s")

    # ---- PQFL Cross-Validation ----
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 1: PQFL Stratified K-Fold Cross-Validation")
    logger.info(f"{'='*70}")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    pqfl_fold_results = []
    all_fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(tangent, labels)):
        fold_start = time.time()
        logger.info(f"\n--- Fold {fold+1}/{args.n_folds} ---")
        X_train, X_val = tangent[train_idx], tangent[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        fdt_train = fdt_features[train_idx] if fdt_features is not None else None
        fdt_val = fdt_features[val_idx] if fdt_features is not None else None

        logger.info(f"  Train: {len(y_train)} (SZ={int(y_train.sum())}/HC={int((1-y_train).sum())})")
        logger.info(f"  Val:   {len(y_val)} (SZ={int(y_val.sum())}/HC={int((1-y_val).sum())})")

        metrics = train_fold(
            X_train, X_val, y_train, y_val,
            fdt_train, fdt_val, fdt_dim,
            config, args.n_rounds, args.patience, args.seed + fold, args.device,
        )

        pqfl_fold_results.append(metrics)
        if "round_history" in metrics:
            all_fold_histories.append(metrics["round_history"])

        elapsed = time.time() - fold_start
        logger.info(
            f"  Fold {fold+1} result ({elapsed:.1f}s): "
            f"BA={metrics.get('balanced_accuracy', 0):.4f}, "
            f"AUC={metrics.get('auc_roc', 0.5):.4f}, "
            f"Sens={metrics.get('sensitivity', 0):.4f}, "
            f"Spec={metrics.get('specificity', 0):.4f}, "
            f"best_round={metrics.get('best_round', '?')}/{metrics.get('total_rounds', '?')}"
        )

    # ---- Classical Baselines Cross-Validation ----
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 2: Classical Baselines Cross-Validation")
    logger.info(f"{'='*70}")

    svm_bas, svm_aucs, svm_sens, svm_specs = [], [], [], []
    lr_bas, lr_aucs, lr_sens, lr_specs = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(tangent, labels)):
        X_train, X_val = tangent[train_idx], tangent[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        if len(np.unique(y_val)) < 2:
            continue

        # SVM
        svm = TangentSpaceSVM(kernel="rbf", C=1.0)
        svm.fit(X_train, y_train)
        svm_preds = svm.predict(X_val)
        svm_probs = svm.predict_proba(X_val)[:, 1]
        svm_bas.append(balanced_accuracy_score(y_val, svm_preds))
        try:
            svm_aucs.append(roc_auc_score(y_val, svm_probs))
        except ValueError:
            svm_aucs.append(0.5)
        tn, fp, fn, tp = confusion_matrix(y_val, svm_preds, labels=[0, 1]).ravel()
        svm_sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        svm_specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

        # Logistic Regression
        lr = RiemannianLogisticRegression(C=1.0)
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_val)
        lr_probs = lr.predict_proba(X_val)[:, 1]
        lr_bas.append(balanced_accuracy_score(y_val, lr_preds))
        try:
            lr_aucs.append(roc_auc_score(y_val, lr_probs))
        except ValueError:
            lr_aucs.append(0.5)
        tn, fp, fn, tp = confusion_matrix(y_val, lr_preds, labels=[0, 1]).ravel()
        lr_sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        lr_specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    baseline_results = {
        "TangentSVM_CV": {
            "ba_mean": float(np.mean(svm_bas)), "ba_std": float(np.std(svm_bas)),
            "auc_mean": float(np.mean(svm_aucs)), "auc_std": float(np.std(svm_aucs)),
            "sens_mean": float(np.mean(svm_sens)), "spec_mean": float(np.mean(svm_specs)),
        },
        "RiemannianLR_CV": {
            "ba_mean": float(np.mean(lr_bas)), "ba_std": float(np.std(lr_bas)),
            "auc_mean": float(np.mean(lr_aucs)), "auc_std": float(np.std(lr_aucs)),
            "sens_mean": float(np.mean(lr_sens)), "spec_mean": float(np.mean(lr_specs)),
        },
    }

    # ---- Summary ----
    pqfl_bas = [r.get("balanced_accuracy", 0) for r in pqfl_fold_results]
    pqfl_aucs = [r.get("auc_roc", 0.5) for r in pqfl_fold_results]
    pqfl_sens = [r.get("sensitivity", 0) for r in pqfl_fold_results]
    pqfl_specs = [r.get("specificity", 0) for r in pqfl_fold_results]

    pqfl_summary = {
        "ba_mean": float(np.mean(pqfl_bas)), "ba_std": float(np.std(pqfl_bas)),
        "auc_mean": float(np.mean(pqfl_aucs)), "auc_std": float(np.std(pqfl_aucs)),
        "sens_mean": float(np.mean(pqfl_sens)), "sens_std": float(np.std(pqfl_sens)),
        "spec_mean": float(np.mean(pqfl_specs)), "spec_std": float(np.std(pqfl_specs)),
    }

    total_time = time.time() - start_time

    logger.info(f"\n{'='*70}")
    logger.info("FINAL TRAINING RESULTS SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"  Optimal Config: {config}")
    logger.info(f"  PCA Variance: {engine.tangent_pca.total_explained_variance:.2%}")
    logger.info(f"  Total Time: {total_time:.1f}s")
    logger.info("")
    logger.info(
        f"  PQFL ({args.n_folds}-fold CV): "
        f"BA={pqfl_summary['ba_mean']:.4f} +/- {pqfl_summary['ba_std']:.4f}, "
        f"AUC={pqfl_summary['auc_mean']:.4f} +/- {pqfl_summary['auc_std']:.4f}, "
        f"Sens={pqfl_summary['sens_mean']:.4f} +/- {pqfl_summary['sens_std']:.4f}, "
        f"Spec={pqfl_summary['spec_mean']:.4f} +/- {pqfl_summary['spec_std']:.4f}"
    )
    for bl_name, bl_metrics in baseline_results.items():
        logger.info(
            f"  {bl_name}: "
            f"BA={bl_metrics['ba_mean']:.4f} +/- {bl_metrics['ba_std']:.4f}, "
            f"AUC={bl_metrics['auc_mean']:.4f} +/- {bl_metrics['auc_std']:.4f}"
        )

    # Comparison
    logger.info("")
    logger.info("  PQFL vs Baselines:")
    for bl_name, bl_metrics in baseline_results.items():
        pqfl_wins_ba = pqfl_summary["ba_mean"] > bl_metrics["ba_mean"]
        pqfl_wins_auc = pqfl_summary["auc_mean"] > bl_metrics["auc_mean"]
        ba_diff = pqfl_summary["ba_mean"] - bl_metrics["ba_mean"]
        auc_diff = pqfl_summary["auc_mean"] - bl_metrics["auc_mean"]
        logger.info(
            f"  vs {bl_name}: "
            f"BA {'WINS' if pqfl_wins_ba else 'loses'} by {ba_diff:+.4f}, "
            f"AUC {'WINS' if pqfl_wins_auc else 'loses'} by {auc_diff:+.4f}"
        )

    # Per-fold detail
    logger.info(f"\n  Per-fold PQFL results:")
    for i, r in enumerate(pqfl_fold_results):
        logger.info(
            f"    Fold {i+1}: BA={r.get('balanced_accuracy', 0):.4f}, "
            f"AUC={r.get('auc_roc', 0.5):.4f}, "
            f"Sens={r.get('sensitivity', 0):.4f}, Spec={r.get('specificity', 0):.4f}, "
            f"best_round={r.get('best_round', '?')}/{r.get('total_rounds', '?')}"
        )

    # Save comprehensive results
    all_results = {
        "config": config,
        "n_folds": args.n_folds,
        "n_rounds": args.n_rounds,
        "patience": args.patience,
        "seed": args.seed,
        "total_time_seconds": total_time,
        "pqfl_fold_results": pqfl_fold_results,
        "pqfl_summary": pqfl_summary,
        "baseline_results": baseline_results,
        "pca_variance_explained": float(engine.tangent_pca.total_explained_variance),
        "n_components": config["n_components"],
        "n_samples": len(labels),
        "n_sz": int(labels.sum()),
        "n_hc": int((1 - labels).sum()),
        "fold_histories": all_fold_histories,
    }

    with open(output_dir / "final_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_dir / 'final_results.json'}")
    logger.info("=" * 70)
    logger.info("FINAL TRAINING COMPLETE")
    logger.info("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
