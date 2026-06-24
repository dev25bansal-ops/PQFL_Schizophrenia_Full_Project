#!/usr/bin/env python3
"""Stratified k-fold cross-validation for PQFL schizophrenia classification.

Provides a fair comparison between PQFL and classical baselines by
evaluating both under the same cross-validation scheme. This eliminates
the train/val split variance that can make single-run comparisons unreliable.

Usage:
    # 5-fold CV with default config (6 qubits)
    python experiments/evaluate_pqfl_cv.py --data_dir data/processed --n_folds 5

    # Custom config
    python experiments/evaluate_pqfl_cv.py --data_dir data/processed --n_qubits 6 \
        --n_components 71 --learning_rate 0.0005 --dropout 0.5

    # Quick test with 3 folds
    python experiments/evaluate_pqfl_cv.py --data_dir data/processed --n_folds 3 --n_rounds 15
"""

import argparse
import sys
import os
import json
import copy
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


def load_and_preprocess(data_dir, n_components, n_rois):
    """Load data and run Riemannian preprocessing."""
    from pqfl.data.dataset import FCDataset, SiteFCDataset

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


def train_and_evaluate_fold(
    X_train, X_val, y_train, y_val,
    fdt_train, fdt_val, fdt_dim,
    config, n_rounds, patience, seed, device
):
    """Train and evaluate PQFL on a single fold."""
    n_features = X_train.shape[1]
    n_rois_placeholder = 1  # Not used for tangent features

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

    # Training loop with early stopping
    best_ba = 0.0
    best_auc = 0.5
    best_state = None
    best_round = 0
    rounds_without_improvement = 0
    params = client.get_parameters()

    for round_num in range(n_rounds):
        params, n_samples, metrics = client.fit(params)

        val_ba = metrics.get("balanced_accuracy", 0)
        val_auc = metrics.get("auc_roc", 0.5)

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

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="PQFL Stratified K-Fold Cross-Validation")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_rounds", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--n_qubits", type=int, default=6)
    parser.add_argument("--n_components", type=int, default=71)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--n_base_layers", type=int, default=2)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./cv_results")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "n_qubits": args.n_qubits,
        "n_components": args.n_components,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "n_base_layers": args.n_base_layers,
        "label_smoothing": args.label_smoothing,
        "batch_size": args.batch_size,
    }

    logger.info(f"PQFL Cross-Validation - Config: {config}")
    logger.info(f"K={args.n_folds} folds, max {args.n_rounds} rounds, patience={args.patience}")

    # Load and preprocess
    tangent, labels, fdt_features, n_rois, engine = load_and_preprocess(
        args.data_dir, args.n_components, 100
    )
    fdt_dim = fdt_features.shape[1] if fdt_features is not None else 0

    # ---- PQFL Cross-Validation ----
    logger.info(f"\n{'='*60}")
    logger.info("PQFL Stratified K-Fold Cross-Validation")
    logger.info(f"{'='*60}")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    pqfl_fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(tangent, labels)):
        logger.info(f"\n--- Fold {fold+1}/{args.n_folds} ---")
        X_train, X_val = tangent[train_idx], tangent[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        fdt_train = fdt_features[train_idx] if fdt_features is not None else None
        fdt_val = fdt_features[val_idx] if fdt_features is not None else None

        logger.info(f"  Train: {len(y_train)} (SZ={int(y_train.sum())}/HC={int((1-y_train).sum())})")
        logger.info(f"  Val:   {len(y_val)} (SZ={int(y_val.sum())}/HC={int((1-y_val).sum())})")

        metrics = train_and_evaluate_fold(
            X_train, X_val, y_train, y_val,
            fdt_train, fdt_val, fdt_dim,
            config, args.n_rounds, args.patience, args.seed + fold, args.device,
        )

        pqfl_fold_results.append(metrics)
        logger.info(
            f"  Fold {fold+1} result: BA={metrics.get('balanced_accuracy', 0):.4f}, "
            f"AUC={metrics.get('auc_roc', 0.5):.4f}, "
            f"Sens={metrics.get('sensitivity', 0):.4f}, "
            f"Spec={metrics.get('specificity', 0):.4f}, "
            f"best_round={metrics.get('best_round', '?')}"
        )

    # ---- Classical Baselines Cross-Validation ----
    logger.info(f"\n{'='*60}")
    logger.info("Classical Baselines Stratified K-Fold Cross-Validation")
    logger.info(f"{'='*60}")

    svm_bas, svm_aucs = [], []
    lr_bas, lr_aucs = [], []

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

    baseline_results = {
        "TangentSVM_CV": {
            "ba_mean": float(np.mean(svm_bas)), "ba_std": float(np.std(svm_bas)),
            "auc_mean": float(np.mean(svm_aucs)), "auc_std": float(np.std(svm_aucs)),
        },
        "RiemannianLR_CV": {
            "ba_mean": float(np.mean(lr_bas)), "ba_std": float(np.std(lr_bas)),
            "auc_mean": float(np.mean(lr_aucs)), "auc_std": float(np.std(lr_aucs)),
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

    logger.info(f"\n{'='*60}")
    logger.info("CROSS-VALIDATION RESULTS SUMMARY")
    logger.info(f"{'='*60}")
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

    # Check if PQFL beats baselines
    for bl_name, bl_metrics in baseline_results.items():
        pqfl_wins_ba = pqfl_summary["ba_mean"] > bl_metrics["ba_mean"]
        pqfl_wins_auc = pqfl_summary["auc_mean"] > bl_metrics["auc_mean"]
        logger.info(
            f"  PQFL vs {bl_name}: "
            f"BA {'WINS' if pqfl_wins_ba else 'loses'} "
            f"({pqfl_summary['ba_mean']:.4f} vs {bl_metrics['ba_mean']:.4f}), "
            f"AUC {'WINS' if pqfl_wins_auc else 'loses'} "
            f"({pqfl_summary['auc_mean']:.4f} vs {bl_metrics['auc_mean']:.4f})"
        )

    # Save results
    all_results = {
        "config": config,
        "n_folds": args.n_folds,
        "n_rounds": args.n_rounds,
        "patience": args.patience,
        "pqfl_fold_results": pqfl_fold_results,
        "pqfl_summary": pqfl_summary,
        "baseline_results": baseline_results,
        "pca_variance_explained": float(engine.tangent_pca.total_explained_variance),
        "n_components": args.n_components,
    }

    with open(output_dir / "cv_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_dir / 'cv_results.json'}")


if __name__ == "__main__":
    main()
