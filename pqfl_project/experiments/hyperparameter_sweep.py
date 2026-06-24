#!/usr/bin/env python3
"""Hyperparameter sweep for PQFL schizophrenia classification.

Systematically explores hyperparameter combinations to find the
optimal configuration. Supports both grid search and random search.

Key hyperparameters explored:
- n_qubits: 4, 6, 8
- n_components: 32, 50, 71, 100
- learning_rate: 0.0001, 0.0005, 0.001
- dropout: 0.3, 0.5, 0.7
- n_base_layers: 1, 2, 3
- label_smoothing: 0.0, 0.1, 0.2
- batch_size: 16, 32

Each configuration is evaluated with stratified 5-fold cross-validation
for robust comparison.

Usage:
    # Quick sweep (top candidates only)
    python experiments/hyperparameter_sweep.py --data_dir data/processed --mode quick

    # Full grid search
    python experiments/hyperparameter_sweep.py --data_dir data/processed --mode full

    # Random search with N trials
    python experiments/hyperparameter_sweep.py --data_dir data/processed --mode random --n_trials 20
"""

import argparse
import sys
import os
import json
import copy
import itertools
import time
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pqfl.data.dataset import FCDataset, SiteFCDataset, MultiSiteDataset
from pqfl.riemannian.engine import RiemannianEngine
from pqfl.quantum.vqc import HybridVQC, VQCConfig
from pqfl.federated.client import PQFLClient
from pqfl.federated.server import PQFLServer
from pqfl.evaluation.metrics import compute_classification_metrics
from pqfl.baselines.classical import TangentSpaceSVM, RiemannianLogisticRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Hyperparameter search spaces
QUICK_CONFIGS = [
    # Best current config (6q, 71c) as baseline
    {"n_qubits": 6, "n_components": 71, "learning_rate": 0.0005, "dropout": 0.5,
     "n_base_layers": 2, "label_smoothing": 0.1, "batch_size": 32},
    # Fewer qubits
    {"n_qubits": 4, "n_components": 50, "learning_rate": 0.0005, "dropout": 0.5,
     "n_base_layers": 2, "label_smoothing": 0.1, "batch_size": 32},
    # More qubits, more components
    {"n_qubits": 8, "n_components": 71, "learning_rate": 0.0005, "dropout": 0.5,
     "n_base_layers": 2, "label_smoothing": 0.1, "batch_size": 32},
    # Lower learning rate
    {"n_qubits": 6, "n_components": 71, "learning_rate": 0.0001, "dropout": 0.5,
     "n_base_layers": 2, "label_smoothing": 0.1, "batch_size": 32},
    # Higher learning rate
    {"n_qubits": 6, "n_components": 71, "learning_rate": 0.001, "dropout": 0.5,
     "n_base_layers": 2, "label_smoothing": 0.1, "batch_size": 32},
    # Less dropout
    {"n_qubits": 6, "n_components": 71, "learning_rate": 0.0005, "dropout": 0.3,
     "n_base_layers": 2, "label_smoothing": 0.1, "batch_size": 32},
    # More dropout
    {"n_qubits": 6, "n_components": 71, "learning_rate": 0.0005, "dropout": 0.7,
     "n_base_layers": 2, "label_smoothing": 0.1, "batch_size": 32},
    # Single base layer
    {"n_qubits": 6, "n_components": 71, "learning_rate": 0.0005, "dropout": 0.5,
     "n_base_layers": 1, "label_smoothing": 0.1, "batch_size": 32},
    # 3 base layers
    {"n_qubits": 6, "n_components": 71, "learning_rate": 0.0005, "dropout": 0.5,
     "n_base_layers": 3, "label_smoothing": 0.1, "batch_size": 32},
    # No label smoothing
    {"n_qubits": 6, "n_components": 71, "learning_rate": 0.0005, "dropout": 0.5,
     "n_base_layers": 2, "label_smoothing": 0.0, "batch_size": 32},
    # Smaller batch size
    {"n_qubits": 6, "n_components": 71, "learning_rate": 0.0005, "dropout": 0.5,
     "n_base_layers": 2, "label_smoothing": 0.1, "batch_size": 16},
    # 4 qubits, fewer components
    {"n_qubits": 4, "n_components": 32, "learning_rate": 0.0005, "dropout": 0.5,
     "n_base_layers": 2, "label_smoothing": 0.1, "batch_size": 32},
]

FULL_GRID = {
    "n_qubits": [4, 6, 8],
    "n_components": [32, 50, 71],
    "learning_rate": [0.0001, 0.0005, 0.001],
    "dropout": [0.3, 0.5, 0.7],
    "n_base_layers": [1, 2, 3],
    "label_smoothing": [0.0, 0.1, 0.2],
    "batch_size": [16, 32],
}

RANDOM_SPACE = {
    "n_qubits": [4, 6, 8],
    "n_components": [32, 50, 71, 100],
    "learning_rate": [0.00005, 0.0001, 0.0003, 0.0005, 0.001, 0.002],
    "dropout": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "n_base_layers": [1, 2, 3],
    "label_smoothing": [0.0, 0.05, 0.1, 0.15, 0.2],
    "batch_size": [16, 32],
}


def load_data(data_dir):
    """Load preprocessed data from .npz files."""
    data_dir = Path(data_dir)
    npz_files = sorted(data_dir.glob("*_processed.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No *_processed.npz files found in {data_dir}")

    logger.info(f"Found {len(npz_files)} processed site files")

    sites = {}
    validation_site_ids = set()

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        fc_matrices = data["fc_matrices"]
        labels = data["labels"]
        site_id = int(data["site_id"])
        site_name = str(data["site_name"])
        role = str(data.get("role", "training"))
        fdt_features = data["fdt_features"] if "fdt_features" in data else None

        fc_dataset = FCDataset(
            fc_matrices=fc_matrices, labels=labels,
            fdt_features=fdt_features, site_id=site_id,
        )
        sites[site_id] = SiteFCDataset(
            fc_dataset=fc_dataset, site_name=site_name, site_id=site_id,
        )
        if role == "validation":
            validation_site_ids.add(site_id)

    n_rois = list(sites.values())[0].dataset.n_rois
    return sites, validation_site_ids, n_rois


def prepare_riemannian_features(sites, n_components, n_rois):
    """Run Riemannian preprocessing and return tangent features + labels."""
    # Collect all FC matrices
    all_fc, all_labels = [], []
    for site_id, site_ds in sites.items():
        fc, lbl = site_ds.dataset.get_data_for_riemannian()
        all_fc.append(fc)
        all_labels.append(lbl)

    combined_fc = np.concatenate(all_fc, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    # Fit Riemannian engine
    engine = RiemannianEngine(n_rois=n_rois, n_components=n_components)
    engine.fit(combined_fc)

    # Transform
    tangent = engine.transform(combined_fc, return_tensor=False)

    # Get FDT features
    fdt_all = []
    for site_id, site_ds in sites.items():
        if site_ds.dataset.fdt_features is not None:
            fdt_all.append(site_ds.dataset.fdt_features)
        else:
            fdt_all.append(np.zeros((len(site_ds.dataset), 0)))

    fdt_features = np.concatenate(fdt_all, axis=0) if any(f.size > 0 for f in fdt_all) else None

    return tangent, combined_labels, fdt_features, engine


def evaluate_config_with_cv(
    config, tangent_features, labels, fdt_features, n_rois,
    n_folds=5, n_rounds=30, seed=42, device="cpu"
):
    """Evaluate a single hyperparameter configuration using stratified k-fold CV.

    Returns dict with mean and std of BA and AUC across folds.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(tangent_features, labels)):
        X_train, X_val = tangent_features[train_idx], tangent_features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        if fdt_features is not None:
            fdt_train = fdt_features[train_idx]
            fdt_val = fdt_features[val_idx]
        else:
            fdt_train = None
            fdt_val = None

        fdt_dim = fdt_train.shape[1] if fdt_train is not None else 0

        # Create datasets
        train_ds = FCDataset(
            fc_matrices=np.zeros((len(y_train), n_rois, n_rois)),
            labels=y_train,
            tangent_features=X_train,
            fdt_features=fdt_train,
        )
        val_ds = FCDataset(
            fc_matrices=np.zeros((len(y_val), n_rois, n_rois)),
            labels=y_val,
            tangent_features=X_val,
            fdt_features=fdt_val,
        )

        # Create model
        n_features = X_train.shape[1]
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

        has_fdt = fdt_train is not None

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
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            site_id=0,
            site_name="cv_fold",
            local_epochs=2,
            learning_rate=config["learning_rate"],
            label_smoothing=config["label_smoothing"],
            device=device,
        )

        # Get initial parameters
        initial_params = client.get_parameters()

        # Simple training loop (no federated — just local training with early stopping)
        best_ba = 0.0
        best_state = None
        patience = 8
        rounds_without_improvement = 0

        for round_num in range(n_rounds):
            # Train one round
            updated_params, n_samples, metrics = client.fit(initial_params)

            val_ba = metrics.get("balanced_accuracy", 0)
            val_auc = metrics.get("auc_roc", 0.5)

            if val_ba > best_ba:
                best_ba = val_ba
                best_state = copy.deepcopy(client.model.state_dict())
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

            if rounds_without_improvement >= patience:
                break

            # Update params for next round
            initial_params = updated_params

        # Restore best model and evaluate
        if best_state is not None:
            client.model.load_state_dict(best_state)

        val_metrics = client.evaluate()
        fold_results.append(val_metrics)

    # Aggregate fold results
    bas = [r.get("balanced_accuracy", 0) for r in fold_results]
    aucs = [r.get("auc_roc", 0.5) for r in fold_results]
    sens = [r.get("sensitivity", 0) for r in fold_results]
    specs = [r.get("specificity", 0) for r in fold_results]

    return {
        "ba_mean": float(np.mean(bas)),
        "ba_std": float(np.std(bas)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "sens_mean": float(np.mean(sens)),
        "sens_std": float(np.std(sens)),
        "spec_mean": float(np.mean(specs)),
        "spec_std": float(np.std(specs)),
        "n_folds": len(fold_results),
    }


def run_classical_baselines_cv(tangent_features, labels, n_folds=5, seed=42):
    """Run classical baselines with stratified k-fold CV for comparison."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    svm_bas, svm_aucs = [], []
    lr_bas, lr_aucs = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(tangent_features, labels)):
        X_train, X_test = tangent_features[train_idx], tangent_features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        if len(np.unique(y_test)) < 2:
            continue

        # SVM
        svm = TangentSpaceSVM(kernel="rbf", C=1.0)
        svm.fit(X_train, y_train)
        svm_preds = svm.predict(X_test)
        svm_probs = svm.predict_proba(X_test)[:, 1]
        svm_bas.append(balanced_accuracy_score(y_test, svm_preds))
        try:
            svm_aucs.append(roc_auc_score(y_test, svm_probs))
        except ValueError:
            svm_aucs.append(0.5)

        # LR
        lr = RiemannianLogisticRegression(C=1.0)
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        lr_probs = lr.predict_proba(X_test)[:, 1]
        lr_bas.append(balanced_accuracy_score(y_test, lr_preds))
        try:
            lr_aucs.append(roc_auc_score(y_test, lr_probs))
        except ValueError:
            lr_aucs.append(0.5)

    return {
        "TangentSVM_CV": {
            "ba_mean": float(np.mean(svm_bas)), "ba_std": float(np.std(svm_bas)),
            "auc_mean": float(np.mean(svm_aucs)), "auc_std": float(np.std(svm_aucs)),
        },
        "RiemannianLR_CV": {
            "ba_mean": float(np.mean(lr_bas)), "ba_std": float(np.std(lr_bas)),
            "auc_mean": float(np.mean(lr_aucs)), "auc_std": float(np.std(lr_aucs)),
        },
    }


def generate_configs(mode, n_trials=20, seed=42):
    """Generate list of hyperparameter configurations to evaluate."""
    if mode == "quick":
        return QUICK_CONFIGS
    elif mode == "full":
        keys = list(FULL_GRID.keys())
        values = [FULL_GRID[k] for k in keys]
        configs = []
        for combo in itertools.product(*values):
            configs.append(dict(zip(keys, combo)))
        logger.info(f"Full grid: {len(configs)} configurations")
        return configs
    elif mode == "random":
        rng = np.random.RandomState(seed)
        configs = []
        for _ in range(n_trials):
            config = {}
            for key, values in RANDOM_SPACE.items():
                config[key] = rng.choice(values)
            configs.append(config)
        return configs
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="PQFL Hyperparameter Sweep")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["quick", "full", "random"])
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of random search trials")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--n_rounds", type=int, default=30,
                        help="Max training rounds per fold")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./sweep_results")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup output
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Hyperparameter Sweep - Mode: {args.mode}")
    logger.info(f"Data directory: {args.data_dir}")

    # Load data
    sites, val_site_ids, n_rois = load_data(args.data_dir)
    logger.info(f"Loaded {len(sites)} sites, {n_rois} ROIs")

    # Generate configs
    configs = generate_configs(args.mode, args.n_trials, args.seed)
    logger.info(f"Evaluating {len(configs)} configurations with {args.n_folds}-fold CV")

    # Cache Riemannian features for the most common n_components
    # We'll compute per-config as needed
    riemannian_cache = {}

    # Run baseline once (using max n_components)
    max_components = max(c["n_components"] for c in configs)
    logger.info(f"Computing Riemannian features with {max_components} components for baselines...")
    tangent_max, labels, fdt_features, engine_max = prepare_riemannian_features(
        sites, max_components, n_rois
    )
    baselines = run_classical_baselines_cv(tangent_max, labels, args.n_folds, args.seed)
    logger.info(f"Baseline results: {baselines}")

    # Evaluate each configuration
    results = []
    for i, config in enumerate(configs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Config {i+1}/{len(configs)}: {config}")

        # Get or compute Riemannian features for this n_components
        n_comp = config["n_components"]
        if n_comp not in riemannian_cache:
            logger.info(f"  Computing Riemannian features with {n_comp} components...")
            tangent, _, _, _ = prepare_riemannian_features(sites, n_comp, n_rois)
            riemannian_cache[n_comp] = tangent
        else:
            tangent = riemannian_cache[n_comp]
            logger.info(f"  Using cached Riemannian features ({n_comp} components)")

        start_time = time.time()
        try:
            metrics = evaluate_config_with_cv(
                config, tangent, labels, fdt_features, n_rois,
                n_folds=args.n_folds, n_rounds=args.n_rounds,
                seed=args.seed, device=args.device,
            )
            elapsed = time.time() - start_time
            logger.info(
                f"  Result: BA={metrics['ba_mean']:.4f} ± {metrics['ba_std']:.4f}, "
                f"AUC={metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}, "
                f"Sens={metrics['sens_mean']:.4f}, Spec={metrics['spec_mean']:.4f} "
                f"({elapsed:.1f}s)"
            )
        except Exception as e:
            logger.error(f"  Config failed: {e}")
            metrics = {
                "ba_mean": 0, "ba_std": 0, "auc_mean": 0.5, "auc_std": 0,
                "sens_mean": 0, "sens_std": 0, "spec_mean": 0, "spec_std": 0,
                "error": str(e),
            }
            elapsed = time.time() - start_time

        result = {
            "config_id": i,
            "config": config,
            "metrics": metrics,
            "elapsed_seconds": elapsed,
        }
        results.append(result)

        # Save intermediate results
        with open(output_dir / "sweep_results.json", "w") as f:
            json.dump({
                "baselines": baselines,
                "results": results,
                "n_configs": len(configs),
                "n_completed": i + 1,
            }, f, indent=2, default=str)

    # Find best configuration
    valid_results = [r for r in results if "error" not in r.get("metrics", {})]
    if valid_results:
        best = max(valid_results, key=lambda r: r["metrics"]["ba_mean"])
        logger.info(f"\n{'='*60}")
        logger.info("BEST CONFIGURATION")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {best['config']}")
        logger.info(
            f"BA={best['metrics']['ba_mean']:.4f} ± {best['metrics']['ba_std']:.4f}, "
            f"AUC={best['metrics']['auc_mean']:.4f} ± {best['metrics']['auc_std']:.4f}"
        )
        logger.info(f"Sens={best['metrics']['sens_mean']:.4f}, Spec={best['metrics']['spec_mean']:.4f}")

        # Compare with baselines
        for bl_name, bl_metrics in baselines.items():
            logger.info(
                f"  vs {bl_name}: BA={bl_metrics['ba_mean']:.4f} ± {bl_metrics['ba_std']:.4f}, "
                f"AUC={bl_metrics['auc_mean']:.4f} ± {bl_metrics['auc_std']:.4f}"
            )

    # Save final results
    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump({
            "baselines": baselines,
            "results": results,
            "best_config": best["config"] if valid_results else None,
            "best_metrics": best["metrics"] if valid_results else None,
            "n_configs": len(configs),
        }, f, indent=2, default=str)

    logger.info(f"\nSweep complete! Results saved to {output_dir / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
