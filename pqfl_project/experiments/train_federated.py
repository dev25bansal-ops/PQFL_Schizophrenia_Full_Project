#!/usr/bin/env python3
"""Federated training script for PQFL schizophrenia classification.

Usage:
    # Synthetic data (quick test)
    python experiments/train_federated.py --synthetic --n_sites 5 --n_rounds 50

    # Real preprocessed data
    python experiments/train_federated.py --data_dir ./data/processed --n_rounds 50

    # With FedProx strategy
    python experiments/train_federated.py --data_dir ./data/processed --strategy fedprox --mu 0.01

This script implements the complete PQFL training pipeline:
1. Data loading/generation with site partitioning
2. Riemannian preprocessing (SPD regularization, Frechet mean, tangent PCA)
3. Quantum model creation (HybridVQC with RQFM)
4. Federated training with FedPer/FedProx + early stopping
5. Evaluation with balanced accuracy, AUC-ROC, sensitivity, specificity
6. Classical baseline comparison (cross-validated)
"""

import argparse
import sys
import os
import json
import logging
import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pqfl.data.site_partitioning import SitePartitioner
from pqfl.data.dataset import MultiSiteDataset
from pqfl.riemannian.engine import RiemannianEngine
from pqfl.harmonization.combat import TangentSpaceCombat
from pqfl.quantum.vqc import HybridVQC, VQCConfig
from pqfl.federated.client import PQFLClient
from pqfl.federated.server import PQFLServer
from pqfl.evaluation.metrics import compute_classification_metrics
from pqfl.baselines.classical import TangentSpaceSVM, MDMClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="PQFL Federated Training")
    parser.add_argument("--n_sites", type=int, default=5, help="Number of training sites")
    parser.add_argument("--n_rounds", type=int, default=50, help="Number of federated rounds")
    parser.add_argument("--n_local_epochs", type=int, default=2, help="Local epochs per round (use 2 for small datasets)")
    parser.add_argument("--strategy", type=str, default="fedper", choices=["fedavg", "fedper", "fedprox"])
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx mu parameter")
    parser.add_argument("--n_qubits", type=int, default=6, help="Number of qubits (6 for small data, 12 for multi-site)")
    parser.add_argument("--n_rois", type=int, default=20, help="Number of ROIs (use 20 for demo, 100 for full)")
    parser.add_argument("--n_samples", type=int, default=200, help="Samples per site (synthetic)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory with preprocessed .npz files from scripts/preprocess_real_data.py")
    parser.add_argument("--harmonize", action="store_true", help="Apply ComBat harmonization")
    parser.add_argument("--early_stop_patience", type=int, default=8,
                        help="Stop if val BA doesn't improve for N rounds (0=disabled)")
    parser.add_argument("--n_components", type=int, default=None,
                        help="Number of tangent PCA components (default: auto-compute)")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor (0=none, 0.1=light)")
    parser.add_argument("--learning_rate", type=float, default=0.0005,
                        help="Learning rate for AdamW")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout probability (0.5 for small datasets)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training (16 for small datasets)")
    parser.add_argument("--n_base_layers", type=int, default=2,
                        help="Number of base (shared) VQC layers")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    return parser.parse_args()


def setup_synthetic_data(args):
    """Generate synthetic multi-site data for testing."""
    partitioner = SitePartitioner(
        n_sites=args.n_sites,
        validation_site_ids=set(range(args.n_sites - 1, args.n_sites)),
    )
    
    sites = {}
    for site_id in range(args.n_sites):
        sz_ratio = np.random.uniform(0.3, 0.7)
        site_dataset = SitePartitioner.generate_synthetic_site(
            site_id=site_id,
            n_samples=args.n_samples,
            n_rois=args.n_rois,
            sz_ratio=sz_ratio,
            signal_strength=0.15,
            seed=args.seed + site_id,
        )
        sites[site_id] = site_dataset
    
    return MultiSiteDataset(
        sites=sites,
        validation_site_ids={args.n_sites - 1},
    )


def setup_real_data(args):
    """Load preprocessed real data from .npz files.

    Expects files produced by scripts/preprocess_real_data.py in the format:
        <site_name>_processed.npz

    Each .npz contains:
        - fc_matrices: (n_samples, n_rois, n_rois)
        - labels: (n_samples,)
        - site_id: int
        - site_name: str
        - role: 'training' or 'validation'
        - fdt_features: (n_samples, n_fdt) [optional]
    """
    from pqfl.data.dataset import FCDataset, SiteFCDataset

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all processed .npz files
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

        # Optional FDT features
        fdt_features = data["fdt_features"] if "fdt_features" in data else None

        # Optional subject IDs
        subject_ids = data["subject_ids"].tolist() if "subject_ids" in data else None

        logger.info(
            f"  Loading {site_name} (ID={site_id}, role={role}): "
            f"{len(labels)} subjects, {fc_matrices.shape[1]} ROIs, "
            f"SZ={int(labels.sum())}/HC={int((1-labels).sum())}"
        )

        fc_dataset = FCDataset(
            fc_matrices=fc_matrices,
            labels=labels,
            fdt_features=fdt_features,
            site_id=site_id,
            subject_ids=subject_ids,
        )

        sites[site_id] = SiteFCDataset(
            fc_dataset=fc_dataset,
            site_name=site_name,
            site_id=site_id,
        )

        if role == "validation":
            validation_site_ids.add(site_id)

    # Update args to match real data dimensions
    args.n_rois = list(sites.values())[0].dataset.n_rois
    args.n_sites = len(sites) - len(validation_site_ids)

    return MultiSiteDataset(
        sites=sites,
        validation_site_ids=validation_site_ids,
    )


def run_riemannian_preprocessing(args, multi_site_data, n_components=16):
    """Process all sites through the Riemannian engine."""
    logger.info("Running Riemannian preprocessing...")
    
    # Compute tangent features for each site
    all_tangent_features = {}
    all_labels = {}
    all_site_labels = []
    
    # First pass: fit engine on all training data
    all_fc = []
    all_lbl = []
    for site_id, site_ds in multi_site_data.get_training_sites().items():
        fc, lbl = site_ds.dataset.get_data_for_riemannian()
        all_fc.append(fc)
        all_lbl.append(lbl)
        all_site_labels.extend([site_id] * len(lbl))
    
    # Also include validation sites for fitting the engine
    for site_id, site_ds in multi_site_data.get_validation_sites().items():
        fc, lbl = site_ds.dataset.get_data_for_riemannian()
        all_fc.append(fc)
        all_lbl.append(lbl)
        all_site_labels.extend([site_id] * len(lbl))
    
    combined_fc = np.concatenate(all_fc, axis=0)
    combined_labels = np.concatenate(all_lbl, axis=0)
    
    # Fit Riemannian engine
    engine = RiemannianEngine(
        n_rois=combined_fc.shape[1],
        n_components=n_components,
    )
    engine.fit(combined_fc)
    
    # Transform each site
    for site_id, site_ds in multi_site_data.sites.items():
        fc, lbl = site_ds.dataset.get_data_for_riemannian()
        tangent = engine.transform(fc, return_tensor=False)
        site_ds.dataset.set_tangent_features(tangent)
        all_tangent_features[site_id] = tangent
        all_labels[site_id] = lbl
    
    # Harmonization
    if args.harmonize:
        logger.info("Applying tangent-space ComBat harmonization...")
        combat = TangentSpaceCombat(biological_covariates=["diagnosis"])
        all_tangent_list = []
        all_lbl_list = []
        all_site_list = []
        for site_id in sorted(multi_site_data.sites.keys()):
            all_tangent_list.append(all_tangent_features[site_id])
            all_lbl_list.append(all_labels[site_id])
            all_site_list.extend([site_id] * len(all_labels[site_id]))
        
        combined_tangent = np.concatenate(all_tangent_list, axis=0)
        combined_lbl = np.concatenate(all_lbl_list, axis=0)
        site_labels = np.array(all_site_list)
        
        harmonized = combat.harmonize(
            combined_tangent, site_labels,
            labels=combined_lbl,
        )
        
        # Split back to sites
        offset = 0
        for site_id in sorted(multi_site_data.sites.keys()):
            n = len(all_labels[site_id])
            site_ds = multi_site_data.sites[site_id]
            site_ds.dataset.set_tangent_features(harmonized[offset:offset + n])
            offset += n
    
    return engine


def create_models(args, n_features):
    """Create HybridVQC models for each site."""
    # Adaptive model sizing: smaller models for smaller datasets
    n_base = 2 if n_features <= 32 else 3
    n_personal = 1
    encoder_hidden = [max(16, n_features // 2), args.n_qubits * 2]
    
    config = VQCConfig(
        n_qubits=args.n_qubits,
        n_base_layers=n_base,
        n_personal_layers=n_personal,
        encoding_type="angle",
        entanglement="functional",
        input_dim=n_features,
        encoder_hidden_dims=encoder_hidden,
        fdt_features=0,  # No FDT in demo
        classifier_hidden_dims=[16],
        dropout=args.dropout,
        use_dual_register=False,
    )
    
    models = {}
    for site_id in range(args.n_sites):
        models[site_id] = HybridVQC(config)
    
    return models, config


def create_models_from_data(args, n_features, multi_site_data):
    """Create HybridVQC models for each site based on actual site IDs."""
    # Check if any site has FDT features
    fdt_dim = 0
    for site_id, site_ds in multi_site_data.sites.items():
        if site_ds.dataset.fdt_features is not None:
            fdt_dim = site_ds.dataset.fdt_features.shape[1]
            break
    
    # Use configured n_base_layers (from --n_base_layers argument)
    total_samples = sum(len(s.dataset) for s in multi_site_data.sites.values())
    n_base = args.n_base_layers
    n_personal = 1
    encoder_hidden = [max(16, n_features // 2), args.n_qubits * 2]
    classifier_hidden = [16] if total_samples < 300 else [32]
    
    config = VQCConfig(
        n_qubits=args.n_qubits,
        n_base_layers=n_base,
        n_personal_layers=n_personal,
        encoding_type="angle",
        entanglement="functional",
        input_dim=n_features,
        encoder_hidden_dims=encoder_hidden,
        fdt_features=fdt_dim,
        classifier_hidden_dims=classifier_hidden,
        dropout=args.dropout,
        use_dual_register=False,
    )
    
    models = {}
    for site_id in multi_site_data.sites.keys():
        models[site_id] = HybridVQC(config)
    
    return models, config


def run_federated_training(args, models, multi_site_data):
    """Run the federated training loop with early stopping."""
    logger.info(f"Starting federated training: {args.n_rounds} rounds, strategy={args.strategy}")
    
    # Create clients
    clients = {}
    for site_id, site_ds in multi_site_data.get_training_sites().items():
        train_ds, val_ds = site_ds.dataset.split(train_ratio=0.8, seed=args.seed, stratified=True)
        
        # Log split balance
        train_sz = int(train_ds.labels.sum())
        train_hc = len(train_ds.labels) - train_sz
        val_sz = int(val_ds.labels.sum())
        val_hc = len(val_ds.labels) - val_sz
        logger.info(
            f"  Site {site_ds.site_name} split: train={len(train_ds.labels)} "
            f"(SZ={train_sz}/HC={train_hc}), val={len(val_ds.labels)} "
            f"(SZ={val_sz}/HC={val_hc})"
        )
        
        from torch.utils.data import DataLoader
        
        # Build collate function based on available features
        has_fdt = train_ds.fdt_features is not None
        
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
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=make_collate(has_fdt),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=make_collate(has_fdt),
        )
        
        clients[site_id] = PQFLClient(
            model=models[site_id],
            train_loader=train_loader,
            val_loader=val_loader,
            site_id=site_id,
            site_name=site_ds.site_name,
            local_epochs=args.n_local_epochs,
            learning_rate=args.learning_rate,
            fedprox_mu=args.mu if args.strategy == "fedprox" else 0.0,
            label_smoothing=args.label_smoothing,
            device=args.device,
        )
    
    # Create server
    # Use the first client's parameters (site IDs may not be contiguous)
    first_site_id = sorted(clients.keys())[0]
    initial_params = clients[first_site_id].get_parameters()
    server = PQFLServer(
        strategy=PQFLServer.create_strategy(args.strategy, fedprox_mu=args.mu),
        initial_parameters=initial_params,
        n_rounds=args.n_rounds,
    )
    
    # Early stopping state
    best_val_ba = 0.0
    best_round = 0
    best_model_state = None
    patience = args.early_stop_patience
    
    # Training loop
    for round_num in range(args.n_rounds):
        logger.info(f"=== Round {round_num + 1}/{args.n_rounds} ===")
        
        # Get global parameters
        global_params = server.get_parameters()
        
        # Each client trains locally
        results = []
        for site_id, client in clients.items():
            updated_params, n_samples, metrics = client.fit(global_params)
            results.append((updated_params, n_samples, metrics))
        
        # Server aggregates
        server.aggregate_fit(results)
        
        # Early stopping check: use average val BA across clients
        if patience > 0 and results:
            val_bas = []
            for _, _, metrics in results:
                ba = metrics.get("balanced_accuracy", None)
                if ba is not None:
                    val_bas.append(ba)
            
            if val_bas:
                avg_val_ba = np.mean(val_bas)
                if avg_val_ba > best_val_ba:
                    best_val_ba = avg_val_ba
                    best_round = round_num + 1
                    # Save best model state (first client as proxy for global model)
                    best_model_state = copy.deepcopy(clients[first_site_id].model.state_dict())
                    logger.info(f"  New best val BA: {best_val_ba:.4f} at round {best_round}")
                elif round_num - best_round >= patience:
                    logger.info(
                        f"  Early stopping: val BA didn't improve for {patience} rounds "
                        f"(best={best_val_ba:.4f} at round {best_round})"
                    )
                    break
    
    # Restore best model if early stopping was used
    if patience > 0 and best_model_state is not None:
        for site_id, client in clients.items():
            client.model.load_state_dict(best_model_state)
        logger.info(f"Restored best model from round {best_round} (val BA={best_val_ba:.4f})")
    
    # Log final training summary
    history = server.get_round_history()
    if history:
        final = history[-1]
        logger.info(
            f"Training summary: {len(history)} rounds completed, "
            f"final train_loss={final.get('avg_train_loss', 'N/A'):.4f}, "
            f"final train_acc={final.get('avg_train_acc', 'N/A'):.4f}"
            + (f", best_val_BA={best_val_ba:.4f}" if patience > 0 else "")
        )
    
    return server, clients


def run_baselines(multi_site_data, seed=42):
    """Run classical baseline classifiers with cross-validation.
    
    Handles both multi-site (train/validation) and single-site scenarios:
    - Multi-site: train on training sites, test on validation sites
    - Single-site: stratified 5-fold cross-validation on the single site
    """
    logger.info("Running classical baselines...")
    
    validation_sites = multi_site_data.get_validation_sites()
    training_sites = multi_site_data.get_training_sites()
    
    # Collect all training tangent features and labels
    train_tangent = []
    train_labels = []
    for site_id, site_ds in training_sites.items():
        ds = site_ds.dataset
        if ds.tangent_features is not None:
            train_tangent.append(ds.tangent_features)
            train_labels.append(ds.labels)
    
    # Collect validation tangent features and labels
    test_tangent = []
    test_labels = []
    for site_id, site_ds in validation_sites.items():
        ds = site_ds.dataset
        if ds.tangent_features is not None:
            test_tangent.append(ds.tangent_features)
            test_labels.append(ds.labels)
    
    baselines = {}
    
    if train_tangent and test_tangent:
        # Multi-site: train on training sites, evaluate on validation sites
        X_train = np.concatenate(train_tangent)
        y_train = np.concatenate(train_labels)
        X_test = np.concatenate(test_tangent)
        y_test = np.concatenate(test_labels)
        
        logger.info(f"  Baselines: {len(y_train)} train, {len(y_test)} test samples")
        
        # Tangent Space SVM
        svm = TangentSpaceSVM(kernel="rbf", C=1.0)
        svm.fit(X_train, y_train)
        svm_preds = svm.predict(X_test)
        svm_probs = svm.predict_proba(X_test)[:, 1]
        baselines["TangentSVM"] = compute_classification_metrics(y_test, svm_preds, y_prob=svm_probs)
        
        # Riemannian Logistic Regression
        from pqfl.baselines.classical import RiemannianLogisticRegression
        lr = RiemannianLogisticRegression(C=1.0)
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        lr_probs = lr.predict_proba(X_test)[:, 1]
        baselines["RiemannianLR"] = compute_classification_metrics(y_test, lr_preds, y_prob=lr_probs)
        
        # MDM classifier (operates on raw SPD matrices)
        all_train_fc = []
        for site_id, site_ds in training_sites.items():
            fc, _ = site_ds.dataset.get_data_for_riemannian()
            all_train_fc.append(fc)
        all_test_fc = []
        for site_id, site_ds in validation_sites.items():
            fc, _ = site_ds.dataset.get_data_for_riemannian()
            all_test_fc.append(fc)
        
        if all_train_fc and all_test_fc:
            X_train_fc = np.concatenate(all_train_fc)
            X_test_fc = np.concatenate(all_test_fc)
            mdm = MDMClassifier()
            mdm.fit(X_train_fc, y_train)
            mdm_preds = mdm.predict(X_test_fc)
            baselines["MDM"] = compute_classification_metrics(y_test, mdm_preds)
    
    elif train_tangent:
        # Single-site: use stratified 5-fold cross-validation
        X_all = np.concatenate(train_tangent)
        y_all = np.concatenate(train_labels)
        
        logger.info(f"  Baselines: single-site mode, 5-fold CV on {len(y_all)} samples")
        
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
        # Collect CV results
        svm_bas, svm_aucs = [], []
        lr_bas, lr_aucs = [], []
        mdm_bas = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = y_all[train_idx], y_all[test_idx]
            
            # Skip folds with only one class in test
            if len(np.unique(y_test)) < 2:
                continue
            
            # Tangent Space SVM
            svm = TangentSpaceSVM(kernel="rbf", C=1.0)
            svm.fit(X_train, y_train)
            svm_preds = svm.predict(X_test)
            svm_probs = svm.predict_proba(X_test)[:, 1]
            svm_bas.append(balanced_accuracy_score(y_test, svm_preds))
            try:
                svm_aucs.append(roc_auc_score(y_test, svm_probs))
            except ValueError:
                svm_aucs.append(0.5)
            
            # Riemannian Logistic Regression
            from pqfl.baselines.classical import RiemannianLogisticRegression
            lr = RiemannianLogisticRegression(C=1.0)
            lr.fit(X_train, y_train)
            lr_preds = lr.predict(X_test)
            lr_probs = lr.predict_proba(X_test)[:, 1]
            lr_bas.append(balanced_accuracy_score(y_test, lr_preds))
            try:
                lr_aucs.append(roc_auc_score(y_test, lr_probs))
            except ValueError:
                lr_aucs.append(0.5)
        
        if svm_bas:
            baselines["TangentSVM_CV"] = {
                "balanced_accuracy_mean": float(np.mean(svm_bas)),
                "balanced_accuracy_std": float(np.std(svm_bas)),
                "auc_roc_mean": float(np.mean(svm_aucs)),
                "auc_roc_std": float(np.std(svm_aucs)),
                "n_folds": len(svm_bas),
            }
        if lr_bas:
            baselines["RiemannianLR_CV"] = {
                "balanced_accuracy_mean": float(np.mean(lr_bas)),
                "balanced_accuracy_std": float(np.std(lr_bas)),
                "auc_roc_mean": float(np.mean(lr_aucs)),
                "auc_roc_std": float(np.std(lr_aucs)),
                "n_folds": len(lr_bas),
            }
    else:
        logger.warning("No tangent features available for baselines")
        return {}
    
    for name, metrics in baselines.items():
        if "balanced_accuracy_mean" in metrics:
            logger.info(
                f"  {name}: BA={metrics['balanced_accuracy_mean']:.4f} "
                f"± {metrics['balanced_accuracy_std']:.4f}, "
                f"AUC={metrics['auc_roc_mean']:.4f} ± {metrics['auc_roc_std']:.4f}"
            )
        else:
            logger.info(
                f"  {name}: BA={metrics.get('balanced_accuracy', 0):.4f}, "
                f"AUC={metrics.get('auc_roc', 'N/A')}"
            )
    
    return baselines


def evaluate_federated(clients, multi_site_data):
    """Evaluate federated models on validation sites.
    
    For multi-site: evaluate on dedicated validation sites.
    For single-site: evaluate the client's internal val split.
    """
    logger.info("Evaluating federated models...")
    
    results = {}
    validation_sites = multi_site_data.get_validation_sites()
    
    if validation_sites:
        # Multi-site: evaluate on dedicated validation sites
        for site_id, site_ds in validation_sites.items():
            ds = site_ds.dataset
            if ds.tangent_features is None:
                continue
            
            # Use the first training site's model (in real FL, global model would be used)
            first_site_id = sorted(clients.keys())[0]
            model = clients[first_site_id].model
            model.eval()
            
            from torch.utils.data import DataLoader
            has_fdt = ds.fdt_features is not None
            
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
            
            loader = DataLoader(
                ds,
                batch_size=32,
                shuffle=False,
                collate_fn=make_collate(has_fdt),
            )
            
            all_preds = []
            all_labels = []
            all_probs = []
            with torch.no_grad():
                for batch in loader:
                    x = batch["tangent_features"]
                    y = batch["label"]
                    fdt = batch.get("fdt_features")
                    logits = model(x, fdt_features=fdt)
                    probs = torch.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)
                    all_preds.extend(preds.numpy())
                    all_labels.extend(y.numpy())
                    all_probs.extend(probs[:, 1].numpy())
            
            metrics = compute_classification_metrics(
                np.array(all_labels), np.array(all_preds), y_prob=np.array(all_probs)
            )
            results[site_ds.site_name] = metrics
            logger.info(
                f"  Site {site_ds.site_name}: BA={metrics.get('balanced_accuracy', 0):.4f}, "
                f"AUC={metrics.get('auc_roc', 'N/A')}, "
                f"Sens={metrics.get('sensitivity', 0):.4f}, Spec={metrics.get('specificity', 0):.4f}"
            )
    else:
        # Single-site: evaluate on each client's internal val split
        logger.info("  No validation sites — evaluating on internal val splits")
        for site_id, client in clients.items():
            val_metrics = client.evaluate()
            if val_metrics:
                results[client.site_name] = val_metrics
                logger.info(
                    f"  Site {client.site_name} (val split): "
                    f"BA={val_metrics.get('balanced_accuracy', 0):.4f}, "
                    f"AUC={val_metrics.get('auc_roc', 'N/A')}, "
                    f"Sens={val_metrics.get('sensitivity', 0):.4f}, "
                    f"Spec={val_metrics.get('specificity', 0):.4f}"
                )
    
    return results


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup output directory
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"PQFL Federated Training - Config: {vars(args)}")
    
    # Step 1: Load data (real or synthetic)
    if args.data_dir:
        logger.info(f"Loading real preprocessed data from: {args.data_dir}")
        multi_site_data = setup_real_data(args)
    elif args.synthetic:
        logger.info("Using synthetic data")
        multi_site_data = setup_synthetic_data(args)
    else:
        logger.info("No --data_dir or --synthetic specified. Defaulting to synthetic data.")
        args.synthetic = True
        multi_site_data = setup_synthetic_data(args)
    
    data_summary = multi_site_data.get_summary()
    logger.info(
        f"Data summary: {data_summary['total_samples']['total']} total samples, "
        f"SZ={data_summary['total_samples']['total_sz']}, HC={data_summary['total_samples']['total_hc']}, "
        f"{data_summary['n_training_sites']} training + {data_summary['n_validation_sites']} validation sites"
    )
    
    # Step 2: Riemannian preprocessing
    # Auto-compute n_components: use enough to capture >95% variance
    # but not more than min(n_samples-1, tangent_dim) or a reasonable upper bound
    tangent_dim = args.n_rois * (args.n_rois + 1) // 2
    n_samples_total = data_summary['total_samples']['total']
    if args.n_components is not None:
        n_components = args.n_components
    else:
        # Rule of thumb: enough components for rich representation
        # but capped by sample size and a practical upper bound
        n_components = min(
            n_samples_total - 1,   # Can't have more components than samples
            tangent_dim - 1,        # Can't have more than tangent dimension
            max(32, int(np.sqrt(tangent_dim))),  # At least sqrt(dim) or 32
        )
        # For 100 ROIs: tangent_dim=5050, sqrt=71 → use 71 components
        # For 20 ROIs: tangent_dim=210, sqrt=14 → use 32 components
    logger.info(f"Using {n_components} tangent PCA components (from {tangent_dim} tangent dimensions)")
    engine = run_riemannian_preprocessing(args, multi_site_data, n_components=n_components)
    
    # Step 3: Create models (use real site IDs for real data)
    if args.data_dir:
        models, config = create_models_from_data(args, n_features=n_components, multi_site_data=multi_site_data)
    else:
        models, config = create_models(args, n_features=n_components)
    first_model = list(models.values())[0]
    logger.info(f"Model: {config}")
    logger.info(f"Shared params: {first_model.count_shared_params()}, Personal params: {first_model.count_personal_params()}")
    
    # Step 4: Federated training (with early stopping)
    server, clients = run_federated_training(args, models, multi_site_data)
    
    # Step 5: Evaluation
    fed_results = evaluate_federated(clients, multi_site_data)
    
    # Step 6: Baselines
    baseline_results = run_baselines(multi_site_data, seed=args.seed)
    
    # Collect client histories
    client_histories = {}
    for site_id, client in clients.items():
        client_histories[client.site_name] = client.get_history()
    
    # Save results
    def make_serializable(obj):
        """Convert numpy types to Python native for JSON serialization."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    all_results = {
        "config": vars(args),
        "federated_results": make_serializable(fed_results),
        "baseline_results": make_serializable(baseline_results),
        "round_history": make_serializable(server.get_round_history()),
        "client_histories": make_serializable(client_histories),
    }
    
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print final summary
    logger.info("=" * 60)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 60)
    
    for site_name, metrics in fed_results.items():
        ba = metrics.get("balanced_accuracy", metrics.get("balanced_accuracy_mean", 0))
        auc = metrics.get("auc_roc", metrics.get("auc_roc_mean", "N/A"))
        sens = metrics.get("sensitivity", 0)
        spec = metrics.get("specificity", 0)
        logger.info(
            f"  PQFL @ {site_name}: BA={ba:.4f}, AUC={auc}, "
            f"Sens={sens:.4f}, Spec={spec:.4f}"
        )
    
    for name, metrics in baseline_results.items():
        if "balanced_accuracy_mean" in metrics:
            ba = metrics["balanced_accuracy_mean"]
            ba_std = metrics.get("balanced_accuracy_std", 0)
            auc = metrics.get("auc_roc_mean", "N/A")
            logger.info(f"  {name}: BA={ba:.4f} ± {ba_std:.4f}, AUC={auc}")
        else:
            ba = metrics.get("balanced_accuracy", 0)
            auc = metrics.get("auc_roc", "N/A")
            logger.info(f"  {name}: BA={ba:.4f}, AUC={auc}")
    
    logger.info(f"Results saved to {results_path}")
    logger.info("Training complete!")
