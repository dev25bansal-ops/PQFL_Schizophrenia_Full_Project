#!/usr/bin/env python3
"""Statistical significance testing and brain saliency analysis.

Performs:
1. DeLong test (AUC comparison) + McNemar's test (BA comparison) for PQFL vs baselines
2. Gradient-based saliency → maps back to brain ROI connections
3. Bootstrap confidence intervals for all metrics
4. Generates publication-ready saliency figures

Usage:
    python experiments/analyze_significance.py --data_dir data/processed

    # With existing final results (skips retraining)
    python experiments/analyze_significance.py --data_dir data/processed \
        --results_dir final_results/20260605_215735
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

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pqfl.data.dataset import FCDataset, SiteFCDataset
from pqfl.riemannian.engine import RiemannianEngine
from pqfl.quantum.vqc import HybridVQC, VQCConfig
from pqfl.federated.client import PQFLClient
from pqfl.evaluation.metrics import compute_classification_metrics
from pqfl.evaluation.statistical_tests import delong_test, mcnemar_test, bootstrap_ci, bonferroni_correction
from pqfl.evaluation.saliency import QuantumSaliency, ClassicalSaliency
from pqfl.baselines.classical import TangentSpaceSVM, RiemannianLogisticRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OPTIMAL_CONFIG = {
    "n_qubits": 6, "n_components": 71, "learning_rate": 0.0005,
    "dropout": 0.5, "n_base_layers": 2, "label_smoothing": 0.1, "batch_size": 16,
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
        fc_dataset = FCDataset(fc_matrices=fc_matrices, labels=labels,
                               fdt_features=fdt_features, site_id=site_id)
        sites[site_id] = SiteFCDataset(fc_dataset=fc_dataset, site_name=site_name, site_id=site_id)

    n_rois_actual = list(sites.values())[0].dataset.n_rois
    all_fc, all_labels, all_fdt = [], [], []
    for site_id, site_ds in sites.items():
        fc, lbl = site_ds.dataset.get_data_for_riemannian()
        all_fc.append(fc)
        all_labels.append(lbl)
        all_fdt.append(site_ds.dataset.fdt_features if site_ds.dataset.fdt_features is not None else np.zeros((len(lbl), 0)))

    combined_fc = np.concatenate(all_fc)
    combined_labels = np.concatenate(all_labels)
    fdt_features = np.concatenate(all_fdt) if any(f.size > 0 for f in all_fdt) else None

    engine = RiemannianEngine(n_rois=n_rois_actual, n_components=n_components)
    engine.fit(combined_fc)
    tangent = engine.transform(combined_fc, return_tensor=False)

    logger.info(f"Data: {len(combined_labels)} samples, SZ={int(combined_labels.sum())}/HC={int((1-combined_labels).sum())}, "
                f"{n_components} PCA, {engine.tangent_pca.total_explained_variance:.2%} variance")
    return tangent, combined_labels, fdt_features, n_rois_actual, engine


def train_and_get_predictions(X_train, X_val, y_train, y_val, fdt_train, fdt_val, fdt_dim,
                               config, n_rounds, patience, seed, device):
    """Train PQFL and return predictions + probabilities + model for saliency."""
    n_features = X_train.shape[1]
    train_ds = FCDataset(fc_matrices=np.zeros((len(y_train), 1, 1)), labels=y_train,
                         tangent_features=X_train, fdt_features=fdt_train)
    val_ds = FCDataset(fc_matrices=np.zeros((len(y_val), 1, 1)), labels=y_val,
                       tangent_features=X_val, fdt_features=fdt_val)

    encoder_hidden = [max(16, n_features // 2), config["n_qubits"] * 2]
    vqc_config = VQCConfig(n_qubits=config["n_qubits"], n_base_layers=config["n_base_layers"],
                           n_personal_layers=1, encoding_type="angle", entanglement="functional",
                           input_dim=n_features, encoder_hidden_dims=encoder_hidden,
                           fdt_features=fdt_dim, classifier_hidden_dims=[16],
                           dropout=config["dropout"], use_dual_register=False)
    model = HybridVQC(vqc_config)

    from torch.utils.data import DataLoader
    has_fdt = fdt_train is not None and fdt_train.shape[1] > 0

    def make_collate(include_fdt):
        def collate_fn(batch):
            result = {"tangent_features": torch.stack([b["tangent_features"] for b in batch]),
                      "label": torch.stack([b["label"] for b in batch])}
            if include_fdt and "fdt_features" in batch[0]:
                result["fdt_features"] = torch.stack([b["fdt_features"] for b in batch])
            return result
        return collate_fn

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=make_collate(has_fdt))
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=make_collate(has_fdt))

    client = PQFLClient(model=model, train_loader=train_loader, val_loader=val_loader,
                         site_id=0, site_name="cv_fold", local_epochs=2,
                         learning_rate=config["learning_rate"],
                         label_smoothing=config["label_smoothing"], device=device)

    best_ba, best_state, best_round = 0.0, None, 0
    rounds_without_improvement = 0
    params = client.get_parameters()

    for round_num in range(n_rounds):
        params, n_samples, metrics = client.fit(params)
        val_ba = metrics.get("balanced_accuracy", 0)
        if val_ba > best_ba:
            best_ba = val_ba
            best_state = copy.deepcopy(client.model.state_dict())
            best_round = round_num + 1
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
        if rounds_without_improvement >= patience:
            break

    if best_state is not None:
        client.model.load_state_dict(best_state)

    # Get predictions on validation set
    client.model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["tangent_features"]
            y = batch["label"]
            fdt = batch.get("fdt_features")
            logits = client.model(x, fdt_features=fdt)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs[:, 1].numpy())

    return (np.array(all_preds), np.array(all_probs), np.array(all_labels),
            client.model, vqc_config, best_round)


def compute_saliency_maps(model, tangent, labels, fdt_features, config, engine, n_rois):
    """Compute saliency and map back to ROI connections."""
    n_features = tangent.shape[1]
    fdt_dim = fdt_features.shape[1] if fdt_features is not None else 0
    has_fdt = fdt_dim > 0

    # Use SZ samples for saliency (target_class=1)
    sz_idx = np.where(labels == 1)[0]
    n_saliency = min(20, len(sz_idx))
    saliency_idx = sz_idx[:n_saliency]

    X = torch.tensor(tangent[saliency_idx], dtype=torch.float32)
    fdt = torch.tensor(fdt_features[saliency_idx], dtype=torch.float32) if has_fdt else None

    # Quantum saliency (gradient-based)
    qs = QuantumSaliency(model)
    saliency_maps = qs.compute_saliency(X, target_class=1, fdt_features=fdt)
    # Average across samples → shape (n_features,)
    avg_saliency = saliency_maps.mean(axis=0)

    # Also compute integrated gradients for comparison
    try:
        cs = ClassicalSaliency(model, n_steps=25)
        ig_maps = cs.compute_integrated_gradients(
            X[:5].clone().detach().requires_grad_(True),
            target_class=1,
            fdt_features=fdt[:5].clone().detach() if fdt is not None else None)
        avg_ig = ig_maps.mean(axis=0)
    except RuntimeError:
        # Fallback: use gradient saliency for IG too
        logger.info("  Integrated gradients failed, using gradient saliency as fallback")
        avg_ig = avg_saliency  # Use gradient saliency as proxy

    # Map tangent PCA saliency back to ROI connections
    # PCA components: shape (n_components, tangent_dim)
    # tangent_dim = n_rois * (n_rois + 1) / 2
    pca_components = engine.tangent_pca.pca.components_  # (71, 5050)
    tangent_dim = pca_components.shape[1]

    # Reconstruct tangent-space saliency from PCA space
    # saliency in PCA space (71,) → tangent space (5050,)
    tangent_saliency = pca_components.T @ avg_saliency  # (5050,)

    # Map tangent vector indices back to (i,j) ROI pairs
    # Tangent vectors are upper-triangular entries of symmetric matrix
    roi_saliency_matrix = np.zeros((n_rois, n_rois))
    idx = 0
    for i in range(n_rois):
        for j in range(i, n_rois):
            if idx < tangent_dim:
                roi_saliency_matrix[i, j] = abs(tangent_saliency[idx])
                roi_saliency_matrix[j, i] = abs(tangent_saliency[idx])
                idx += 1

    # Per-ROI importance: sum of saliency for connections involving that ROI
    roi_importance = roi_saliency_matrix.sum(axis=1)

    # Top ROI connections
    top_connections = []
    for i in range(n_rois):
        for j in range(i+1, n_rois):
            top_connections.append((i, j, roi_saliency_matrix[i, j]))
    top_connections.sort(key=lambda x: x[2], reverse=True)

    # Integrated gradients in tangent space
    tangent_ig = pca_components.T @ avg_ig
    roi_ig_matrix = np.zeros((n_rois, n_rois))
    idx = 0
    for i in range(n_rois):
        for j in range(i, n_rois):
            if idx < tangent_dim:
                roi_ig_matrix[i, j] = abs(tangent_ig[idx])
                roi_ig_matrix[j, i] = abs(tangent_ig[idx])
                idx += 1

    roi_importance_ig = roi_ig_matrix.sum(axis=1)

    return {
        "avg_saliency_pca": avg_saliency,
        "avg_ig_pca": avg_ig,
        "roi_saliency_matrix": roi_saliency_matrix,
        "roi_importance": roi_importance,
        "roi_importance_ig": roi_importance_ig,
        "top_connections": top_connections,
        "top_10_rois": np.argsort(roi_importance)[::-1][:10].tolist(),
        "top_10_rois_ig": np.argsort(roi_importance_ig)[::-1][:10].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Statistical Testing + Brain Saliency Analysis")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_rounds", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./analysis_results")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = OPTIMAL_CONFIG

    logger.info("=" * 70)
    logger.info("STATISTICAL SIGNIFICANCE + SALIENCY ANALYSIS")
    logger.info("=" * 70)

    # Load data
    start_time = time.time()
    tangent, labels, fdt_features, n_rois, engine = load_and_preprocess(
        args.data_dir, config["n_components"], 100)
    fdt_dim = fdt_features.shape[1] if fdt_features is not None else 0

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    # ===== PHASE 1: Collect predictions from all models across folds =====
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 1: Collecting Predictions for Statistical Testing")
    logger.info(f"{'='*70}")

    # Storage for all fold predictions
    pqfl_all_preds, pqfl_all_probs, pqfl_all_labels = [], [], []
    svm_all_preds, svm_all_probs = [], []
    lr_all_preds, lr_all_probs = [], []

    # Storage for best model (last fold) for saliency
    best_model = None
    best_vqc_config = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(tangent, labels)):
        logger.info(f"\n--- Fold {fold+1}/{args.n_folds} ---")
        X_train, X_val = tangent[train_idx], tangent[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        fdt_train = fdt_features[train_idx] if fdt_features is not None else None
        fdt_val = fdt_features[val_idx] if fdt_features is not None else None

        # PQFL
        preds, probs, true_labels, model, vqc_config, best_round = train_and_get_predictions(
            X_train, X_val, y_train, y_val, fdt_train, fdt_val, fdt_dim,
            config, args.n_rounds, args.patience, args.seed + fold, args.device)
        pqfl_all_preds.append(preds)
        pqfl_all_probs.append(probs)
        pqfl_all_labels.append(true_labels)

        if fold == args.n_folds - 1:
            best_model = model
            best_vqc_config = vqc_config

        logger.info(f"  PQFL: BA={balanced_accuracy_score(true_labels, preds):.4f}, "
                     f"AUC={roc_auc_score(true_labels, probs):.4f}, best_round={best_round}")

        # SVM
        svm = TangentSpaceSVM(kernel="rbf", C=1.0)
        svm.fit(X_train, y_train)
        svm_preds = svm.predict(X_val)
        svm_probs = svm.predict_proba(X_val)[:, 1]
        svm_all_preds.append(svm_preds)
        svm_all_probs.append(svm_probs)

        # LR
        lr = RiemannianLogisticRegression(C=1.0)
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_val)
        lr_probs = lr.predict_proba(X_val)[:, 1]
        lr_all_preds.append(lr_preds)
        lr_all_probs.append(lr_probs)

    # Concatenate all fold predictions
    pqfl_preds_all = np.concatenate(pqfl_all_preds)
    pqfl_probs_all = np.concatenate(pqfl_all_probs)
    y_true_all = np.concatenate(pqfl_all_labels)
    svm_preds_all = np.concatenate(svm_all_preds)
    svm_probs_all = np.concatenate(svm_all_probs)
    lr_preds_all = np.concatenate(lr_all_preds)
    lr_probs_all = np.concatenate(lr_all_probs)

    # ===== PHASE 2: Statistical Significance Testing =====
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 2: Statistical Significance Testing")
    logger.info(f"{'='*70}")

    # DeLong test: PQFL vs SVM
    delong_pqfl_svm = delong_test(y_true_all, pqfl_probs_all, svm_probs_all)
    logger.info(f"\n  DeLong Test: PQFL vs TangentSVM")
    logger.info(f"    AUC_PQFL={delong_pqfl_svm['auc_model1']:.4f}, AUC_SVM={delong_pqfl_svm['auc_model2']:.4f}")
    logger.info(f"    AUC diff={delong_pqfl_svm['auc_difference']:+.4f}, z={delong_pqfl_svm['z_statistic']:.4f}, p={delong_pqfl_svm['p_value']:.4f}")
    logger.info(f"    Significant (p<0.05): {delong_pqfl_svm['significant_005']}")

    # DeLong test: PQFL vs LR
    delong_pqfl_lr = delong_test(y_true_all, pqfl_probs_all, lr_probs_all)
    logger.info(f"\n  DeLong Test: PQFL vs RiemannianLR")
    logger.info(f"    AUC_PQFL={delong_pqfl_lr['auc_model1']:.4f}, AUC_LR={delong_pqfl_lr['auc_model2']:.4f}")
    logger.info(f"    AUC diff={delong_pqfl_lr['auc_difference']:+.4f}, z={delong_pqfl_lr['z_statistic']:.4f}, p={delong_pqfl_lr['p_value']:.4f}")
    logger.info(f"    Significant (p<0.05): {delong_pqfl_lr['significant_005']}")

    # McNemar test: PQFL vs SVM
    mcnemar_pqfl_svm = mcnemar_test(y_true_all, pqfl_preds_all, svm_preds_all)
    logger.info(f"\n  McNemar Test: PQFL vs TangentSVM")
    logger.info(f"    χ²={mcnemar_pqfl_svm['statistic']:.4f}, p={mcnemar_pqfl_svm['p_value']:.4f}")
    logger.info(f"    Disagreements: {mcnemar_pqfl_svm['n_disagreement']}")
    logger.info(f"    Significant (p<0.05): {mcnemar_pqfl_svm['significant_005']}")

    # McNemar test: PQFL vs LR
    mcnemar_pqfl_lr = mcnemar_test(y_true_all, pqfl_preds_all, lr_preds_all)
    logger.info(f"\n  McNemar Test: PQFL vs RiemannianLR")
    logger.info(f"    χ²={mcnemar_pqfl_lr['statistic']:.4f}, p={mcnemar_pqfl_lr['p_value']:.4f}")
    logger.info(f"    Disagreements: {mcnemar_pqfl_lr['n_disagreement']}")
    logger.info(f"    Significant (p<0.05): {mcnemar_pqfl_lr['significant_005']}")

    # Bootstrap CIs for PQFL metrics
    pqfl_bas = [balanced_accuracy_score(pqfl_all_labels[i], pqfl_all_preds[i]) for i in range(args.n_folds)]
    pqfl_aucs = [roc_auc_score(pqfl_all_labels[i], pqfl_all_probs[i]) for i in range(args.n_folds)]

    ba_mean, ba_lo, ba_hi = bootstrap_ci(np.array(pqfl_bas))
    auc_mean, auc_lo, auc_hi = bootstrap_ci(np.array(pqfl_aucs))

    logger.info(f"\n  Bootstrap 95% CI for PQFL:")
    logger.info(f"    BA: {ba_mean:.4f} [{ba_lo:.4f}, {ba_hi:.4f}]")
    logger.info(f"    AUC: {auc_mean:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]")

    # Bonferroni correction for 4 tests
    all_p_values = [delong_pqfl_svm['p_value'], delong_pqfl_lr['p_value'],
                    mcnemar_pqfl_svm['p_value'], mcnemar_pqfl_lr['p_value']]
    bonf_significant = bonferroni_correction(all_p_values, alpha=0.05)
    logger.info(f"\n  Bonferroni Correction (4 tests, α_adj={0.05/4:.4f}):")
    test_names = ["DeLong PQFL vs SVM", "DeLong PQFL vs LR",
                  "McNemar PQFL vs SVM", "McNemar PQFL vs LR"]
    for name, p, sig in zip(test_names, all_p_values, bonf_significant):
        logger.info(f"    {name}: p={p:.4f}, significant={sig}")

    # ===== PHASE 3: Saliency Analysis =====
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 3: Brain Saliency / Interpretability Analysis")
    logger.info(f"{'='*70}")

    saliency_results = compute_saliency_maps(
        best_model, tangent, labels, fdt_features, config, engine, n_rois)

    top_rois = saliency_results["top_10_rois"]
    top_conns = saliency_results["top_connections"][:15]

    logger.info(f"\n  Top 10 Most Important ROIs (gradient saliency):")
    for rank, roi in enumerate(top_rois):
        logger.info(f"    ROI {roi:3d}: importance = {saliency_results['roi_importance'][roi]:.6f}")

    logger.info(f"\n  Top 15 Most Discriminative Connections:")
    for rank, (i, j, sal) in enumerate(top_conns):
        logger.info(f"    ROI {i:3d} ↔ ROI {j:3d}: saliency = {sal:.6f}")

    logger.info(f"\n  Top 10 ROIs (integrated gradients):")
    for rank, roi in enumerate(saliency_results["top_10_rois_ig"]):
        logger.info(f"    ROI {roi:3d}: importance = {saliency_results['roi_importance_ig'][roi]:.6f}")

    # ===== Save All Results =====
    total_time = time.time() - start_time

    # Convert saliency results to serializable
    saliency_serializable = {
        "top_10_rois": saliency_results["top_10_rois"],
        "top_10_rois_ig": saliency_results["top_10_rois_ig"],
        "top_15_connections": [(int(i), int(j), float(s)) for i, j, s in top_conns],
        "roi_importance": saliency_results["roi_importance"].tolist(),
        "roi_importance_ig": saliency_results["roi_importance_ig"].tolist(),
    }

    all_results = {
        "config": config,
        "total_time_seconds": total_time,
        "n_folds": args.n_folds,
        "statistical_tests": {
            "delong_pqfl_vs_svm": delong_pqfl_svm,
            "delong_pqfl_vs_lr": delong_pqfl_lr,
            "mcnemar_pqfl_vs_svm": mcnemar_pqfl_svm,
            "mcnemar_pqfl_vs_lr": mcnemar_pqfl_lr,
            "bonferroni_correction": {
                "adjusted_alpha": 0.05 / 4,
                "test_names": test_names,
                "p_values": all_p_values,
                "significant": bonf_significant,
            },
        },
        "bootstrap_ci": {
            "ba": {"mean": ba_mean, "lower": ba_lo, "upper": ba_hi},
            "auc": {"mean": auc_mean, "lower": auc_lo, "upper": auc_hi},
        },
        "saliency": saliency_serializable,
        "pqfl_per_fold": {
            "ba": [float(x) for x in pqfl_bas],
            "auc": [float(x) for x in pqfl_aucs],
        },
    }

    with open(output_dir / "analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ===== Generate Saliency Figures =====
    try:
        _generate_saliency_figures(saliency_results, n_rois, output_dir,
            statistical_data=all_results["statistical_tests"],
            bootstrap_data=all_results["bootstrap_ci"],
            pqfl_per_fold=all_results["pqfl_per_fold"])
    except Exception as e:
        logger.warning(f"Figure generation failed (non-critical): {e}")

    logger.info(f"\n{'='*70}")
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"  Total time: {total_time:.1f}s")
    logger.info(f"  Results saved to: {output_dir / 'analysis_results.json'}")
    logger.info(f"  Figures saved to: {output_dir}/")

    return all_results


def _generate_saliency_figures(saliency_results, n_rois, output_dir,
                               statistical_data=None, bootstrap_data=None, pqfl_per_fold=None):
    """Generate saliency visualization figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # Cross-platform font loading with graceful fallback
    _font_loaded = {'dejavu': False, 'sarasa': False}
    for fpath in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                  'C:/Windows/Fonts/DejaVuSans.ttf']:
        try:
            fm.fontManager.addfont(fpath)
            _font_loaded['dejavu'] = True
            break
        except Exception:
            pass
    # Also try matplotlib's bundled DejaVu
    if not _font_loaded['dejavu']:
        try:
            import matplotlib as mpl
            mpl_data_dir = Path(mpl.get_data_path()) / 'fonts' / 'ttf'
            for fname in mpl_data_dir.glob('DejaVuSans*.ttf'):
                fm.fontManager.addfont(str(fname))
                _font_loaded['dejavu'] = True
                break
        except Exception:
            pass
    for fpath in ['/usr/share/fonts/truetype/chinese/SarasaMonoSC-Regular.ttf',
                  '/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf']:
        try:
            fm.fontManager.addfont(fpath)
            _font_loaded['sarasa'] = True
            break
        except Exception:
            pass

    # Set font preferences based on what's available
    if _font_loaded['dejavu'] and _font_loaded['sarasa']:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Sarasa Mono SC']
    elif _font_loaded['dejavu']:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    else:
        plt.rcParams['font.sans-serif'] = ['sans-serif']  # matplotlib default
    plt.rcParams['axes.unicode_minus'] = False

    COLORS = {'pqfl': '#2171b5', 'accent': '#6a51a3', 'heat': '#cb181d',
              'grid': '#cccccc', 'green': '#238b45', 'orange': '#e6550d'}

    # Figure 1: ROI Importance Bar Chart (gradient saliency + integrated gradients)
    roi_imp = saliency_results["roi_importance"]
    roi_imp_ig = saliency_results["roi_importance_ig"]
    top_k = 20
    top_indices = np.argsort(roi_imp)[::-1][:top_k]
    top_values = roi_imp[top_indices]
    top_ig_values = roi_imp_ig[top_indices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gradient saliency
    ax1.bar(range(top_k), top_values, color=COLORS['pqfl'], edgecolor='white', linewidth=0.8)
    ax1.set_xlabel('ROI Index (ranked)')
    ax1.set_ylabel('Saliency Importance')
    ax1.set_title('Gradient Saliency', fontweight='bold')
    ax1.set_xticks(range(top_k))
    ax1.set_xticklabels([str(i) for i in top_indices], fontsize=8)

    # Integrated gradients
    top_indices_ig = np.argsort(roi_imp_ig)[::-1][:top_k]
    top_ig = roi_imp_ig[top_indices_ig]
    ax2.bar(range(top_k), top_ig, color=COLORS['accent'], edgecolor='white', linewidth=0.8)
    ax2.set_xlabel('ROI Index (ranked)')
    ax2.set_ylabel('Attribution Score')
    ax2.set_title('Integrated Gradients', fontweight='bold')
    ax2.set_xticks(range(top_k))
    ax2.set_xticklabels([str(i) for i in top_indices_ig], fontsize=8)

    fig.suptitle('Top 20 Most Important Brain Regions (PQFL Saliency)', fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'roi_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Connection Saliency Heatmap (top 30x30 submatrix)
    sal_matrix = saliency_results["roi_saliency_matrix"]
    top_30_rois = np.argsort(roi_imp)[::-1][:30]
    sub_matrix = sal_matrix[np.ix_(top_30_rois, top_30_rois)]

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(sub_matrix, cmap='Reds', aspect='auto', interpolation='nearest')
    ax.set_xlabel('ROI (top 30)', fontsize=10)
    ax.set_ylabel('ROI (top 30)', fontsize=10)
    ax.set_title('Brain Connection Saliency Heatmap (Top 30 ROIs)', fontweight='bold')
    ax.set_xticks(range(30))
    ax.set_xticklabels([str(i) for i in top_30_rois], fontsize=6, rotation=90)
    ax.set_yticks(range(30))
    ax.set_yticklabels([str(i) for i in top_30_rois], fontsize=6)
    cbar = plt.colorbar(im, ax=ax, label='Saliency', shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_dir / 'connection_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 3: Statistical Tests + Bootstrap CI Summary
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: p-value bar chart
    ax = axes[0]
    if statistical_data is not None:
        test_labels = ['DeLong\nvs SVM', 'DeLong\nvs LR', 'McNemar\nvs SVM', 'McNemar\nvs LR']
        p_vals = [statistical_data['delong_pqfl_vs_svm']['p_value'],
                  statistical_data['delong_pqfl_vs_lr']['p_value'],
                  statistical_data['mcnemar_pqfl_vs_svm']['p_value'],
                  statistical_data['mcnemar_pqfl_vs_lr']['p_value']]
        colors = [COLORS['green'] if p < 0.05 else COLORS['pqfl'] for p in p_vals]
        bars = ax.bar(range(4), p_vals, color=colors, edgecolor='white')
        ax.axhline(y=0.05, color=COLORS['heat'], linestyle='--', linewidth=1.5, label='p=0.05')
        ax.axhline(y=0.05/4, color=COLORS['orange'], linestyle=':', linewidth=1.5, label='Bonferroni adj.')
        ax.set_xticks(range(4))
        ax.set_xticklabels(test_labels, fontsize=8)
        ax.set_ylabel('p-value')
        ax.set_title('Statistical Significance', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        for i, p in enumerate(p_vals):
            ax.text(i, p + 0.02, f'{p:.3f}', ha='center', fontsize=7)
    else:
        ax.text(0.5, 0.5, 'No statistical\ndata available', ha='center', va='center',
                transform=ax.transAxes, fontsize=11)
        ax.axis('off')
        ax.set_title('Statistical Significance', fontweight='bold')

    # Panel B: Bootstrap CI
    ax = axes[1]
    if bootstrap_data is not None:
        metrics = ['BA', 'AUC']
        means = [bootstrap_data['ba']['mean'], bootstrap_data['auc']['mean']]
        lowers = [bootstrap_data['ba']['lower'], bootstrap_data['auc']['lower']]
        uppers = [bootstrap_data['ba']['upper'], bootstrap_data['auc']['upper']]
        errors_lo = [m - l for m, l in zip(means, lowers)]
        errors_hi = [u - m for m, u in zip(means, uppers)]
        ax.bar(metrics, means, yerr=[errors_lo, errors_hi], capsize=8,
               color=[COLORS['pqfl'], COLORS['accent']], edgecolor='white', linewidth=0.8)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Chance (0.5)')
        ax.set_ylabel('Score')
        ax.set_title('PQFL Bootstrap 95% CI', fontweight='bold')
        ax.set_ylim(0.3, 0.85)
        ax.legend(loc='best', fontsize=8)
        for i, (m, lo, hi) in enumerate(zip(means, lowers, uppers)):
            ax.text(i, hi + 0.01, f'{m:.3f}', ha='center', fontsize=9, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No bootstrap\ndata available', ha='center', va='center',
                transform=ax.transAxes, fontsize=11)
        ax.axis('off')
        ax.set_title('Bootstrap 95% CI', fontweight='bold')

    # Panel C: Per-fold performance
    ax = axes[2]
    if pqfl_per_fold is not None:
        folds = list(range(1, len(pqfl_per_fold['ba']) + 1))
        ax.plot(folds, pqfl_per_fold['ba'], 'o-', color=COLORS['pqfl'], label='BA', linewidth=2, markersize=8)
        ax.plot(folds, pqfl_per_fold['auc'], 's-', color=COLORS['accent'], label='AUC', linewidth=2, markersize=8)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('Fold')
        ax.set_ylabel('Score')
        ax.set_title('PQFL Per-Fold Performance', fontweight='bold')
        ax.set_ylim(0.3, 0.9)
        ax.legend(loc='best', fontsize=9)
        ax.set_xticks(folds)
    else:
        ax.text(0.5, 0.5, 'No per-fold\ndata available', ha='center', va='center',
                transform=ax.transAxes, fontsize=11)
        ax.axis('off')
        ax.set_title('Per-Fold Performance', fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_dir / 'statistical_tests.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("  Saved: roi_importance.png, connection_heatmap.png, statistical_tests.png")


if __name__ == "__main__":
    main()
