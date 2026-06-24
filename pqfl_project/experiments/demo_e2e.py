#!/usr/bin/env python3
"""End-to-end demonstration of the PQFL pipeline.

This demo runs a complete (but small-scale) version of the PQFL pipeline
to verify all components work together. Uses synthetic data and small
quantum circuits for fast execution.

Pipeline:
1. Generate synthetic multi-site SPD data
2. Riemannian preprocessing (Fréchet mean, tangent PCA)
3. Quantum model creation (HybridVQC with RQFM)
4. Single-site training and evaluation
5. Classical baseline comparison

No external fMRI data required - all data is generated synthetically.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    print("=" * 70)
    print("PQFL End-to-End Demo: Federated QML for Schizophrenia fMRI")
    print("=" * 70)
    
    # Configuration (small-scale for demo)
    N_ROIS = 20         # Small for demo (100 for full)
    N_SAMPLES = 80      # Per site
    N_QUBITS = 4        # Small for demo (12 for full)
    N_COMPONENTS = 8    # Tangent PCA components
    N_SITES = 3         # Training sites
    SEED = 42
    
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # ========================================
    # Step 1: Generate Synthetic Data
    # ========================================
    print("\n[Step 1] Generating synthetic multi-site fMRI data...")
    
    from pqfl.data.site_partitioning import SitePartitioner
    from pqfl.data.dataset import MultiSiteDataset
    
    sites = {}
    for site_id in range(N_SITES):
        site = SitePartitioner.generate_synthetic_site(
            site_id=site_id,
            n_samples=N_SAMPLES,
            n_rois=N_ROIS,
            sz_ratio=0.45 + 0.1 * site_id,  # Varying SZ ratios
            signal_strength=0.2,
            seed=SEED + site_id,
        )
        sites[site_id] = site
        print(f"  Site {site.site_name}: {len(site)} subjects "
              f"({site.dataset.n_sz} SZ, {site.dataset.n_hc} HC)")
    
    multi_site_data = MultiSiteDataset(
        sites=sites,
        validation_site_ids={N_SITES - 1},
    )
    
    # ========================================
    # Step 2: Riemannian Preprocessing
    # ========================================
    print("\n[Step 2] Riemannian preprocessing (SPD → tangent space → PCA)...")
    
    from pqfl.riemannian.engine import RiemannianEngine
    
    engine = RiemannianEngine(
        n_rois=N_ROIS,
        n_components=N_COMPONENTS,
        regularization_lambda=1e-3,
    )
    
    # Process each site
    for site_id, site_ds in multi_site_data.sites.items():
        fc, labels = site_ds.dataset.get_data_for_riemannian()
        
        if site_id == 0:
            # Fit on first site
            tangent = engine.fit_transform(fc, site_id=site_id, return_tensor=False)
        else:
            tangent = engine.transform(fc, return_tensor=False)
        
        site_ds.dataset.set_tangent_features(tangent)
        print(f"  Site {site_id}: {fc.shape} → {tangent.shape} "
              f"({engine.tangent_pca.total_explained_variance:.1%} variance retained)")
    
    # ========================================
    # Step 3: Harmonization (Optional)
    # ========================================
    print("\n[Step 3] Tangent-space ComBat harmonization...")
    
    from pqfl.harmonization.combat import TangentSpaceCombat
    
    combat = TangentSpaceCombat(biological_covariates=["diagnosis"])
    
    # Combine for harmonization
    all_tangent = []
    all_labels = []
    all_site_ids = []
    for site_id in sorted(multi_site_data.sites.keys()):
        ds = multi_site_data.sites[site_id].dataset
        all_tangent.append(ds.tangent_features)
        all_labels.append(ds.labels)
        all_site_ids.extend([site_id] * len(ds.labels))
    
    combined_tangent = np.concatenate(all_tangent)
    combined_labels = np.concatenate(all_labels)
    site_labels = np.array(all_site_ids)
    
    harmonized = combat.harmonize(
        combined_tangent, site_labels,
        labels=combined_labels,
    )
    
    # Split back
    offset = 0
    for site_id in sorted(multi_site_data.sites.keys()):
        ds = multi_site_data.sites[site_id].dataset
        n = len(ds.labels)
        ds.set_tangent_features(harmonized[offset:offset + n])
        offset += n
    
    print("  Harmonization complete!")
    
    # ========================================
    # Step 4: Create Quantum Model
    # ========================================
    print("\n[Step 4] Creating HybridVQC model with RQFM...")
    
    from pqfl.quantum.vqc import HybridVQC, VQCConfig
    
    config = VQCConfig(
        n_qubits=N_QUBITS,
        n_base_layers=2,
        n_personal_layers=1,
        encoding_type="angle",
        entanglement="functional",
        input_dim=N_COMPONENTS,
        encoder_hidden_dims=[16, N_QUBITS * 2],
        fdt_features=0,
        classifier_hidden_dims=[16],
        dropout=0.2,
        use_dual_register=False,
    )
    
    model = HybridVQC(config)
    
    n_shared = model.count_shared_params()
    n_personal = model.count_personal_params()
    n_total = model.count_total_params()
    
    print(f"  Model: {N_QUBITS} qubits, {config.n_base_layers} base + "
          f"{config.n_personal_layers} personal layers")
    print(f"  Parameters: {n_shared} shared (federated) + "
          f"{n_personal} personal (local) = {n_total} total")
    
    # ========================================
    # Step 5: Single-Site Training
    # ========================================
    print("\n[Step 5] Training quantum model on Site 0...")
    
    import torch.nn as nn
    from torch.utils.data import DataLoader
    
    site_0 = multi_site_data.sites[0]
    train_ds, val_ds = site_0.dataset.split(train_ratio=0.8, seed=SEED)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda batch: {
            "tangent_features": torch.stack([b["tangent_features"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
        },
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=lambda batch: {
            "tangent_features": torch.stack([b["tangent_features"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
        },
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    n_epochs = 10
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            x = batch["tangent_features"]
            y = batch["label"]
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        
        train_acc = correct / total if total > 0 else 0
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}: "
                  f"Loss={total_loss/len(train_loader):.4f}, Acc={train_acc:.4f}")
    
    # ========================================
    # Step 6: Evaluation
    # ========================================
    print("\n[Step 6] Evaluating on validation set...")
    
    from pqfl.evaluation.metrics import compute_classification_metrics
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch["tangent_features"]
            y = batch["label"]
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())
    
    quantum_metrics = compute_classification_metrics(
        np.array(all_labels), np.array(all_preds)
    )
    
    print(f"  Quantum VQC Results:")
    print(f"    Balanced Accuracy: {quantum_metrics['balanced_accuracy']:.4f}")
    print(f"    Accuracy: {quantum_metrics['accuracy']:.4f}")
    print(f"    F1: {quantum_metrics['f1']:.4f}")
    print(f"    Sensitivity: {quantum_metrics['sensitivity']:.4f}")
    print(f"    Specificity: {quantum_metrics['specificity']:.4f}")
    
    # ========================================
    # Step 7: Classical Baselines
    # ========================================
    print("\n[Step 7] Running classical baselines...")
    
    from pqfl.baselines.classical import TangentSpaceSVM, RiemannianLogisticRegression
    
    # Prepare data for baselines
    train_tangent = train_ds.tangent_features
    train_labels = train_ds.labels
    val_tangent = val_ds.tangent_features
    val_labels = val_ds.labels
    
    # SVM
    svm = TangentSpaceSVM(kernel="rbf")
    svm.fit(train_tangent, train_labels)
    svm_preds = svm.predict(val_tangent)
    svm_metrics = compute_classification_metrics(val_labels, svm_preds)
    print(f"  Tangent-SVM: BA={svm_metrics['balanced_accuracy']:.4f}")
    
    # Logistic Regression
    lr = RiemannianLogisticRegression()
    lr.fit(train_tangent, train_labels)
    lr_preds = lr.predict(val_tangent)
    lr_metrics = compute_classification_metrics(val_labels, lr_preds)
    print(f"  Riemannian-LR: BA={lr_metrics['balanced_accuracy']:.4f}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Quantum VQC (RQFM):  BA = {quantum_metrics['balanced_accuracy']:.4f}")
    print(f"  Tangent-SVM:         BA = {svm_metrics['balanced_accuracy']:.4f}")
    print(f"  Riemannian-LR:       BA = {lr_metrics['balanced_accuracy']:.4f}")
    print(f"\n  Target: >80% balanced accuracy (full-scale system)")
    print(f"  Classical ceiling: 77.3% (from literature)")
    
    # ========================================
    # Step 8: Quantum Saliency
    # ========================================
    print("\n[Step 8] Computing quantum saliency map...")
    
    from pqfl.evaluation.saliency import QuantumSaliency
    
    saliency = QuantumSaliency(model)
    sample = torch.tensor(val_tangent[:4], dtype=torch.float32)
    saliency_map = saliency.compute_saliency(sample, target_class=1)
    print(f"  Saliency map shape: {saliency_map.shape}")
    print(f"  Top-3 important features: {np.argsort(saliency_map.mean(axis=0))[-3:]}")
    
    print("\n" + "=" * 70)
    print("Demo complete! All PQFL components verified.")
    print("=" * 70)


if __name__ == "__main__":
    import torch.nn as nn  # Needed for clip_grad_norm_
    main()
