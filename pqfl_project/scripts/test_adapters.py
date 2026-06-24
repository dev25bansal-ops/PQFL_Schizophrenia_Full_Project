"""Smoke test for the dataset_adapters module.

Tests:
1. Kaggle FNC reconstruction (5460-dim vec → 105×105 matrix → 100×100 truncate)
2. Label remapping (3-class vs 4-class vs 2-class)
3. SPD regularization
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from pqfl.data.dataset_adapters import (
    _load_kaggle_fnc_to_matrix,
    _regularize_spd,
    remap_labels_for_n_classes,
    LABEL_SZ, LABEL_HC, LABEL_OTHER, LABEL_BP,
)


def test_kaggle_fnc_reconstruction():
    """Test that we can reconstruct a 105×105 symmetric matrix from 5460-dim vec."""
    # Simulate a Kaggle subject directory
    tmp_dir = Path("/tmp/kaggle_smoke_test")
    sub_dir = tmp_dir / "sub_test"
    sub_dir.mkdir(parents=True, exist_ok=True)

    # Build a synthetic 105×105 symmetric correlation matrix
    n_icn = 105
    rng = np.random.default_rng(42)
    A = rng.standard_normal((n_icn, n_icn))
    sym = (A + A.T) / 2
    # Make it a correlation matrix
    d = np.sqrt(np.diag(sym))
    corr = sym / d[:, None] / d[None, :]
    np.fill_diagonal(corr, 1.0)

    # Extract upper triangle (5460 elements) — this is what Kaggle stores
    iu = np.triu_indices(n_icn, k=1)
    fnc_vec = corr[iu]
    assert len(fnc_vec) == 5460, f"Expected 5460, got {len(fnc_vec)}"

    # Save as (5460, 1) array like Kaggle format
    np.save(sub_dir / "fnc.npy", fnc_vec.reshape(-1, 1))

    # Load using our adapter
    fc_100 = _load_kaggle_fnc_to_matrix(sub_dir, n_target_rois=100)
    assert fc_100 is not None, "Failed to load FNC"
    assert fc_100.shape == (100, 100), f"Expected (100,100), got {fc_100.shape}"

    # Check symmetric
    assert np.allclose(fc_100, fc_100.T, atol=1e-6), "Matrix not symmetric"

    # Check SPD (all eigenvalues > 0)
    eigvals = np.linalg.eigvalsh(fc_100)
    assert eigvals.min() > 0, f"Matrix not SPD: min eigenvalue = {eigvals.min()}"
    print(f"✓ Kaggle FNC reconstruction: shape={fc_100.shape}, "
          f"eigvals=[{eigvals.min():.4f}, {eigvals.max():.4f}]")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir)


def test_label_remapping():
    """Test label remapping for 2/3/4 class schemes."""
    # Create a fake label array with all 4 classes
    labels = np.array([
        LABEL_SZ, LABEL_HC, LABEL_OTHER, LABEL_BP,  # one of each
        LABEL_SZ, LABEL_HC, LABEL_OTHER, LABEL_BP,
        LABEL_SZ, LABEL_HC,
    ])

    # 4-class: no change
    remapped_4 = remap_labels_for_n_classes(labels, n_classes=4)
    assert np.array_equal(remapped_4, labels), "4-class remap should be identity"
    assert len(remapped_4) == 10
    print(f"✓ 4-class remap: {np.bincount(remapped_4, minlength=4)}")

    # 3-class: BP (3) → Other (2)
    remapped_3 = remap_labels_for_n_classes(labels, n_classes=3)
    assert (remapped_3 == LABEL_BP).sum() == 0, "BP should be gone"
    assert (remapped_3 == LABEL_OTHER).sum() == 4, "Should have 4 Other (2 orig + 2 BP)"
    print(f"✓ 3-class remap: {np.bincount(remapped_3, minlength=3)}")

    # 2-class: only SZ and HC kept (6 of 10)
    remapped_2 = remap_labels_for_n_classes(labels, n_classes=2)
    assert len(remapped_2) == 6, f"Expected 6, got {len(remapped_2)}"
    assert (remapped_2 == LABEL_SZ).sum() == 3
    assert (remapped_2 == LABEL_HC).sum() == 3
    print(f"✓ 2-class remap: {np.bincount(remapped_2, minlength=2)} (filtered to 6 samples)")


def test_spd_regularization():
    """Test SPD regularization."""
    # Build a non-SPD symmetric matrix
    A = np.array([[1.0, 2.0, 3.0],
                  [2.0, 1.0, 0.5],
                  [3.0, 0.5, 1.0]])
    # Check it's not SPD
    eigvals_before = np.linalg.eigvalsh(A)
    assert eigvals_before.min() < 0, "Test setup wrong — matrix should not be SPD"

    # Regularize
    A_reg = _regularize_spd(A, lambda_reg=1e-3)
    eigvals_after = np.linalg.eigvalsh(A_reg)
    assert eigvals_after.min() > 0, f"Regularization failed: min eigval = {eigvals_after.min()}"

    # Check symmetric
    assert np.allclose(A_reg, A_reg.T), "Regularized matrix not symmetric"
    print(f"✓ SPD regularization: eigvals {eigvals_before.min():.4f} → {eigvals_after.min():.4f}")


def test_metrics_multiclass():
    """Test that compute_classification_metrics handles multi-class."""
    from pqfl.evaluation.metrics import compute_classification_metrics

    # 3-class test
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 0, 1, 1, 2, 2, 1, 0, 1, 0])  # 7/10 correct
    y_prob = np.array([
        [0.7, 0.2, 0.1], [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1], [0.1, 0.7, 0.2],
        [0.1, 0.2, 0.7], [0.1, 0.1, 0.8],
        [0.3, 0.5, 0.2], [0.5, 0.4, 0.1],
        [0.2, 0.5, 0.3], [0.6, 0.3, 0.1],
    ])

    metrics = compute_classification_metrics(
        y_true, y_pred, y_prob,
        n_classes=3, class_names=["SZ", "HC", "Other"],
    )

    assert "balanced_accuracy" in metrics
    assert "f1_macro" in metrics
    assert "auc_roc_macro" in metrics
    assert "sensitivity_SZ" in metrics
    assert "sensitivity_HC" in metrics
    assert "sensitivity_Other" in metrics
    assert "confusion_matrix" in metrics
    print(f"✓ 3-class metrics: BA={metrics['balanced_accuracy']:.3f}, "
          f"F1_macro={metrics['f1_macro']:.3f}, "
          f"AUC_macro={metrics['auc_roc_macro']:.3f}")
    print(f"  Per-class sens: SZ={metrics['sensitivity_SZ']:.2f}, "
          f"HC={metrics['sensitivity_HC']:.2f}, "
          f"Other={metrics['sensitivity_Other']:.2f}")


if __name__ == "__main__":
    print("=" * 60)
    print("PQFL Phase 2 — Dataset Adapters Smoke Test")
    print("=" * 60)
    print()
    test_kaggle_fnc_reconstruction()
    test_label_remapping()
    test_spd_regularization()
    test_metrics_multiclass()
    print()
    print("✓ All smoke tests passed!")
