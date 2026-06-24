"""Classification metrics for schizophrenia fMRI analysis.

Primary metric: Balanced Accuracy (BA) to address class imbalance (~47% SZ / 53% HC).
Additional metrics: AUC-ROC, F1, Sensitivity, Specificity.

Clinical priority: maximize Specificity (minimize false SZ diagnoses).
Target: exceed 80% balanced accuracy in federated cross-site validation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    prefix: str = "",
    n_classes: int = 2,
    class_names: Optional[list] = None,
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.

    Supports both binary (n_classes=2) and multi-class (n_classes>=3) settings.

    Args:
        y_true: True labels, shape (n_samples,).
            Binary: 0=HC, 1=SZ.
            3-class: 0=SZ, 1=HC, 2=Other.
            4-class: 0=SZ, 1=HC, 2=Other, 3=BP.
        y_pred: Predicted labels, shape (n_samples,).
        y_prob: Predicted probabilities, shape (n_samples, n_classes) for multi-class
                or (n_samples,) for binary positive class.
        prefix: Optional prefix for metric names.
        n_classes: Number of classes (2, 3, or 4).
        class_names: Optional list of class names for per-class metrics.

    Returns:
        Dictionary of metric name → value.
    """
    metrics = {}
    p = prefix

    if class_names is None:
        if n_classes == 2:
            class_names = ["HC", "SZ"]
        elif n_classes == 3:
            class_names = ["SZ", "HC", "Other"]
        elif n_classes == 4:
            class_names = ["SZ", "HC", "Other", "BP"]
        else:
            class_names = [f"class_{i}" for i in range(n_classes)]

    # Core metrics (work for both binary and multi-class)
    metrics[f"{p}accuracy"] = accuracy_score(y_true, y_pred)
    metrics[f"{p}balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    if n_classes == 2:
        # Binary-specific metrics (legacy behavior)
        metrics[f"{p}f1"] = f1_score(y_true, y_pred, zero_division=0)
        metrics[f"{p}precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics[f"{p}sensitivity"] = recall_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            metrics[f"{p}specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            metrics[f"{p}specificity"] = 0.0

        if y_prob is not None:
            try:
                y_prob_1d = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                metrics[f"{p}auc_roc"] = roc_auc_score(y_true, y_prob_1d)
            except (ValueError, IndexError):
                metrics[f"{p}auc_roc"] = 0.5

        n_total = len(y_true)
        n_sz = int(y_true.sum())
        n_hc = n_total - n_sz
        metrics[f"{p}n_total"] = n_total
        metrics[f"{p}n_sz"] = n_sz
        metrics[f"{p}n_hc"] = n_hc

    else:
        # Multi-class metrics
        metrics[f"{p}f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        metrics[f"{p}f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics[f"{p}precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)

        # Per-class recall (= sensitivity for that class)
        per_class_recall = recall_score(y_true, y_pred, average=None,
                                        labels=list(range(n_classes)), zero_division=0)
        for i, name in enumerate(class_names):
            metrics[f"{p}sensitivity_{name}"] = float(per_class_recall[i])

        # Per-class specificity
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        for i, name in enumerate(class_names):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            metrics[f"{p}specificity_{name}"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # Multi-class AUC (macro OvR) — handles missing classes in LOSO CV
        if y_prob is not None and y_prob.ndim == 2:
            present_classes = np.unique(y_true)
            if len(present_classes) > 1:
                try:
                    # Compute per-class AUC only for classes present in y_true
                    auc_scores = []
                    for cls in present_classes:
                        if cls < y_prob.shape[1]:
                            y_binary = (y_true == cls).astype(int)
                            if y_binary.sum() > 0 and (1 - y_binary).sum() > 0:
                                auc = roc_auc_score(y_binary, y_prob[:, cls])
                                auc_scores.append(auc)
                    metrics[f"{p}auc_roc_macro"] = float(np.mean(auc_scores)) if auc_scores else 0.5
                except (ValueError, IndexError):
                    metrics[f"{p}auc_roc_macro"] = 0.5
            else:
                metrics[f"{p}auc_roc_macro"] = 0.5

        # Class distribution
        n_total = len(y_true)
        metrics[f"{p}n_total"] = n_total
        for i, name in enumerate(class_names):
            metrics[f"{p}n_{name}"] = int((y_true == i).sum())

        # Confusion matrix (flattened) for reporting
        metrics[f"{p}confusion_matrix"] = cm.tolist()

    return metrics


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute balanced accuracy.
    
    BA = (Sensitivity + Specificity) / 2
    
    This is the primary evaluation metric because it handles
    class imbalance (~47% SZ / 53% HC in our data).
    """
    return float(balanced_accuracy_score(y_true, y_pred))


def sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute sensitivity (recall for SZ class)."""
    return float(recall_score(y_true, y_pred, zero_division=0))


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute specificity (recall for HC class).
    
    Clinical priority: minimize false SZ diagnoses.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp = cm.ravel()[:2]
        return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return 0.0


def auc_roc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute AUC-ROC from probabilities."""
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return 0.5
