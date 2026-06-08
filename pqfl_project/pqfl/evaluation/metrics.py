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
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels (0=HC, 1=SZ), shape (n_samples,).
        y_pred: Predicted labels, shape (n_samples,).
        y_prob: Predicted probabilities for positive class, shape (n_samples,).
        prefix: Optional prefix for metric names.
    
    Returns:
        Dictionary of metric name → value.
    """
    metrics = {}
    p = prefix
    
    # Core metrics
    metrics[f"{p}accuracy"] = accuracy_score(y_true, y_pred)
    metrics[f"{p}balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics[f"{p}f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics[f"{p}precision"] = precision_score(y_true, y_pred, zero_division=0)
    
    # Sensitivity (recall for SZ class) = True Positive Rate
    metrics[f"{p}sensitivity"] = recall_score(y_true, y_pred, zero_division=0)
    
    # Specificity = True Negative Rate
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        metrics[f"{p}specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        metrics[f"{p}specificity"] = 0.0
    
    # AUC-ROC (requires probabilities)
    if y_prob is not None:
        try:
            metrics[f"{p}auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics[f"{p}auc_roc"] = 0.5  # Constant prediction
    
    # Class distribution
    n_total = len(y_true)
    n_sz = y_true.sum()
    n_hc = n_total - n_sz
    metrics[f"{p}n_total"] = n_total
    metrics[f"{p}n_sz"] = int(n_sz)
    metrics[f"{p}n_hc"] = int(n_hc)
    
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
