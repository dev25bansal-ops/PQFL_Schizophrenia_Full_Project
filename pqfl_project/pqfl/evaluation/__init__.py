"""Evaluation module for PQFL model assessment.

Provides classification metrics, statistical tests, and
quantum/classical saliency methods.
"""

from .metrics import (
    compute_classification_metrics,
    balanced_accuracy,
    sensitivity,
    specificity,
    auc_roc,
)
from .statistical_tests import delong_test, mcnemar_test, bootstrap_ci
from .saliency import QuantumSaliency, ClassicalSaliency

__all__ = [
    "compute_classification_metrics",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "auc_roc",
    "delong_test",
    "mcnemar_test",
    "bootstrap_ci",
    "QuantumSaliency",
    "ClassicalSaliency",
]
