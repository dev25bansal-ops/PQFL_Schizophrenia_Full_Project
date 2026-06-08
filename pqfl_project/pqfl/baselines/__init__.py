"""Classical baseline classifiers for comparison with PQFL.

Implements established methods for SPD matrix classification:
- Tangent-space SVM (standard baseline)
- MDM (Minimum Distance to Mean) classifier
- Riemannian logistic regression
"""

from .classical import (
    TangentSpaceSVM,
    MDMClassifier,
    RiemannianLogisticRegression,
)

__all__ = [
    "TangentSpaceSVM",
    "MDMClassifier",
    "RiemannianLogisticRegression",
]
