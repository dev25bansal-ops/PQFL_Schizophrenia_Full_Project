"""Riemannian geometry module for SPD manifold operations.

Provides tools for processing functional connectivity matrices on the
Symmetric Positive Definite (SPD) manifold using the affine-invariant
Riemannian metric (AIRM).
"""

from .engine import RiemannianEngine
from .tangent_space import TangentSpaceProjector, TangentPCA
from .spd_utils import (
    ensure_spd,
    regularize_spd,
    validate_spd,
    nearest_spd,
    spd_frechet_mean,
    parallel_transport,
)
from .aggregation import RiemannianAggregator

__all__ = [
    "RiemannianEngine",
    "TangentSpaceProjector",
    "TangentPCA",
    "ensure_spd",
    "regularize_spd",
    "validate_spd",
    "nearest_spd",
    "spd_frechet_mean",
    "parallel_transport",
    "RiemannianAggregator",
]
