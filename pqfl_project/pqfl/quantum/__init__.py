"""Quantum module for RQFM encoding and VQC circuits.

Implements the Riemannian Quantum Feature Map (RQFM) and
Variational Quantum Classifier (VQC) using PennyLane.
"""

from .rqfm import RQFMFeatureMap, create_rqfm_circuit
from .vqc import HybridVQC, VQCConfig
from .simulator import QuantumSimulator
from .kernels import RQFMKernel

__all__ = [
    "RQFMFeatureMap",
    "create_rqfm_circuit",
    "HybridVQC",
    "VQCConfig",
    "QuantumSimulator",
    "RQFMKernel",
]
