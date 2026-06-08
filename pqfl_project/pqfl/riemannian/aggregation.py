"""Riemannian-aware federated aggregation strategies.

Standard FedAvg computes arithmetic mean of parameters, which violates
the geometric structure of SPD manifold data. This module provides
geometry-aware aggregation strategies specifically for SPD parameters.

Based on FedSPDnet (arXiv:2604.22494):
- ProjAvg: Arithmetic mean + Stiefel projection via polar decomposition
- RLAvg: Tangent space Euclidean mean + retraction
- FréchetAvg: Fréchet mean for SPD matrices (recommended)
"""

import numpy as np
import scipy.linalg as la
from typing import List, Dict, Optional, Union
import logging

from .spd_utils import spd_frechet_mean, _log_map, _exp_map, regularize_spd

logger = logging.getLogger(__name__)


class RiemannianAggregator:
    """Riemannian-aware aggregation for SPD parameters in federated learning.
    
    For the PQFL architecture, VQC parameters are classical (not on SPD manifold),
    so standard FedAvg is appropriate for those. However, for any SPD matrix
    aggregation (e.g., shared covariance estimates, tangent space references),
    this aggregator provides geometry-correct aggregation.
    
    Args:
        strategy: Aggregation strategy:
            - "frechet": Fréchet mean (recommended for SPD matrices)
            - "proj_avg": Projected averaging (ProjAvg from FedSPDnet)
            - "rl_avg": Retraction-lifting averaging (RLAvg from FedSPDnet)
            - "log_euclidean": Log-Euclidean mean (fast approximation)
    """
    
    def __init__(self, strategy: str = "frechet"):
        self.strategy = strategy
    
    def aggregate(
        self,
        matrices: List[np.ndarray],
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Aggregate SPD matrices using the specified strategy.
        
        Args:
            matrices: List of SPD matrices to aggregate.
            weights: Optional per-matrix weights (proportional to sample sizes).
        
        Returns:
            Aggregated SPD matrix.
        """
        if len(matrices) == 0:
            raise ValueError("Cannot aggregate empty list")
        if len(matrices) == 1:
            return matrices[0].copy()
        
        if weights is not None:
            weights = np.array(weights, dtype=float)
            weights = weights / weights.sum()
        
        if self.strategy == "frechet":
            return self._frechet_aggregate(matrices, weights)
        elif self.strategy == "proj_avg":
            return self._proj_avg(matrices, weights)
        elif self.strategy == "rl_avg":
            return self._rl_avg(matrices, weights)
        elif self.strategy == "log_euclidean":
            return self._log_euclidean_aggregate(matrices, weights)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _frechet_aggregate(
        self,
        matrices: List[np.ndarray],
        weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """Weighted Fréchet mean aggregation.
        
        Iteratively computes the weighted Fréchet mean on the SPD manifold.
        This is the gold standard for SPD matrix aggregation.
        """
        if weights is None:
            return spd_frechet_mean(matrices)
        
        # Weighted Fréchet mean via iterative log-Euclidean
        mean = matrices[0].copy()
        for _ in range(20):
            tangent_vecs = [_log_map(m, mean) for m in matrices]
            # Weighted average in tangent space
            avg_tangent = np.zeros_like(mean)
            for w, tv in zip(weights, tangent_vecs):
                avg_tangent += w * tv
            mean = _exp_map(avg_tangent, mean)
            mean = 0.5 * (mean + mean.T)
        
        return mean
    
    def _proj_avg(
        self,
        matrices: List[np.ndarray],
        weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """ProjAvg: Arithmetic mean + Stiefel projection via polar decomposition.
        
        From FedSPDnet (arXiv:2604.22494):
        1. Compute arithmetic mean of matrices
        2. Project back to SPD manifold via polar decomposition (SVD)
        """
        if weights is None:
            weights = np.ones(len(matrices)) / len(matrices)
        
        # Arithmetic mean
        avg = np.zeros_like(matrices[0])
        for w, m in zip(weights, matrices):
            avg += w * m
        avg = 0.5 * (avg + avg.T)
        
        # Project via polar decomposition
        try:
            U, S, Vh = la.svd(avg)
            # Ensure positive eigenvalues
            S = np.maximum(S, 1e-10)
            projected = U @ np.diag(S) @ Vh
            return 0.5 * (projected + projected.T)
        except la.LinAlgError:
            logger.warning("SVD failed in ProjAvg, falling back to regularization")
            return regularize_spd(avg, lambda_val=1e-3)
    
    def _rl_avg(
        self,
        matrices: List[np.ndarray],
        weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """RLAvg: Retraction-lifting averaging.
        
        From FedSPDnet (arXiv:2604.22494):
        1. Map each local matrix to tangent space at reference point
        2. Compute Euclidean mean in tangent space
        3. Retract back to SPD manifold
        """
        # Use first matrix as reference
        ref = matrices[0]
        
        if weights is None:
            weights = np.ones(len(matrices)) / len(matrices)
        
        # Map to tangent space and compute weighted mean
        avg_tangent = np.zeros_like(ref)
        for w, m in zip(weights, matrices):
            tangent = _log_map(m, ref)
            avg_tangent += w * tangent
        
        # Retract
        result = _exp_map(avg_tangent, ref)
        return 0.5 * (result + result.T)
    
    def _log_euclidean_aggregate(
        self,
        matrices: List[np.ndarray],
        weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """Log-Euclidean mean: fast approximation of Fréchet mean.
        
        Computes: expm(weighted_mean(logm(matrices)))
        
        Faster than full Fréchet mean but less geometrically accurate.
        Good for initial estimates or when speed is critical.
        """
        if weights is None:
            weights = np.ones(len(matrices)) / len(matrices)
        
        # Compute matrix logs
        log_matrices = []
        for m in matrices:
            m_sym = 0.5 * (m + m.T)
            try:
                log_m = la.logm(m_sym)
                log_matrices.append(log_m)
            except la.LinAlgError:
                logger.warning("logm failed, using regularized matrix")
                log_m = la.logm(regularize_spd(m_sym))
                log_matrices.append(log_m)
        
        # Weighted average in log domain
        avg_log = np.zeros_like(log_matrices[0])
        for w, lm in zip(weights, log_matrices):
            avg_log += w * lm
        avg_log = 0.5 * (avg_log + avg_log.T)
        
        # Exponential map
        result = la.expm(avg_log)
        return 0.5 * (result + result.T)
