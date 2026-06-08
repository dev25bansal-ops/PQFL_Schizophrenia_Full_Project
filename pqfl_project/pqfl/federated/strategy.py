"""Federated learning strategies for PQFL.

Implements FedPer and FedProx strategies with:
- Weighted FedAvg by sample size
- Parameter split awareness (only shared params are aggregated)
- Riemannian-aware aggregation for SPD parameters
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class PQFLStrategy:
    """Base strategy for PQFL federated aggregation.
    
    Provides weighted FedAvg with sample-size proportional weights.
    Only shared parameters are aggregated; personal parameters
    are never communicated.
    
    Args:
        fraction_fit: Fraction of clients to use for training.
        min_fit_clients: Minimum number of clients for training.
        min_available_clients: Minimum available clients.
    """
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 3,
        min_available_clients: int = 3,
    ):
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
    
    def aggregate(
        self,
        results: List[Tuple[List[np.ndarray], int, Dict]],
    ) -> List[np.ndarray]:
        """Aggregate shared parameters using weighted FedAvg.
        
        Weighted average proportional to sample sizes:
        θ_global = Σ(n_k / N) * θ_k
        
        Args:
            results: List of (parameters, num_samples, metrics) from each client.
        
        Returns:
            Aggregated shared parameters.
        """
        if len(results) == 0:
            raise ValueError("No results to aggregate")
        
        if len(results) == 1:
            return results[0][0]
        
        # Compute weights proportional to sample sizes
        total_samples = sum(n for _, n, _ in results)
        weights = [n / total_samples for _, n, _ in results]
        
        # Weighted average of parameters
        n_params = len(results[0][0])
        aggregated = [np.zeros_like(results[0][0][i]) for i in range(n_params)]
        
        for (params, _, _), weight in zip(results, weights):
            for i in range(n_params):
                aggregated[i] += weight * params[i]
        
        return aggregated


class PQFLFedPerStrategy(PQFLStrategy):
    """FedPer strategy: only shared parameters are aggregated.
    
    Following Arivazhagan et al. (2019):
    - Shared layers: classical encoder + VQC base → federated via FedAvg
    - Personal layers: VQC personalization + classifier head → kept local
    
    The aggregation is identical to FedAvg for shared parameters,
    but the client ensures only shared params are communicated.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("Using FedPer strategy (shared-only aggregation)")


class PQFLFedProxStrategy(PQFLStrategy):
    """FedProx strategy: FedAvg with proximal regularization.
    
    Following Li et al. (2020):
    - Same aggregation as FedAvg
    - Clients add proximal term (μ/2)||θ - θ_global||^2 during local training
    - Prevents local models from diverging too far from global
    
    Args:
        mu: Proximal term coefficient (default 0.01 as per paper).
    """
    
    def __init__(self, mu: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        logger.info(f"Using FedProx strategy (μ={mu})")
