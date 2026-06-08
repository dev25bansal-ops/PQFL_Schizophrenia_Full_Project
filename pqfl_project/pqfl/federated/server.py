"""PQFL Flower server for federated orchestration.

Implements the central server that:
1. Broadcasts shared parameters to all clients
2. Collects parameter updates from clients
3. Aggregates using the selected strategy (FedPer/FedProx)
4. Distributes updated global parameters
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from .strategy import PQFLStrategy, PQFLFedPerStrategy, PQFLFedProxStrategy
from .parameter_utils import ndarrays_to_parameters, parameters_to_ndarrays

logger = logging.getLogger(__name__)


class PQFLServer:
    """Central server for PQFL federated learning.
    
    Orchestrates the federated training process:
    1. Initialize global model parameters
    2. Each round: distribute → collect → aggregate → distribute
    3. Track global and per-site metrics
    
    Args:
        strategy: Aggregation strategy (FedPer, FedProx, or FedAvg).
        initial_parameters: Initial shared model parameters.
        n_rounds: Number of federated communication rounds.
        fraction_fit: Fraction of clients participating per round.
    """
    
    def __init__(
        self,
        strategy: Optional[PQFLStrategy] = None,
        initial_parameters: Optional[List[np.ndarray]] = None,
        n_rounds: int = 50,
        fraction_fit: float = 1.0,
    ):
        self.strategy = strategy or PQFLFedPerStrategy()
        self.current_parameters = initial_parameters
        self.n_rounds = n_rounds
        self.fraction_fit = fraction_fit
        
        # Tracking
        self._round_metrics: List[Dict] = []
        self._current_round = 0
    
    def initialize(self, initial_parameters: List[np.ndarray]) -> None:
        """Initialize global parameters.
        
        Args:
            initial_parameters: Initial shared model parameters.
        """
        self.current_parameters = [p.copy() for p in initial_parameters]
        logger.info(
            f"Server initialized with {len(initial_parameters)} parameter arrays"
        )
    
    def aggregate_fit(
        self,
        results: List[Tuple[List[np.ndarray], int, Dict]],
    ) -> List[np.ndarray]:
        """Aggregate parameter updates from clients.
        
        Args:
            results: List of (parameters, num_samples, metrics) from clients.
        
        Returns:
            Updated global shared parameters.
        """
        # Aggregate using strategy
        new_parameters = self.strategy.aggregate(results)
        self.current_parameters = new_parameters
        
        # Log round metrics
        round_metrics = {
            "round": self._current_round,
            "n_clients": len(results),
            "total_samples": sum(n for _, n, _ in results),
        }
        
        # Aggregate client metrics
        client_metrics = [m for _, _, m in results if m]
        if client_metrics:
            avg_train_loss = np.mean([m.get("train_loss", 0) for m in client_metrics])
            avg_train_acc = np.mean([m.get("train_accuracy", 0) for m in client_metrics])
            round_metrics["avg_train_loss"] = avg_train_loss
            round_metrics["avg_train_acc"] = avg_train_acc
            
            # Track validation metrics if available
            val_bas = [m.get("balanced_accuracy", None) for m in client_metrics]
            val_bas = [v for v in val_bas if v is not None]
            if val_bas:
                round_metrics["avg_val_balanced_accuracy"] = float(np.mean(val_bas))
            
            val_losses = [m.get("val_loss", None) for m in client_metrics]
            val_losses = [v for v in val_losses if v is not None]
            if val_losses:
                round_metrics["avg_val_loss"] = float(np.mean(val_losses))
            
            val_aucs = [m.get("auc_roc", None) for m in client_metrics]
            val_aucs = [v for v in val_aucs if v is not None]
            if val_aucs:
                round_metrics["avg_val_auc_roc"] = float(np.mean(val_aucs))
        
        self._round_metrics.append(round_metrics)
        self._current_round += 1
        
        logger.info(
            f"Round {self._current_round}: "
            f"{len(results)} clients, "
            f"avg_loss={round_metrics.get('avg_train_loss', 'N/A'):.4f}, "
            f"avg_acc={round_metrics.get('avg_train_acc', 'N/A'):.4f}"
            + (f", avg_val_BA={round_metrics['avg_val_balanced_accuracy']:.4f}" 
               if 'avg_val_balanced_accuracy' in round_metrics else "")
        )
        
        return new_parameters
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get current global shared parameters."""
        return self.current_parameters
    
    def get_round_history(self) -> List[Dict]:
        """Get metrics from all completed rounds."""
        return self._round_metrics.copy()
    
    def select_clients(
        self,
        available_clients: List,
    ) -> List:
        """Select clients for the current round.
        
        Args:
            available_clients: All available client objects.
        
        Returns:
            Selected clients for this round.
        """
        n_select = max(
            self.min_fit_clients,
            int(len(available_clients) * self.fraction_fit),
        )
        n_select = min(n_select, len(available_clients))
        
        # Random selection (could be improved with importance sampling)
        indices = np.random.choice(
            len(available_clients), size=n_select, replace=False
        )
        return [available_clients[i] for i in indices]
    
    @property
    def min_fit_clients(self) -> int:
        return self.strategy.min_fit_clients
    
    @staticmethod
    def create_strategy(
        strategy_name: str = "fedper",
        fedprox_mu: float = 0.01,
        **kwargs,
    ) -> PQFLStrategy:
        """Factory method for creating aggregation strategies.
        
        Args:
            strategy_name: "fedavg", "fedper", or "fedprox".
            fedprox_mu: Mu parameter for FedProx.
        
        Returns:
            PQFLStrategy instance.
        """
        if strategy_name == "fedavg":
            return PQFLStrategy(**kwargs)
        elif strategy_name == "fedper":
            return PQFLFedPerStrategy(**kwargs)
        elif strategy_name == "fedprox":
            return PQFLFedProxStrategy(mu=fedprox_mu, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
