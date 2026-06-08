"""Utility functions for federated parameter management.

Handles conversion between PyTorch parameters and Flower's
numpy-based parameter format, with support for the FedPer
parameter split (shared vs personal).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def parameters_to_ndarrays(parameters) -> List[np.ndarray]:
    """Convert Flower Parameters to list of numpy arrays.
    
    Args:
        parameters: Flower Parameters object.
    
    Returns:
        List of numpy arrays.
    """
    try:
        import flwr as fl
        return [np.array(p) for p in parameters.tensors]
    except (ImportError, AttributeError):
        # Fallback for older Flower versions or testing
        if isinstance(parameters, list):
            return parameters
        return []


def ndarrays_to_parameters(ndarrays: List[np.ndarray]):
    """Convert list of numpy arrays to Flower Parameters.
    
    Args:
        ndarrays: List of numpy arrays.
    
    Returns:
        Flower Parameters object.
    """
    try:
        import flwr as fl
        # Convert to bytes for Flower
        tensors = [arr.tobytes() for arr in ndarrays]
        return fl.common.Parameters(
            tensors=tensors,
            tensor_type="numpy.ndarray",
        )
    except ImportError:
        return ndarrays


def get_shared_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract shared (federated) parameters from a HybridVQC model.
    
    Shared parameters = encoder + VQC base weights + FC projection
    These are the only parameters communicated to the Flower server.
    
    Args:
        model: HybridVQC model.
    
    Returns:
        List of numpy arrays (shared parameters).
    """
    shared_params = []
    
    if hasattr(model, 'get_shared_state_dict'):
        state_dict = model.get_shared_state_dict()
        for key in sorted(state_dict.keys()):
            shared_params.append(state_dict[key].cpu().numpy())
    else:
        # Fallback: get all parameters
        for param in model.parameters():
            shared_params.append(param.detach().cpu().numpy())
    
    return shared_params


def set_shared_parameters(
    model: nn.Module,
    parameters: List[np.ndarray],
) -> None:
    """Set shared (federated) parameters in a HybridVQC model.
    
    Only updates shared parameters; personal parameters are left unchanged.
    
    Args:
        model: HybridVQC model.
        parameters: List of numpy arrays from Flower server.
    """
    if hasattr(model, 'set_shared_state_dict'):
        # Reconstruct state dict
        shared_state = model.get_shared_state_dict()
        keys = sorted(shared_state.keys())
        
        state_dict = {}
        for key, param in zip(keys, parameters):
            state_dict[key] = torch.tensor(param, dtype=torch.float32)
        
        model.set_shared_state_dict(state_dict)
    else:
        # Fallback: set all parameters
        params = list(model.parameters())
        for param, new_val in zip(params, parameters):
            param.data = torch.tensor(new_val, dtype=param.dtype)


def get_personal_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract personal (local) parameters from a HybridVQC model.
    
    Personal parameters = VQC personalization weights + classifier head
    These are NEVER communicated to the server.
    
    Args:
        model: HybridVQC model.
    
    Returns:
        List of numpy arrays (personal parameters).
    """
    if hasattr(model, 'get_personal_parameters'):
        return [p.detach().cpu().numpy() for p in model.get_personal_parameters()]
    return []


def compute_model_diff(
    old_params: List[np.ndarray],
    new_params: List[np.ndarray],
) -> List[np.ndarray]:
    """Compute the difference between two sets of parameters.
    
    Useful for FedProx regularization: compute how far
    local parameters have diverged from the global model.
    
    Args:
        old_params: Original parameters.
        new_params: Updated parameters.
    
    Returns:
        List of parameter differences.
    """
    return [new - old for new, old in zip(new_params, old_params)]
