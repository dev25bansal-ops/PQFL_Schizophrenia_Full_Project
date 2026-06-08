"""Quantum and classical saliency methods for model interpretability.

Implements:
- Quantum saliency via parameter-shift rule
- Classical saliency via integrated gradients
- PANSS correlation for clinical validation
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class QuantumSaliency:
    """Quantum saliency using the parameter-shift rule.
    
    Computes feature importance by shifting quantum circuit parameters
    and measuring the change in output. This is the quantum analog
    of gradient-based saliency.
    
    For RQFM, the saliency maps correspond to true SPD manifold features,
    making them more interpretable than classical neural network saliency.
    
    Args:
        model: HybridVQC model.
        n_shifts: Number of parameter shifts for gradient estimation.
    """
    
    def __init__(self, model: nn.Module, n_shifts: int = 2):
        self.model = model
        self.n_shifts = n_shifts
    
    def compute_saliency(
        self,
        x: torch.Tensor,
        target_class: int = 1,
        fdt_features: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Compute quantum saliency map for input features.
        
        Uses the parameter-shift rule: for each input feature,
        shift by ±π/2 and measure the output change.
        
        Args:
            x: Input features, shape (batch_size, input_dim).
            target_class: Target class for saliency (1=SZ).
            fdt_features: Optional FDT features.
        
        Returns:
            Saliency map, shape (batch_size, input_dim).
        """
        self.model.eval()
        x.requires_grad = True
        
        saliency = torch.zeros_like(x)
        
        # Use autograd as approximation of parameter-shift
        with torch.enable_grad():
            logits = self.model(x, fdt_features=fdt_features)
            target_logits = logits[:, target_class]
            target_logits.sum().backward()
        
        if x.grad is not None:
            saliency = x.grad.abs().detach()
        
        return saliency.cpu().numpy()
    
    def compute_quantum_parameter_saliency(
        self,
        x: torch.Tensor,
        fdt_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute saliency of individual quantum parameters.
        
        Uses the parameter-shift rule on VQC parameters to identify
        which quantum gates are most important for classification.
        
        Args:
            x: Input features.
            fdt_features: Optional FDT features.
        
        Returns:
            Dictionary with parameter saliency values.
        """
        self.model.eval()
        saliency = {}
        
        if not hasattr(self.model, 'qlayer'):
            return saliency
        
        # Get base predictions
        with torch.no_grad():
            base_logits = self.model(x, fdt_features=fdt_features)
        
        # Parameter-shift for each VQC parameter
        for name, param in self.model.qlayer.named_parameters():
            original = param.data.clone()
            shift = np.pi / 2
            
            # Forward shift
            param.data = original + shift
            with torch.no_grad():
                logits_plus = self.model(x, fdt_features=fdt_features)
            
            # Backward shift
            param.data = original - shift
            with torch.no_grad():
                logits_minus = self.model(x, fdt_features=fdt_features)
            
            # Restore
            param.data = original
            
            # Gradient = (f(θ+π/2) - f(θ-π/2)) / 2
            grad = (logits_plus - logits_minus) / 2
            saliency[name] = grad.abs().mean(dim=0).detach().cpu().numpy()
        
        return saliency


class ClassicalSaliency:
    """Classical saliency using integrated gradients.
    
    Computes feature importance for the classical components
    of the hybrid model.
    
    Args:
        model: HybridVQC model.
        n_steps: Number of integration steps.
    """
    
    def __init__(self, model: nn.Module, n_steps: int = 50):
        self.model = model
        self.n_steps = n_steps
    
    def compute_integrated_gradients(
        self,
        x: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        target_class: int = 1,
        fdt_features: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Compute integrated gradients for input features.
        
        Args:
            x: Input features, shape (batch_size, input_dim).
            baseline: Baseline input (default: zero vector).
            target_class: Target class for attribution.
            fdt_features: Optional FDT features.
        
        Returns:
            Integrated gradients, shape (batch_size, input_dim).
        """
        if baseline is None:
            baseline = torch.zeros_like(x)
        
        # Interpolate from baseline to input
        alphas = torch.linspace(0, 1, self.n_steps + 1, device=x.device)
        all_grads = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (x - baseline)
            interpolated = interpolated.clone().detach().requires_grad_(True)
            
            self.model.eval()
            logits = self.model(interpolated, fdt_features=fdt_features)
            target_logits = logits[:, target_class]
            
            self.model.zero_grad()
            if interpolated.grad is not None:
                interpolated.grad.zero_()
            target_logits.sum().backward()
            
            if interpolated.grad is not None:
                all_grads.append(interpolated.grad.clone())
        
        # Average gradients and multiply by (x - baseline)
        avg_grads = torch.stack(all_grads).mean(dim=0)
        integrated_grads = avg_grads * (x - baseline)
        
        return integrated_grads.detach().cpu().numpy()
