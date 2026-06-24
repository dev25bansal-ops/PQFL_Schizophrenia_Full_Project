"""Hybrid Variational Quantum Classifier (VQC) for PQFL.

The HybridVQC combines classical pre-processing (tangent space encoder),
quantum feature map (RQFM), and classical post-processing (classifier head)
into a single PyTorch module compatible with Flower federated learning.

Architecture (from the research paper):
1. Classical encoder: Linear(256, 150) → BN → GELU → Linear(150, 128) → BN → GELU → Tanh
2. RQFM quantum layer: Angle encoding → StronglyEntanglingLayers (base) + BasicEntanglerLayers (personal)
3. Classical classifier head: Linear(2 + 128 + 20, 64) → Dropout → Linear(64, 2)

FedPer split:
- Shared parameters: classical encoder + VQC base layers (federated)
- Personal parameters: VQC personalization layers + classifier head (local)
"""

import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import logging

from .rqfm import create_rqfm_circuit, create_qcnn_circuit

logger = logging.getLogger(__name__)


@dataclass
class VQCConfig:
    """Configuration for the Hybrid VQC model.
    
    Attributes:
        n_qubits: Number of qubits in the quantum circuit.
        n_base_layers: Number of shared VQC layers (federated).
        n_personal_layers: Number of personal VQC layers (local).
        encoding_type: "angle" or "amplitude".
        entanglement: Entanglement pattern for RQFM.
        input_dim: Dimension of input features (from tangent PCA).
        encoder_hidden_dims: Hidden layer dimensions for classical encoder.
        encoder_activation: Activation function for encoder.
        fdt_features: Number of frequency-dependent topology features.
        classifier_hidden_dims: Hidden dims for classification head.
        dropout: Dropout probability.
        use_dual_register: Use dual hemispheric register architecture.
        circuit_type: "rqfm_vqc" or "qcnn".
        n_classes: Number of output classes (2=binary SZ/HC, 3=SZ/HC/Other,
                   4=SZ/HC/BP/Other). Defaults to 2 for backward compatibility.
    """
    n_qubits: int = 12
    n_base_layers: int = 3
    n_personal_layers: int = 1
    encoding_type: str = "angle"
    entanglement: str = "functional"
    input_dim: int = 256
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [150, 128])
    encoder_activation: str = "gelu"
    fdt_features: int = 20
    classifier_hidden_dims: List[int] = field(default_factory=lambda: [64])
    dropout: float = 0.3
    use_dual_register: bool = False
    circuit_type: str = "rqfm_vqc"
    n_classes: int = 2  # NEW: supports 2 (binary), 3 (SZ/HC/Other), 4 (SZ/HC/BP/Other)


class ClassicalEncoder(nn.Module):
    """Classical encoder: tangent PCA features → quantum-ready features.
    
    Architecture: Linear(input_dim, 150) → BN → GELU →
                  Linear(150, 128) → BN → GELU → Tanh
    
    The Tanh activation clamps outputs to [-1, 1], making them
    suitable for angle encoding via RY rotations.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 12,
        hidden_dims: List[int] = None,
        activation: str = "gelu",
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [150, 128]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            ])
            
            if activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "elu":
                layers.append(nn.ELU())
            else:
                layers.append(nn.GELU())
            
            prev_dim = hidden_dim
        
        # Final projection to n_qubits with Tanh for quantum compatibility
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.Tanh(),
        ])
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode tangent features to quantum-ready features.
        
        Args:
            x: Tangent PCA features, shape (batch_size, input_dim).
        
        Returns:
            Quantum-ready features in [-1, 1], shape (batch_size, n_qubits).
        """
        return self.encoder(x)


class ClassifierHead(nn.Module):
    """Site-specific classification head (NOT federated).
    
    Takes quantum outputs + classical projection + FDT features
    and produces binary classification (SZ vs HC).
    
    Architecture: Linear(2 + 128 + 20, 64) → Dropout → Linear(64, 2)
    
    The site-specific batch normalization accounts for distribution
    differences across sites.
    """
    
    def __init__(
        self,
        quantum_dim: int = 2,
        classical_dim: int = 128,
        fdt_dim: int = 20,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        n_classes: int = 2,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64]
        
        total_input = quantum_dim + classical_dim + fdt_dim
        
        layers = []
        prev_dim = total_input
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(
        self,
        quantum_out: torch.Tensor,
        classical_proj: torch.Tensor,
        fdt_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Classify from quantum + classical features.
        
        Args:
            quantum_out: Quantum circuit output, shape (batch, quantum_dim).
            classical_proj: Classical FC projection, shape (batch, classical_dim).
            fdt_features: FDT features, shape (batch, fdt_dim). Optional.
        
        Returns:
            Logits, shape (batch, n_classes).
        """
        parts = [quantum_out, classical_proj]
        
        if fdt_features is not None:
            parts.append(fdt_features)
        
        x = torch.cat(parts, dim=-1)
        return self.classifier(x)


class HybridVQC(nn.Module):
    """Hybrid Classical-Quantum VQC for PQFL schizophrenia classification.
    
    Full architecture:
    1. Classical encoder: tangent features → n_qubits features
    2. Quantum layer: RQFM encoding → VQC (base + personal layers)
    3. Classical classifier head: quantum + classical features → SZ/HC
    
    FedPer parameter split:
    - Shared (federated): encoder + VQC base weights
    - Personal (local): VQC personal weights + classifier head
    
    Args:
        config: VQCConfig with all hyperparameters.
    """
    
    def __init__(self, config: Optional[VQCConfig] = None):
        super().__init__()
        
        if config is None:
            config = VQCConfig()
        
        self.config = config
        
        # Classical encoder (shared)
        self.encoder = ClassicalEncoder(
            input_dim=config.input_dim,
            output_dim=config.n_qubits,
            hidden_dims=config.encoder_hidden_dims,
            activation=config.encoder_activation,
        )
        
        # Create quantum circuit
        if config.circuit_type == "rqfm_vqc":
            qnode, weight_shapes = create_rqfm_circuit(
                n_qubits=config.n_qubits,
                n_base_layers=config.n_base_layers,
                n_personal_layers=config.n_personal_layers,
                encoding_type=config.encoding_type,
                entanglement=config.entanglement,
                use_dual_register=config.use_dual_register,
            )
        elif config.circuit_type == "qcnn":
            qnode, weight_shapes = create_qcnn_circuit(
                n_qubits=config.n_qubits,
            )
        else:
            raise ValueError(f"Unknown circuit type: {config.circuit_type}")
        
        # Wrap quantum circuit as PyTorch layer
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        
        # Classical FC projection (parallel to quantum, from Han et al. response)
        self.fc_projection = nn.Sequential(
            nn.Linear(config.input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        
        # Classification head (personal/local)
        self.classifier = ClassifierHead(
            quantum_dim=2,  # 2 qubit measurements
            classical_dim=128,  # FC projection
            fdt_dim=config.fdt_features,
            hidden_dims=config.classifier_hidden_dims,
            dropout=config.dropout,
            n_classes=config.n_classes,  # NEW: configurable output classes
        )
        
        logger.info(
            f"HybridVQC created: {config.n_qubits} qubits, "
            f"{config.n_base_layers} base + {config.n_personal_layers} personal layers, "
            f"{self.count_shared_params()} shared + {self.count_personal_params()} personal params"
        )
    
    def forward(
        self,
        x: torch.Tensor,
        fdt_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the hybrid model.
        
        Args:
            x: Tangent PCA features, shape (batch_size, input_dim).
            fdt_features: Optional FDT features, shape (batch_size, fdt_dim).
        
        Returns:
            Classification logits, shape (batch_size, n_classes).
        """
        # Classical encoder → quantum-ready features
        quantum_input = self.encoder(x)
        
        # Quantum layer
        quantum_output = self.qlayer(quantum_input)
        
        # Ensure quantum output is 2D: (batch, n_quantum_outputs)
        # TorchLayer returns shape (n_outputs, batch, 1) for multiple expvals
        if quantum_output.dim() == 3:
            # Permute from (n_outputs, batch, 1) to (batch, n_outputs)
            quantum_output = quantum_output.squeeze(-1).permute(1, 0)
        elif quantum_output.dim() == 1:
            quantum_output = quantum_output.unsqueeze(0)
        
        # Classical FC projection (parallel path)
        classical_proj = self.fc_projection(x)
        
        # Classification head (uses both quantum and classical)
        logits = self.classifier(quantum_output, classical_proj, fdt_features)
        
        return logits
    
    def get_shared_parameters(self) -> List[nn.Parameter]:
        """Get parameters that should be federated (shared across sites).
        
        Shared = encoder + VQC base weights + FC projection
        
        Returns:
            List of shared parameters.
        """
        shared_params = []
        
        # Encoder parameters
        for param in self.encoder.parameters():
            shared_params.append(param)
        
        # VQC base weights
        if "base_weights" in dict(self.qlayer.named_parameters()):
            shared_params.append(self.qlayer.base_weights)
        
        # FC projection
        for param in self.fc_projection.parameters():
            shared_params.append(param)
        
        return shared_params
    
    def get_personal_parameters(self) -> List[nn.Parameter]:
        """Get parameters that should stay local (personal per site).
        
        Personal = VQC personalization weights + classifier head
        
        Returns:
            List of personal parameters.
        """
        personal_params = []
        
        # VQC personalization weights
        if "personal_weights" in dict(self.qlayer.named_parameters()):
            personal_params.append(self.qlayer.personal_weights)
        
        # Classifier head
        for param in self.classifier.parameters():
            personal_params.append(param)
        
        return personal_params
    
    def get_shared_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dict of shared parameters only.
        
        Returns:
            Dictionary of shared parameter name → tensor.
        """
        shared_state = {}
        
        # Encoder
        for name, param in self.encoder.state_dict().items():
            shared_state[f"encoder.{name}"] = param
        
        # VQC base weights
        for name, param in self.qlayer.named_parameters():
            if "base_weights" in name:
                shared_state[f"qlayer.{name}"] = param.data
        
        # FC projection
        for name, param in self.fc_projection.state_dict().items():
            shared_state[f"fc_projection.{name}"] = param
        
        return shared_state
    
    def set_shared_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Set shared parameters from a state dict.
        
        Args:
            state_dict: Dictionary of shared parameter name → tensor.
        """
        # Encoder
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
        if encoder_state:
            self.encoder.load_state_dict(encoder_state, strict=False)
        
        # VQC base weights
        for name, param in self.qlayer.named_parameters():
            full_name = f"qlayer.{name}"
            if full_name in state_dict and "base_weights" in name:
                param.data.copy_(state_dict[full_name])
        
        # FC projection
        proj_state = {
            k.replace("fc_projection.", ""): v
            for k, v in state_dict.items()
            if k.startswith("fc_projection.")
        }
        if proj_state:
            self.fc_projection.load_state_dict(proj_state, strict=False)
    
    def count_shared_params(self) -> int:
        """Count number of shared (federated) parameters."""
        return sum(p.numel() for p in self.get_shared_parameters())
    
    def count_personal_params(self) -> int:
        """Count number of personal (local) parameters."""
        return sum(p.numel() for p in self.get_personal_parameters())
    
    def count_total_params(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())
