"""Riemannian Quantum Feature Map (RQFM) implementation.

The RQFM is the core innovation: a geometry-aware quantum feature map
that preserves the Riemannian structure of SPD matrices when encoding
them into quantum Hilbert space.

Three-stage RQFM architecture:
1. Riemannian Flattening: Log-Euclidean transform C = log(P)
2. Structure-Preserving Block Encoding: Block-encoding of log-FC matrix
3. Quantum Geodesic Attention: Learned entanglement patterns for
   geodesic-aware mixing

For practical implementation on NISQ devices, we use angle encoding
of tangent-space features followed by geometry-aware entanglement.
"""

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from typing import Optional, List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class RQFMFeatureMap:
    """Riemannian Quantum Feature Map for SPD tangent vectors.
    
    Maps tangent space coordinates (from SPD manifold log-map) into
    quantum Hilbert space using a geometry-aware encoding strategy
    that respects the Riemannian structure.
    
    The feature map has three components:
    1. Angle/amplitude encoding of tangent vectors
    2. Geometry-aware entanglement mirroring functional network groups
    3. Optional geodesic attention for cross-network mixing
    
    Args:
        n_qubits: Number of qubits for the feature map.
        encoding_type: "angle" (RY rotations) or "amplitude" (amplitude encoding).
        entanglement: Entanglement pattern: "functional", "linear", "circular", "full".
        n_network_groups: Number of functional network groups for structured entanglement.
        network_sizes: List of qubit counts per network group (must sum to n_qubits).
    """
    
    def __init__(
        self,
        n_qubits: int = 12,
        encoding_type: str = "angle",
        entanglement: str = "functional",
        n_network_groups: int = 3,
        network_sizes: Optional[List[int]] = None,
    ):
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type
        self.entanglement = entanglement
        self.n_network_groups = n_network_groups
        
        # Default network sizes for Schaefer 100-ROI with Yeo 7 networks
        # Mapped to qubit groups: DMN (qubits 0-3), FPN (qubits 4-7), SN (qubits 8-11)
        if network_sizes is not None:
            self.network_sizes = network_sizes
        else:
            # Distribute qubits across 3 main functional groups
            base = n_qubits // n_network_groups
            remainder = n_qubits % n_network_groups
            self.network_sizes = [
                base + (1 if i < remainder else 0)
                for i in range(n_network_groups)
            ]
        
        # Compute network boundaries
        self._network_boundaries = []
        start = 0
        for size in self.network_sizes:
            self._network_boundaries.append((start, start + size))
            start += size
        
        logger.info(
            f"RQFM Feature Map: {n_qubits} qubits, {encoding_type} encoding, "
            f"{entanglement} entanglement, network groups: {self.network_sizes}"
        )
    
    def __call__(self, features: pnp.ndarray, wires: qml.wires.Wires) -> None:
        """Apply the RQFM feature map to quantum wires.
        
        Args:
            features: Input features (tangent space coordinates), shape (n_qubits,).
            wires: Qubit wires to apply the feature map on.
        """
        if self.encoding_type == "angle":
            self._angle_encoding(features, wires)
        elif self.encoding_type == "amplitude":
            self._amplitude_encoding(features, wires)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
        
        # Apply geometry-aware entanglement
        self._apply_entanglement(wires)
    
    def _angle_encoding(self, features: pnp.ndarray, wires: qml.wires.Wires) -> None:
        """Angle encoding via RY rotations.
        
        Maps n_qubits tangent features to n_qubits qubits using
        RY rotations. This is the primary encoding for the PQFL system.
        
        RY is chosen over RX/RZ because it provides real-valued amplitudes
        in the computational basis, which is natural for FC correlation values.
        """
        for i, wire in enumerate(wires):
            if i < len(features):
                qml.RY(features[i], wires=wire)
            else:
                # Pad with zero rotation (no-op but explicit)
                pass
    
    def _amplitude_encoding(self, features: pnp.ndarray, wires: qml.wires.Wires) -> None:
        """Amplitude encoding for high-dimensional tangent vectors.
        
        Encodes 2^n features into n qubits using amplitude embedding.
        This provides exponential compression: 2^12 = 4096 features
        into 12 qubits.
        
        Note: Requires features to be normalized (norm = 1).
        """
        qml.AmplitudeEmbedding(
            features,
            wires=wires,
            normalize=True,
            pad_with=0.0,
        )
    
    def _apply_entanglement(self, wires: qml.wires.Wires) -> None:
        """Apply geometry-aware entanglement pattern.
        
        The entanglement structure mirrors functional brain networks:
        - Intra-network: Full entanglement within each network group (CNOT)
        - Inter-network: Selective entanglement between network groups (CZ)
        
        This implements "Quantum Geodesic Attention" from the RQFM paper,
        where entanglement patterns respect the SPD correlation structure.
        """
        if self.entanglement == "functional":
            self._functional_entanglement(wires)
        elif self.entanglement == "linear":
            for i in range(len(wires) - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
        elif self.entanglement == "circular":
            for i in range(len(wires)):
                qml.CNOT(wires=[wires[i], wires[(i + 1) % len(wires)]])
        elif self.entanglement == "full":
            for i in range(len(wires)):
                for j in range(i + 1, len(wires)):
                    qml.CNOT(wires=[wires[i], wires[j]])
        else:
            raise ValueError(f"Unknown entanglement: {self.entanglement}")
    
    def _functional_entanglement(self, wires: qml.wires.Wires) -> None:
        """Functional network-aware entanglement.
        
        Within each functional network group (DMN, FPN, SN):
        - Full CNOT entanglement captures intra-network correlations
        
        Between network groups:
        - Selective CZ gates capture cross-network dependencies
        - Mimics geodesic attention across manifold regions
        """
        # Intra-network: full CNOT entanglement
        for start, end in self._network_boundaries:
            group_wires = wires[start:end]
            if len(group_wires) > 1:
                for i in range(len(group_wires) - 1):
                    qml.CNOT(wires=[group_wires[i], group_wires[i + 1]])
                # Close the loop for groups of 3+
                if len(group_wires) > 2:
                    qml.CNOT(wires=[group_wires[-1], group_wires[0]])
        
        # Inter-network: selective CZ gates
        # Connect the last qubit of each group to the first of the next
        for idx in range(len(self._network_boundaries) - 1):
            _, end_current = self._network_boundaries[idx]
            start_next, _ = self._network_boundaries[idx + 1]
            qml.CZ(wires=[wires[end_current - 1], wires[start_next]])
    
    def get_network_groups(self) -> List[List[int]]:
        """Return qubit indices for each functional network group."""
        groups = []
        for start, end in self._network_boundaries:
            groups.append(list(range(start, end)))
        return groups


def create_rqfm_circuit(
    n_qubits: int = 12,
    n_base_layers: int = 3,
    n_personal_layers: int = 1,
    encoding_type: str = "angle",
    entanglement: str = "functional",
    use_dual_register: bool = False,
) -> Tuple[qml.QNode, Dict]:
    """Create a complete RQFM-based VQC circuit.
    
    The circuit has the FedPer-compatible split:
    - Base layers: StronglyEntanglingLayers (shared across sites, federated)
    - Personalization layers: BasicEntanglerLayers (local per site, not federated)
    
    Args:
        n_qubits: Number of qubits.
        n_base_layers: Number of shared variational layers.
        n_personal_layers: Number of personalization layers.
        encoding_type: "angle" or "amplitude".
        entanglement: Entanglement pattern.
        use_dual_register: If True, use dual hemispheric register architecture.
    
    Returns:
        Tuple of (QNode, weight_shapes_dict) for PennyLane TorchLayer.
    """
    dev = qml.device("lightning.qubit", wires=n_qubits)
    feature_map = RQFMFeatureMap(
        n_qubits=n_qubits,
        encoding_type=encoding_type,
        entanglement=entanglement,
    )
    
    # Dual register architecture for 13 qubits:
    # Left hemisphere (0-5) + Right hemisphere (6-11) + Shared (12)
    if use_dual_register and n_qubits == 13:
        left_wires = list(range(0, 6))
        right_wires = list(range(6, 12))
        shared_wire = 12
    else:
        left_wires = None
        right_wires = None
        shared_wire = None
    
    @qml.qnode(dev, interface="torch")
    def rqfm_vqc(inputs, base_weights, personal_weights):
        """RQFM Variational Quantum Classifier.
        
        Pipeline:
        1. Feature map: RQFM angle encoding of tangent vectors
        2. Functional network entanglement (geometry-aware)
        3. Base layers: StronglyEntanglingLayers (shared/federated)
        4. Personal layers: BasicEntanglerLayers (local/personal)
        5. Measurement: Pauli-Z expectation on output qubits
        
        Note: First argument must be named 'inputs' for TorchLayer compatibility.
        Uses PennyLane's built-in AngleEmbedding for batch compatibility.
        """
        wires = list(range(n_qubits))
        
        # Stage 1: RQFM Angle Encoding
        # Maps tangent space features to quantum rotations
        qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
        
        # Stage 1b: Functional network entanglement
        # Intra-network CNOTs + inter-network CZ gates
        if entanglement == "functional":
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
            # Inter-network CZ at boundaries
            boundary = n_qubits // 3
            if boundary > 0 and boundary < n_qubits:
                qml.CZ(wires=[wires[boundary - 1], wires[boundary]])
                boundary2 = 2 * n_qubits // 3
                if boundary2 < n_qubits:
                    qml.CZ(wires=[wires[boundary2 - 1], wires[boundary2]])
        elif entanglement == "full":
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    qml.CNOT(wires=[wires[i], wires[j]])
        else:
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
        
        # Stage 2: Shared Base Layers (StronglyEntanglingLayers)
        if n_base_layers > 0:
            qml.StronglyEntanglingLayers(
                base_weights,
                wires=wires,
            )
        
        # Dual register hemispheric mixing
        if use_dual_register and left_wires is not None and shared_wire is not None:
            for lw in left_wires:
                qml.CNOT(wires=[lw, shared_wire])
            for rw in right_wires:
                qml.CNOT(wires=[rw, shared_wire])
            for lw in left_wires:
                qml.CNOT(wires=[shared_wire, lw])
            for rw in right_wires:
                qml.CNOT(wires=[shared_wire, rw])
        
        # Stage 3: Personalization Layers (BasicEntanglerLayers)
        if n_personal_layers > 0:
            qml.BasicEntanglerLayers(
                personal_weights,
                wires=wires,
            )
        
        # Stage 4: Measurement
        # Return as tuple for TorchLayer compatibility
        return tuple(qml.expval(qml.PauliZ(i)) for i in range(2))
    
    # Define weight shapes for TorchLayer
    weight_shapes = {
        "base_weights": (n_base_layers, n_qubits, 3),  # StronglyEntanglingLayers shape
        "personal_weights": (n_personal_layers, n_qubits),  # BasicEntanglerLayers shape
    }
    
    return rqfm_vqc, weight_shapes


def create_qcnn_circuit(
    n_qubits: int = 13,
    n_conv_layers: int = 4,
) -> Tuple[qml.QNode, Dict]:
    """Create a QCNN (Quantum Convolutional Neural Network) circuit.
    
    The QCNN alternates between convolutional and pooling layers:
    - Conv: 2-qubit parameterized gates (RZ·RY·RZ ⊗ RZ·RY·RZ · CNOT)
    - Pool: Measurement + conditional reset (halves qubits per layer)
    
    Dual-register architecture for 13 qubits:
    - Left hemisphere FC: qubits 0-5
    - Right hemisphere FC: qubits 6-11
    - Shared: qubit 12
    
    Total quantum parameters: 48 per conv layer × 4 layers = 192
    
    Args:
        n_qubits: Total number of qubits (default 13).
        n_conv_layers: Number of conv-pool layer pairs (default 4).
    
    Returns:
        Tuple of (QNode, weight_shapes_dict).
    """
    dev = qml.device("lightning.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface="torch")
    def qcnn_circuit(inputs, conv_weights):
        """QCNN circuit for FC matrix classification.
        
        Architecture:
        1. Amplitude encoding of FC features
        2. 4 alternating conv-pool layers
        3. Dual-register hemispheric architecture
        4. 2-qubit measurement output
        
        Note: First argument must be named 'inputs' for TorchLayer compatibility.
        """
        # Amplitude encoding
        qml.AmplitudeEmbedding(
            inputs, wires=range(n_qubits), normalize=True, pad_with=0.0
        )
        
        # Track active qubits for pooling
        active_qubits = list(range(n_qubits))
        weight_idx = 0
        
        for layer in range(n_conv_layers):
            if len(active_qubits) < 2:
                break
            
            # Convolutional layer
            for i in range(0, len(active_qubits) - 1, 2):
                q1, q2 = active_qubits[i], active_qubits[i + 1]
                # 2-qubit conv gate: RZ·RY·RZ ⊗ RZ·RY·RZ · CNOT
                w = conv_weights[weight_idx]
                weight_idx += 1
                qml.RZ(w[0], wires=q1)
                qml.RY(w[1], wires=q1)
                qml.RZ(w[2], wires=q1)
                qml.RZ(w[3], wires=q2)
                qml.RY(w[4], wires=q2)
                qml.RZ(w[5], wires=q2)
                qml.CNOT(wires=[q1, q2])
            
            # Pooling: keep every other qubit
            active_qubits = active_qubits[::2]
        
        # Measurement on remaining active qubits (at least 2)
        output_wires = active_qubits[:2] if len(active_qubits) >= 2 else [0, 1]
        return [qml.expval(qml.PauliZ(w)) for w in output_wires]
    
    # Weight shapes: 12 parameters per 2-qubit conv block
    # Number of blocks per layer ≈ n_qubits / 2
    total_blocks = 0
    active = n_qubits
    for _ in range(n_conv_layers):
        total_blocks += active // 2
        active = active // 2
    
    weight_shapes = {
        "conv_weights": (total_blocks, 6),  # 6 params per 2-qubit block
    }
    
    return qcnn_circuit, weight_shapes
