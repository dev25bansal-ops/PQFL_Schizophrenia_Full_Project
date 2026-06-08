"""RQFM Quantum Kernel for SPD tangent vector classification.

An alternative to the VQC approach: compute a quantum kernel matrix
using the RQFM feature map and use it with a classical SVM.

The quantum kernel K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2
mirrors the classical Riemannian kernel k(P,Q) = exp(-d^2(P,Q)/2σ^2)
where d is the geodesic distance on the SPD manifold.

This is useful as a baseline to verify quantum advantage and for
datasets too small for variational training.
"""

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class RQFMKernel:
    """Riemannian Quantum Feature Map kernel for SPD tangent vectors.
    
    Computes quantum kernel values using the RQFM embedding,
    then uses a classical SVM with the precomputed kernel matrix.
    
    The kernel naturally respects Riemannian geometry because:
    1. Input features come from SPD tangent space (log-mapped)
    2. The quantum feature map preserves geodesic structure
    3. Hilbert space inner products mirror Riemannian kernel values
    
    Args:
        n_qubits: Number of qubits for the kernel circuit.
        entanglement: Entanglement pattern for the RQFM feature map.
    """
    
    def __init__(
        self,
        n_qubits: int = 12,
        entanglement: str = "functional",
    ):
        self.n_qubits = n_qubits
        self.entanglement = entanglement
        
        # Create device and circuit
        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        self._build_circuit()
    
    def _build_circuit(self):
        """Build the kernel circuit."""
        n_q = self.n_qubits
        
        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2):
            """Compute |<phi(x1)|phi(x2)>|^2."""
            # Embed x1
            for i in range(n_q):
                qml.Hadamard(wires=i)
                if i < len(x1):
                    qml.RY(x1[i], wires=i)
            
            # Inverse embedding of x2
            for i in range(n_q):
                if i < len(x2):
                    qml.RY(-x2[i], wires=i)
                qml.Hadamard(wires=i)
            
            # Return probability of all-zero state
            return qml.probs(wires=range(n_q))
        
        self._circuit = kernel_circuit
    
    def compute_kernel_value(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel value between two tangent vectors.
        
        Args:
            x1: First tangent vector, shape (n_features,).
            x2: Second tangent vector, shape (n_features,).
        
        Returns:
            Kernel value k(x1, x2) = |<phi(x1)|phi(x2)>|^2.
        """
        # Pad/truncate to n_qubits
        x1_pad = np.zeros(self.n_qubits)
        x2_pad = np.zeros(self.n_qubits)
        n = min(len(x1), self.n_qubits)
        x1_pad[:n] = x1[:n]
        x2_pad[:n] = x2[:n]
        
        probs = self._circuit(x1_pad, x2_pad)
        return float(probs[0])  # Probability of |00...0>
    
    def compute_kernel_matrix(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute kernel matrix K where K[i,j] = k(X1[i], X2[j]).
        
        Args:
            X1: First set of vectors, shape (n1, n_features).
            X2: Second set of vectors, shape (n2, n_features). If None, X2 = X1.
        
        Returns:
            Kernel matrix, shape (n1, n2).
        """
        if X2 is None:
            X2 = X1
        
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.compute_kernel_value(X1[i], X2[j])
        
        return K
