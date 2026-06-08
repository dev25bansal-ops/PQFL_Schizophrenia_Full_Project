"""Quantum simulator management for PennyLane.

Handles device creation, backend selection, and simulation configuration
for the PQFL pipeline. Supports CPU and GPU simulators.
"""

import pennylane as qml
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class QuantumSimulator:
    """Manager for quantum simulation backends.
    
    Provides a unified interface for creating and configuring
    PennyLane quantum devices with optimal backend selection.
    
    For circuits with < 20 qubits, CPU simulation via lightning.qubit
    is actually faster than GPU due to GPU overhead exceeding the benefit.
    For larger circuits (> 20 qubits), lightning.gpu with cuQuantum
    provides significant speedup.
    
    Args:
        n_qubits: Number of qubits.
        backend: Simulation backend: "lightning", "default", "gpu", "tensor".
        shots: Number of measurement shots. None for state vector simulation.
    """
    
    # Backend mapping
    BACKEND_MAP = {
        "lightning": "lightning.qubit",    # C++ simulator, ~10x faster
        "default": "default.qubit",         # Pure Python, good for debugging
        "gpu": "lightning.gpu",             # GPU-accelerated with cuQuantum
        "tensor": "lightning.tensor",       # Tensor network for large qubits
    }
    
    def __init__(
        self,
        n_qubits: int = 12,
        backend: str = "lightning",
        shots: Optional[int] = None,
    ):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self._device = None
    
    def get_device(self):
        """Get or create the PennyLane quantum device.
        
        Returns:
            Configured PennyLane device.
        """
        if self._device is not None:
            return self._device
        
        backend_name = self.BACKEND_MAP.get(self.backend, "lightning.qubit")
        
        try:
            self._device = qml.device(
                backend_name,
                wires=self.n_qubits,
                shots=self.shots,
            )
            logger.info(f"Created quantum device: {backend_name}, {self.n_qubits} qubits")
        except Exception as e:
            logger.warning(
                f"Failed to create {backend_name} device: {e}. "
                f"Falling back to default.qubit"
            )
            self._device = qml.device(
                "default.qubit",
                wires=self.n_qubits,
                shots=self.shots,
            )
        
        return self._device
    
    def estimate_memory(self) -> Dict[str, float]:
        """Estimate memory requirements for state vector simulation.
        
        Returns:
            Dictionary with memory estimates in KB, MB, GB.
        """
        # State vector: 2^n complex128 values
        n_complex = 2 ** self.n_qubits
        bytes_per_complex = 16  # complex128
        total_bytes = n_complex * bytes_per_complex
        
        return {
            "state_vector_elements": n_complex,
            "total_bytes": total_bytes,
            "total_kb": total_bytes / 1024,
            "total_mb": total_bytes / (1024 ** 2),
            "total_gb": total_bytes / (1024 ** 3),
        }
    
    def estimate_circuit_time(self, batch_size: int = 32) -> float:
        """Estimate per-batch circuit execution time.
        
        Based on benchmarks from the implementation research:
        - 12 qubits, CPU: ~0.02s per circuit, ~30s per site round
        - 16 qubits, CPU: ~0.05s per circuit
        
        Args:
            batch_size: Number of samples in a batch.
        
        Returns:
            Estimated time in seconds.
        """
        base_time = {12: 0.02, 16: 0.05, 20: 0.15}.get(self.n_qubits, 0.1)
        
        if self.backend == "gpu" and self.n_qubits > 16:
            base_time *= 0.5  # GPU speedup for larger circuits
        
        return base_time * batch_size
    
    @staticmethod
    def recommend_backend(n_qubits: int, has_gpu: bool = False) -> str:
        """Recommend the best backend for a given configuration.
        
        Args:
            n_qubits: Number of qubits.
            has_gpu: Whether GPU is available.
        
        Returns:
            Recommended backend name.
        """
        if n_qubits <= 20:
            return "lightning"  # CPU is faster for small circuits
        elif has_gpu:
            return "gpu"
        else:
            return "tensor"  # Tensor network for large qubits without GPU
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {
            "n_qubits": self.n_qubits,
            "backend": self.backend,
            "shots": self.shots,
            "memory": self.estimate_memory(),
        }
