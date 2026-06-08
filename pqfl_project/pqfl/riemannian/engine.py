"""Main Riemannian engine: unified interface for SPD manifold operations.

The RiemannianEngine provides a single entry point for all Riemannian
geometry operations needed in the PQFL pipeline, including:
- SPD matrix processing and validation
- Tangent space projection at the Fréchet mean
- Dimensionality reduction via tangent PCA
- Site-specific reference point management for federated learning
"""

import numpy as np
import torch
from typing import Optional, Dict, List, Tuple, Union
import logging

from .tangent_space import TangentSpaceProjector, TangentPCA
from .spd_utils import (
    ensure_spd,
    regularize_spd,
    validate_spd,
    spd_frechet_mean,
    parallel_transport,
    geodesic_distance,
)

logger = logging.getLogger(__name__)


class RiemannianEngine:
    """Unified Riemannian geometry engine for SPD manifold processing.
    
    This class orchestrates the complete pipeline from raw FC matrices
    to quantum-ready features:
    
    1. Regularize FC matrices (C + λI)
    2. Compute site-specific Fréchet mean
    3. Log-map to tangent space at Fréchet mean
    4. Reduce dimensionality via tangent PCA
    5. Convert to PyTorch tensors for quantum model
    
    For federated learning, each site has its own Fréchet mean as reference
    point. The global Fréchet mean is computed by aggregating site-specific
    means using parallel transport to a common reference.
    
    Args:
        n_rois: Number of regions of interest.
        n_components: Number of tangent PCA components.
        regularization_lambda: λ for C + λI regularization.
        metric: Riemannian metric type ("affine_invariant" or "log_euclidean").
    """
    
    def __init__(
        self,
        n_rois: int = 100,
        n_components: int = 256,
        regularization_lambda: float = 1e-3,
        metric: str = "affine_invariant",
    ):
        self.n_rois = n_rois
        self.n_components = n_components
        self.regularization_lambda = regularization_lambda
        self.metric = metric
        
        # Tangent space projector
        self.projector = TangentSpaceProjector(
            n_rois=n_rois,
            regularization_lambda=regularization_lambda,
        )
        
        # Tangent PCA
        self.tangent_pca = TangentPCA(
            n_components=n_components,
            n_rois=n_rois,
            regularization_lambda=regularization_lambda,
        )
        
        # Site-specific reference points for federated learning
        self._site_references: Dict[int, np.ndarray] = {}
        self._global_reference: Optional[np.ndarray] = None
        
        self._is_fitted = False
    
    @property
    def tangent_dim(self) -> int:
        """Full tangent space dimension."""
        return self.n_rois * (self.n_rois + 1) // 2
    
    @property
    def reference_point(self) -> Optional[np.ndarray]:
        """Current reference point (Fréchet mean)."""
        return self.projector.reference_point
    
    @property
    def explained_variance_ratio(self) -> Optional[np.ndarray]:
        """PCA explained variance ratio."""
        return self.tangent_pca.explained_variance_ratio
    
    def regularize_fc_matrices(self, fc_matrices: np.ndarray) -> np.ndarray:
        """Regularize a batch of FC matrices to ensure SPD.
        
        Args:
            fc_matrices: FC matrices, shape (n_samples, n_rois, n_rois).
        
        Returns:
            Regularized SPD matrices.
        """
        n_samples = fc_matrices.shape[0]
        regularized = np.zeros_like(fc_matrices)
        
        for i in range(n_samples):
            regularized[i] = regularize_spd(fc_matrices[i], self.regularization_lambda)
        
        # Validate a random sample
        idx = np.random.randint(0, n_samples)
        if not validate_spd(regularized[idx]):
            logger.warning("Regularized matrix is not SPD, applying stronger regularization")
            for i in range(n_samples):
                regularized[i] = ensure_spd(fc_matrices[i], method="nearest")
        
        return regularized
    
    def fit(
        self,
        fc_matrices: np.ndarray,
        site_id: Optional[int] = None,
    ) -> "RiemannianEngine":
        """Fit the Riemannian engine on a set of FC matrices.
        
        Computes the Fréchet mean, tangent space projection, and PCA.
        
        Args:
            fc_matrices: FC matrices, shape (n_samples, n_rois, n_rois).
            site_id: Optional site identifier for federated learning.
        
        Returns:
            self
        """
        # Step 1: Regularize
        spd_matrices = self.regularize_fc_matrices(fc_matrices)
        
        # Step 2-4: Fit tangent space projection and PCA
        self.tangent_pca.fit(spd_matrices)
        self.projector = self.tangent_pca.projector
        
        # Store site reference
        if site_id is not None:
            self._site_references[site_id] = self.reference_point.copy()
            logger.info(f"Site {site_id} reference point stored")
        
        self._is_fitted = True
        logger.info(
            f"RiemannianEngine fitted: {fc_matrices.shape[0]} samples, "
            f"{self.n_rois} ROIs, {self.n_components} PCA components, "
            f"{self.tangent_pca.total_explained_variance:.2%} variance explained"
        )
        
        return self
    
    def transform(
        self,
        fc_matrices: np.ndarray,
        return_tensor: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Transform FC matrices to reduced tangent space features.
        
        Pipeline: FC matrices → regularize → log-map → vectorize → PCA
        
        Args:
            fc_matrices: FC matrices, shape (n_samples, n_rois, n_rois).
            return_tensor: If True, return PyTorch tensor.
        
        Returns:
            Reduced features, shape (n_samples, n_components).
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        # Regularize
        spd_matrices = self.regularize_fc_matrices(fc_matrices)
        
        # Project through tangent space and PCA
        reduced = self.tangent_pca.transform(spd_matrices)
        
        if return_tensor:
            return torch.tensor(reduced, dtype=torch.float32)
        return reduced
    
    def fit_transform(
        self,
        fc_matrices: np.ndarray,
        site_id: Optional[int] = None,
        return_tensor: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Fit and transform in one step."""
        return self.fit(fc_matrices, site_id=site_id).transform(
            fc_matrices, return_tensor=return_tensor
        )
    
    def compute_global_reference(self) -> np.ndarray:
        """Compute global Fréchet mean from site-specific references.
        
        For federated learning, the global reference is the Fréchet mean
        of all site-specific Fréchet means.
        
        Returns:
            Global Fréchet mean SPD matrix.
        """
        if not self._site_references:
            raise RuntimeError("No site references stored. Call fit() with site_id first.")
        
        references = list(self._site_references.values())
        self._global_reference = spd_frechet_mean(references)
        
        logger.info(
            f"Global reference computed from {len(references)} sites"
        )
        return self._global_reference
    
    def transport_to_global(
        self,
        tangent_vec: np.ndarray,
        site_id: int,
    ) -> np.ndarray:
        """Transport a tangent vector from site reference to global reference.
        
        Essential for federated aggregation: site updates computed in local
        tangent spaces must be transported to the global tangent space before
        averaging.
        
        Args:
            tangent_vec: Tangent vector at site's reference point.
            site_id: Source site identifier.
        
        Returns:
            Tangent vector at global reference point.
        """
        if self._global_reference is None:
            self.compute_global_reference()
        
        site_ref = self._site_references.get(site_id)
        if site_ref is None:
            raise ValueError(f"No reference stored for site {site_id}")
        
        return parallel_transport(
            tangent_vec, site_ref, self._global_reference
        )
    
    def geodesic_distances(
        self,
        fc_matrices: np.ndarray,
    ) -> np.ndarray:
        """Compute pairwise geodesic distances between FC matrices.
        
        Useful for analyzing data distribution and site effects.
        
        Args:
            fc_matrices: FC matrices, shape (n_samples, n_rois, n_rois).
        
        Returns:
            Distance matrix, shape (n_samples, n_samples).
        """
        spd_matrices = self.regularize_fc_matrices(fc_matrices)
        n = spd_matrices.shape[0]
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                d = geodesic_distance(spd_matrices[i], spd_matrices[j])
                distances[i, j] = d
                distances[j, i] = d
        
        return distances
    
    def get_config(self) -> Dict:
        """Return configuration dictionary."""
        return {
            "n_rois": self.n_rois,
            "n_components": self.n_components,
            "regularization_lambda": self.regularization_lambda,
            "metric": self.metric,
            "tangent_dim": self.tangent_dim,
            "is_fitted": self._is_fitted,
        }
