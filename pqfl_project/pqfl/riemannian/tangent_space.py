"""Tangent space projection and dimensionality reduction for SPD matrices.

The tangent space at the Fréchet mean provides a vector space representation
of SPD matrices, enabling the use of standard ML tools (PCA, SVM, neural nets)
while preserving the Riemannian geometric structure.

Pipeline: SPD matrices → log-map at Fréchet mean → vectorize → PCA → reduced features
"""

import numpy as np
import scipy.linalg as la
from sklearn.decomposition import PCA
from typing import Optional, Union
import logging

from .spd_utils import (
    spd_frechet_mean,
    _log_map,
    regularize_spd,
    validate_spd,
)

logger = logging.getLogger(__name__)


class TangentSpaceProjector:
    """Project SPD matrices to/from tangent space at the Fréchet mean.
    
    The tangent space projection converts SPD matrices (which lie on a curved
    Riemannian manifold) to flat Euclidean vectors, enabling the application
    of standard ML algorithms. The projection point is chosen as the Fréchet
    mean, which minimizes total squared geodesic distance.
    
    Pipeline:
    1. Compute Fréchet mean of all SPD matrices
    2. Log-map each matrix to tangent space at the Fréchet mean
    3. Vectorize the symmetric tangent matrices (upper triangular)
    
    Args:
        n_rois: Number of regions of interest (matrix dimension).
        regularization_lambda: λ for C + λI regularization.
        frechet_max_iter: Maximum iterations for Fréchet mean computation.
        frechet_tol: Convergence tolerance for Fréchet mean.
    """
    
    def __init__(
        self,
        n_rois: int,
        regularization_lambda: float = 1e-3,
        frechet_max_iter: int = 20,
        frechet_tol: float = 1e-6,
    ):
        self.n_rois = n_rois
        self.regularization_lambda = regularization_lambda
        self.frechet_max_iter = frechet_max_iter
        self.frechet_tol = frechet_tol
        
        # Computed during fit
        self.reference_point: Optional[np.ndarray] = None
        self._is_fitted = False
    
    @property
    def tangent_dim(self) -> int:
        """Dimension of vectorized tangent space (n_rois * (n_rois + 1) / 2)."""
        return self.n_rois * (self.n_rois + 1) // 2
    
    def fit(self, spd_matrices: np.ndarray) -> "TangentSpaceProjector":
        """Compute the Fréchet mean as reference point.
        
        Args:
            spd_matrices: Array of SPD matrices, shape (n_samples, n_rois, n_rois).
        
        Returns:
            self
        """
        if spd_matrices.ndim != 3:
            raise ValueError(
                f"Expected 3D array (n_samples, n_rois, n_rois), "
                f"got shape {spd_matrices.shape}"
            )
        
        n_samples = spd_matrices.shape[0]
        logger.info(
            f"Computing Fréchet mean for {n_samples} SPD matrices "
            f"of dimension {self.n_rois}"
        )
        
        # Regularize all matrices
        matrices = [
            regularize_spd(spd_matrices[i], self.regularization_lambda)
            for i in range(n_samples)
        ]
        
        # Compute Fréchet mean
        self.reference_point = spd_frechet_mean(
            matrices,
            max_iter=self.frechet_max_iter,
            tol=self.frechet_tol,
        )
        self._is_fitted = True
        logger.info("Fréchet mean computed successfully")
        
        return self
    
    def transform(self, spd_matrices: np.ndarray) -> np.ndarray:
        """Project SPD matrices to tangent space at the Fréchet mean.
        
        Args:
            spd_matrices: Array of SPD matrices, shape (n_samples, n_rois, n_rois).
        
        Returns:
            Tangent vectors, shape (n_samples, tangent_dim).
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        n_samples = spd_matrices.shape[0]
        tangent_vectors = np.zeros((n_samples, self.tangent_dim))
        
        for i in range(n_samples):
            mat = regularize_spd(spd_matrices[i], self.regularization_lambda)
            tangent_mat = _log_map(mat, self.reference_point)
            tangent_vectors[i] = self._vectorize_symmetric(tangent_mat)
        
        return tangent_vectors
    
    def fit_transform(self, spd_matrices: np.ndarray) -> np.ndarray:
        """Fit and project in one step.
        
        Args:
            spd_matrices: Array of SPD matrices, shape (n_samples, n_rois, n_rois).
        
        Returns:
            Tangent vectors, shape (n_samples, tangent_dim).
        """
        return self.fit(spd_matrices).transform(spd_matrices)
    
    def inverse_transform(self, tangent_vectors: np.ndarray) -> np.ndarray:
        """Project tangent vectors back to SPD manifold.
        
        Args:
            tangent_vectors: Tangent vectors, shape (n_samples, tangent_dim).
        
        Returns:
            SPD matrices, shape (n_samples, n_rois, n_rois).
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before inverse_transform()")
        
        n_samples = tangent_vectors.shape[0]
        spd_matrices = np.zeros((n_samples, self.n_rois, self.n_rois))
        
        for i in range(n_samples):
            tangent_mat = self._unvectorize_symmetric(tangent_vectors[i])
            spd_matrices[i] = self._exp_map_at_ref(tangent_mat)
        
        return spd_matrices
    
    def _vectorize_symmetric(self, matrix: np.ndarray) -> np.ndarray:
        """Extract upper triangular elements of symmetric matrix.
        
        Args:
            matrix: Symmetric matrix, shape (n_rois, n_rois).
        
        Returns:
            Vector of upper triangular elements, shape (tangent_dim,).
        """
        return matrix[np.triu_indices(self.n_rois)]
    
    def _unvectorize_symmetric(self, vector: np.ndarray) -> np.ndarray:
        """Reconstruct symmetric matrix from upper triangular vector.
        
        Args:
            vector: Upper triangular elements, shape (tangent_dim,).
        
        Returns:
            Symmetric matrix, shape (n_rois, n_rois).
        """
        mat = np.zeros((self.n_rois, self.n_rois))
        mat[np.triu_indices(self.n_rois)] = vector
        # Mirror to lower triangle
        mat = mat + mat.T - np.diag(np.diag(mat))
        return mat
    
    def _exp_map_at_ref(self, tangent_vec: np.ndarray) -> np.ndarray:
        """Exp-map at the reference point (Fréchet mean).
        
        Args:
            tangent_vec: Symmetric tangent matrix.
        
        Returns:
            SPD matrix on the manifold.
        """
        from .spd_utils import _exp_map
        result = _exp_map(tangent_vec, self.reference_point)
        return 0.5 * (result + result.T)


class TangentPCA:
    """Tangent-space PCA for dimensionality reduction of SPD matrices.
    
    This combines tangent space projection with PCA to reduce the
    dimensionality of SPD matrices while preserving maximal variance.
    
    Pipeline:
    1. SPD matrices → tangent space at Fréchet mean (vectorized)
    2. PCA on tangent vectors for dimensionality reduction
    
    For 100-ROI parcellation:
    - Input: 5050-dimensional tangent vectors
    - Output: n_components-dimensional reduced features (default: 256)
    
    Args:
        n_components: Number of PCA components to retain.
        n_rois: Number of ROIs (matrix dimension).
        regularization_lambda: λ for SPD regularization.
    """
    
    def __init__(
        self,
        n_components: int = 256,
        n_rois: int = 100,
        regularization_lambda: float = 1e-3,
    ):
        self.n_components = n_components
        self.n_rois = n_rois
        self.regularization_lambda = regularization_lambda
        
        self.projector = TangentSpaceProjector(
            n_rois=n_rois,
            regularization_lambda=regularization_lambda,
        )
        self.pca = PCA(n_components=n_components)
        self._is_fitted = False
    
    @property
    def explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Fraction of variance explained by each component."""
        if not self._is_fitted:
            return None
        return self.pca.explained_variance_ratio_
    
    @property
    def total_explained_variance(self) -> Optional[float]:
        """Total fraction of variance explained."""
        if not self._is_fitted:
            return None
        return float(np.sum(self.pca.explained_variance_ratio_))
    
    def fit(self, spd_matrices: np.ndarray) -> "TangentPCA":
        """Fit tangent space projection and PCA.
        
        Args:
            spd_matrices: SPD matrices, shape (n_samples, n_rois, n_rois).
        
        Returns:
            self
        """
        # Step 1: Project to tangent space
        tangent_vectors = self.projector.fit_transform(spd_matrices)
        logger.info(
            f"Tangent vectors shape: {tangent_vectors.shape}, "
            f"dim={self.projector.tangent_dim}"
        )
        
        # Step 2: Fit PCA
        self.pca.fit(tangent_vectors)
        self._is_fitted = True
        
        logger.info(
            f"TangentPCA fitted: {self.n_components} components explain "
            f"{self.total_explained_variance:.2%} of variance"
        )
        
        return self
    
    def transform(self, spd_matrices: np.ndarray) -> np.ndarray:
        """Project SPD matrices through tangent space and PCA.
        
        Args:
            spd_matrices: SPD matrices, shape (n_samples, n_rois, n_rois).
        
        Returns:
            Reduced features, shape (n_samples, n_components).
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        tangent_vectors = self.projector.transform(spd_matrices)
        return self.pca.transform(tangent_vectors)
    
    def fit_transform(self, spd_matrices: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(spd_matrices).transform(spd_matrices)
    
    def inverse_transform_pca(self, reduced: np.ndarray) -> np.ndarray:
        """Inverse PCA only (back to tangent space, not to SPD manifold).
        
        Args:
            reduced: Reduced features, shape (n_samples, n_components).
        
        Returns:
            Tangent vectors, shape (n_samples, tangent_dim).
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before inverse_transform_pca()")
        return self.pca.inverse_transform(reduced)
    
    def inverse_transform(self, reduced: np.ndarray) -> np.ndarray:
        """Full inverse: reduced → tangent space → SPD manifold.
        
        Args:
            reduced: Reduced features, shape (n_samples, n_components).
        
        Returns:
            SPD matrices, shape (n_samples, n_rois, n_rois).
        """
        tangent_vectors = self.inverse_transform_pca(reduced)
        return self.projector.inverse_transform(tangent_vectors)
