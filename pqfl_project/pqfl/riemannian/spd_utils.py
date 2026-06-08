"""SPD matrix utilities: regularization, validation, Fréchet mean, parallel transport.

Key mathematical operations for the Riemannian geometry pipeline:
- Regularization: C + λI ensures positive definiteness
- Fréchet mean: iterative log-Euclidean approximation
- Parallel transport: S' = E·S·E^T under AIRM
- Geodesic distance: d(P,Q) = ||log(P^{-1/2}QP^{-1/2})||_F
"""

import numpy as np
import scipy.linalg as la
from typing import Optional, Union, List
import warnings
import logging

logger = logging.getLogger(__name__)


def ensure_spd(
    matrix: np.ndarray,
    method: str = "regularize",
    lambda_val: float = 1e-3,
) -> np.ndarray:
    """Ensure a matrix is symmetric positive definite.
    
    Args:
        matrix: Input matrix, should be square.
        method: Method to ensure SPD. Options:
            - "regularize": Add λI (recommended for FC matrices)
            - "nearest": Project to nearest SPD matrix (Higham 2002)
            - "clip": Clip eigenvalues to be positive
    Returns:
        SPD matrix.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {matrix.shape}")
    
    # Symmetrize first
    matrix = 0.5 * (matrix + matrix.T)
    
    if method == "regularize":
        return regularize_spd(matrix, lambda_val)
    elif method == "nearest":
        return nearest_spd(matrix)
    elif method == "clip":
        return _clip_eigenvalues(matrix)
    else:
        raise ValueError(f"Unknown method: {method}")


def regularize_spd(matrix: np.ndarray, lambda_val: float = 1e-3) -> np.ndarray:
    """Regularize an SPD matrix by adding λI.
    
    This is the recommended approach for fMRI functional connectivity matrices.
    C_reg = C + λ * I ensures all eigenvalues are at least λ.
    
    Args:
        matrix: Symmetric matrix to regularize.
        lambda_val: Regularization parameter. Default 1e-3 as per the research paper.
    
    Returns:
        Regularized SPD matrix.
    """
    n = matrix.shape[0]
    result = matrix + lambda_val * np.eye(n)
    # Ensure symmetry
    result = 0.5 * (result + result.T)
    return result


def validate_spd(
    matrix: np.ndarray,
    tol: float = 1e-8,
    raise_error: bool = False,
) -> bool:
    """Validate that a matrix is symmetric positive definite.
    
    Args:
        matrix: Matrix to validate.
        tol: Tolerance for eigenvalue positivity and symmetry.
        raise_error: If True, raise ValueError instead of returning False.
    
    Returns:
        True if matrix is SPD.
    
    Raises:
        ValueError: If raise_error=True and matrix is not SPD.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        if raise_error:
            raise ValueError(f"Expected square matrix, got shape {matrix.shape}")
        return False
    
    n = matrix.shape[0]
    
    # Check symmetry
    sym_err = np.max(np.abs(matrix - matrix.T))
    if sym_err > tol:
        if raise_error:
            raise ValueError(f"Matrix not symmetric: max asymmetry = {sym_err}")
        return False
    
    # Check positive definiteness via eigenvalues
    try:
        eigenvalues = la.eigvalsh(matrix)
        min_eig = np.min(eigenvalues)
        if min_eig < -tol:
            if raise_error:
                raise ValueError(f"Matrix not positive definite: min eigenvalue = {min_eig}")
            return False
    except la.LinAlgError:
        if raise_error:
            raise ValueError("Eigenvalue computation failed")
        return False
    
    return True


def nearest_spd(matrix: np.ndarray) -> np.ndarray:
    """Find the nearest SPD matrix using Higham's (2002) algorithm.
    
    This is a fallback when regularization is insufficient.
    Projects the matrix to the SPD cone.
    
    Args:
        matrix: Input square matrix.
    
    Returns:
        Nearest SPD matrix.
    """
    # Symmetrize
    B = 0.5 * (matrix + matrix.T)
    
    # Compute polar decomposition
    try:
        U, S, Vh = la.svd(B)
        H = Vh.T @ np.diag(S) @ Vh
    except la.LinAlgError:
        # Fallback: regularize heavily
        return regularize_spd(B, lambda_val=0.1)
    
    A2 = 0.5 * (B + H)
    
    # Ensure positive definiteness
    eigenvalues, eigenvectors = la.eigh(A2)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def _clip_eigenvalues(matrix: np.ndarray, min_eigenvalue: float = 1e-6) -> np.ndarray:
    """Clip eigenvalues to ensure positive definiteness.
    
    Args:
        matrix: Symmetric matrix.
        min_eigenvalue: Minimum allowed eigenvalue.
    
    Returns:
        Matrix with all eigenvalues >= min_eigenvalue.
    """
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = la.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def spd_frechet_mean(
    matrices: List[np.ndarray],
    max_iter: int = 20,
    tol: float = 1e-6,
    init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the Fréchet mean (Riemannian mean) of SPD matrices.
    
    Uses iterative log-Euclidean approximation:
    1. Start with initial guess (first matrix or provided)
    2. Log-map all matrices to tangent space at current mean
    3. Average the tangent vectors
    4. Exp-map back to manifold
    5. Repeat until convergence
    
    Args:
        matrices: List of SPD matrices.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        init: Initial guess for the mean. If None, uses first matrix.
    
    Returns:
        Fréchet mean SPD matrix.
    """
    if len(matrices) == 0:
        raise ValueError("Cannot compute Fréchet mean of empty list")
    if len(matrices) == 1:
        return matrices[0].copy()
    
    # Initialize
    mean = init if init is not None else matrices[0].copy()
    n = mean.shape[0]
    
    for iteration in range(max_iter):
        # Log-map all matrices to tangent space at current mean
        tangent_vectors = []
        for mat in matrices:
            try:
                tv = _log_map(mat, mean)
                tangent_vectors.append(tv)
            except la.LinAlgError:
                logger.warning(
                    f"Log-map failed at iteration {iteration}, "
                    f"adding regularization"
                )
                reg_mat = regularize_spd(mat)
                tv = _log_map(reg_mat, regularize_spd(mean))
                tangent_vectors.append(tv)
        
        # Average in tangent space (Euclidean mean)
        avg_tangent = np.mean(tangent_vectors, axis=0)
        
        # Check convergence
        step_size = la.norm(avg_tangent, 'fro')
        if step_size < tol:
            logger.debug(f"Fréchet mean converged at iteration {iteration}")
            break
        
        # Exp-map back to manifold
        mean = _exp_map(avg_tangent, mean)
        
        # Ensure SPD
        mean = 0.5 * (mean + mean.T)
        eigenvalues = la.eigvalsh(mean)
        if np.min(eigenvalues) < 1e-10:
            mean = regularize_spd(mean, lambda_val=1e-4)
    
    return mean


def _log_map(point: np.ndarray, base_point: np.ndarray) -> np.ndarray:
    """Log-map: project SPD matrix to tangent space at base_point.
    
    Under the affine-invariant Riemannian metric:
    log_P(Q) = P^{1/2} @ logm(P^{-1/2} @ Q @ P^{-1/2}) @ P^{1/2}
    
    Args:
        point: SPD matrix to project (Q).
        base_point: Reference SPD matrix (P).
    
    Returns:
        Tangent vector (symmetric matrix).
    """
    # Compute P^{-1/2}
    eigvals, eigvecs = la.eigh(base_point)
    eigvals = np.maximum(eigvals, 1e-12)  # Safety
    base_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    
    # Compute P^{-1/2} @ Q @ P^{-1/2}
    inner = base_inv_sqrt @ point @ base_inv_sqrt
    
    # Matrix logarithm
    log_inner = la.logm(inner)
    
    # Map back: P^{1/2} @ logm(...) @ P^{1/2}
    base_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    tangent = base_sqrt @ log_inner @ base_sqrt
    
    return 0.5 * (tangent + tangent.T)  # Ensure symmetry


def _exp_map(tangent_vec: np.ndarray, base_point: np.ndarray) -> np.ndarray:
    """Exp-map: project tangent vector back to SPD manifold.
    
    Under the affine-invariant Riemannian metric:
    exp_P(V) = P^{1/2} @ expm(P^{-1/2} @ V @ P^{-1/2}) @ P^{1/2}
    
    Args:
        tangent_vec: Tangent vector at base_point (V).
        base_point: Reference SPD matrix (P).
    
    Returns:
        SPD matrix on the manifold.
    """
    # Compute P^{-1/2}
    eigvals, eigvecs = la.eigh(base_point)
    eigvals = np.maximum(eigvals, 1e-12)
    base_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    
    # Compute P^{-1/2} @ V @ P^{-1/2}
    inner = base_inv_sqrt @ tangent_vec @ base_inv_sqrt
    
    # Matrix exponential
    exp_inner = la.expm(inner)
    
    # Map back: P^{1/2} @ expm(...) @ P^{1/2}
    base_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    result = base_sqrt @ exp_inner @ base_sqrt
    
    return 0.5 * (result + result.T)


def geodesic_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute the affine-invariant geodesic distance between SPD matrices.
    
    d(P,Q) = ||log(P^{-1/2} @ Q @ P^{-1/2})||_F
    
    Args:
        P: First SPD matrix.
        Q: Second SPD matrix.
    
    Returns:
        Geodesic distance.
    """
    # Compute P^{-1/2}
    eigvals, eigvecs = la.eigh(P)
    eigvals = np.maximum(eigvals, 1e-12)
    P_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    
    # Compute P^{-1/2} @ Q @ P^{-1/2}
    inner = P_inv_sqrt @ Q @ P_inv_sqrt
    
    # Matrix logarithm and Frobenius norm
    log_inner = la.logm(inner)
    return la.norm(log_inner, 'fro')


def parallel_transport(
    tangent_vec: np.ndarray,
    base_point: np.ndarray,
    target_point: np.ndarray,
) -> np.ndarray:
    """Parallel transport of tangent vector along geodesic under AIRM.
    
    Under the affine-invariant metric, parallel transport has a closed-form:
    S' = E @ S @ E^T, where E = (target @ base^{-1})^{1/2}
    
    This is essential for federated aggregation where updates from different
    sites must be transported to a common reference point.
    
    Args:
        tangent_vec: Tangent vector S at base_point.
        base_point: Origin SPD matrix.
        target_point: Destination SPD matrix.
    
    Returns:
        Transported tangent vector at target_point.
    """
    # Compute E = (target @ base^{-1})^{1/2}
    base_inv = la.inv(base_point)
    M = target_point @ base_inv
    
    # Matrix square root of M
    eigvals, eigvecs = la.eigh(M)
    eigvals = np.maximum(eigvals, 1e-12)
    E = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    
    # Parallel transport: S' = E @ S @ E^T
    transported = E @ tangent_vec @ E.T
    
    return 0.5 * (transported + transported.T)
