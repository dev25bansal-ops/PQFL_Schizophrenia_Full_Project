"""Tests for the Riemannian geometry module."""

import sys
from pathlib import Path
import numpy as np
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pqfl.riemannian.spd_utils import (
    ensure_spd,
    regularize_spd,
    validate_spd,
    spd_frechet_mean,
    geodesic_distance,
    parallel_transport,
    _log_map,
    _exp_map,
)
from pqfl.riemannian.tangent_space import TangentSpaceProjector, TangentPCA
from pqfl.riemannian.engine import RiemannianEngine
from pqfl.riemannian.aggregation import RiemannianAggregator


def _random_spd(n, seed=42):
    """Generate a random SPD matrix."""
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n)
    return A @ A.T + 1e-3 * np.eye(n)


class TestSPDUtils:
    """Tests for SPD matrix utilities."""
    
    def test_regularize_spd(self):
        """Test C + λI regularization."""
        n = 10
        mat = np.random.randn(n, n)
        mat = 0.5 * (mat + mat.T)
        result = regularize_spd(mat, lambda_val=1e-3)
        assert validate_spd(result), "Regularized matrix should be SPD"
    
    def test_validate_spd(self):
        """Test SPD validation."""
        n = 5
        spd = _random_spd(n)
        assert validate_spd(spd), "Random SPD should be valid"
        
        non_spd = np.array([[1, 2], [2, 1]])  # Eigenvalues: 3, -1
        assert not validate_spd(non_spd), "Non-SPD should be detected"
    
    def test_ensure_spd(self):
        """Test ensure_spd with different methods."""
        n = 5
        mat = np.random.randn(n, n)
        mat = 0.5 * (mat + mat.T)
        
        for method in ["regularize", "nearest", "clip"]:
            result = ensure_spd(mat, method=method)
            assert validate_spd(result), f"ensure_spd({method}) should produce SPD"
    
    def test_log_exp_roundtrip(self):
        """Test log-map / exp-map roundtrip."""
        n = 10
        P = _random_spd(n, seed=1)
        Q = _random_spd(n, seed=2)
        
        tangent = _log_map(Q, P)
        Q_reconstructed = _exp_map(tangent, P)
        
        error = np.linalg.norm(Q - Q_reconstructed, 'fro') / np.linalg.norm(Q, 'fro')
        assert error < 1e-10, f"Roundtrip error too large: {error}"
    
    def test_geodesic_distance(self):
        """Test geodesic distance properties."""
        n = 5
        P = _random_spd(n, seed=1)
        Q = _random_spd(n, seed=2)
        
        d_PQ = geodesic_distance(P, Q)
        d_QP = geodesic_distance(Q, P)
        
        assert d_PQ > 0, "Distance should be positive for different matrices"
        assert abs(d_PQ - d_QP) < 1e-10, "Distance should be symmetric"
    
    def test_frechet_mean(self):
        """Test Fréchet mean computation."""
        n = 5
        matrices = [_random_spd(n, seed=i) for i in range(5)]
        mean = spd_frechet_mean(matrices)
        
        assert validate_spd(mean), "Fréchet mean should be SPD"
        assert mean.shape == (n, n), "Mean should have correct shape"
    
    def test_parallel_transport(self):
        """Test parallel transport under AIRM."""
        n = 5
        base = _random_spd(n, seed=1)
        target = _random_spd(n, seed=2)
        tangent = np.random.randn(n, n)
        tangent = 0.5 * (tangent + tangent.T)
        
        transported = parallel_transport(tangent, base, target)
        assert transported.shape == tangent.shape, "Shape should be preserved"
        
        # Transported should be symmetric
        assert np.allclose(transported, transported.T), "Should be symmetric"


class TestTangentSpace:
    """Tests for tangent space projection."""
    
    def test_projector_roundtrip(self):
        """Test tangent space projection roundtrip."""
        n = 10
        n_samples = 5
        matrices = np.array([_random_spd(n, seed=i) for i in range(n_samples)])
        
        projector = TangentSpaceProjector(n_rois=n)
        tangent = projector.fit_transform(matrices)
        
        assert tangent.shape == (n_samples, n * (n + 1) // 2)
        
        reconstructed = projector.inverse_transform(tangent)
        assert reconstructed.shape == (n_samples, n, n)
    
    def test_tangent_pca(self):
        """Test tangent PCA."""
        n = 20
        n_samples = 30
        n_components = 5
        matrices = np.array([_random_spd(n, seed=i) for i in range(n_samples)])
        
        tpca = TangentPCA(n_components=n_components, n_rois=n)
        reduced = tpca.fit_transform(matrices)
        
        assert reduced.shape == (n_samples, n_components)
        assert tpca.total_explained_variance is not None
        assert tpca.total_explained_variance > 0


class TestRiemannianEngine:
    """Tests for the Riemannian engine."""
    
    def test_full_pipeline(self):
        """Test the complete Riemannian engine pipeline."""
        n = 20
        n_samples = 30
        n_components = 8
        matrices = np.array([_random_spd(n, seed=i) for i in range(n_samples)])
        
        engine = RiemannianEngine(n_rois=n, n_components=n_components)
        features = engine.fit_transform(matrices, site_id=0)
        
        assert features.shape == (n_samples, n_components)
        assert features.dtype == np.float32 or torch.is_tensor(features)
    
    def test_multi_site(self):
        """Test multi-site federated reference points."""
        n = 10
        n_samples = 10
        
        engine = RiemannianEngine(n_rois=n, n_components=5)
        
        for site_id in range(3):
            matrices = np.array([_random_spd(n, seed=i + site_id * 100) for i in range(n_samples)])
            engine.fit(matrices, site_id=site_id)
        
        global_ref = engine.compute_global_reference()
        assert global_ref.shape == (n, n)


class TestAggregation:
    """Tests for Riemannian aggregation strategies."""
    
    def test_frechet_aggregation(self):
        """Test Fréchet mean aggregation."""
        n = 5
        matrices = [_random_spd(n, seed=i) for i in range(3)]
        
        agg = RiemannianAggregator(strategy="frechet")
        result = agg.aggregate(matrices)
        
        assert validate_spd(result), "Aggregated result should be SPD"
    
    def test_all_strategies(self):
        """Test all aggregation strategies."""
        n = 5
        matrices = [_random_spd(n, seed=i) for i in range(3)]
        
        for strategy in ["frechet", "proj_avg", "rl_avg", "log_euclidean"]:
            agg = RiemannianAggregator(strategy=strategy)
            result = agg.aggregate(matrices)
            assert result.shape == (n, n), f"{strategy} should return correct shape"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
