"""Classical baseline classifiers for SPD matrix classification.

These baselines establish the performance ceiling for classical
methods on the same data, allowing assessment of quantum advantage.

Target: PQFL should exceed 77.3% balanced accuracy (classical ceiling)
and reach >80% balanced accuracy in federated cross-site validation.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TangentSpaceSVM:
    """SVM classifier in tangent space of SPD manifold.
    
    This is the standard baseline for SPD matrix classification:
    1. Project SPD matrices to tangent space at Fréchet mean
    2. Apply PCA for dimensionality reduction
    3. Train SVM with RBF kernel
    
    This matches the classical pipeline in pyRiemann.
    
    Args:
        n_pca_components: Number of PCA components.
        kernel: SVM kernel type.
        C: SVM regularization parameter.
    """
    
    def __init__(
        self,
        n_pca_components: int = 256,
        kernel: str = "rbf",
        C: float = 1.0,
    ):
        self.n_pca_components = n_pca_components
        self.kernel = kernel
        self.C = C
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel=kernel, C=C, probability=True)),
        ])
        self._is_fitted = False
    
    def fit(
        self,
        tangent_features: np.ndarray,
        labels: np.ndarray,
    ) -> "TangentSpaceSVM":
        """Train the SVM on tangent space features.
        
        Args:
            tangent_features: Tangent space features, shape (n_samples, n_features).
            labels: Binary labels (0=HC, 1=SZ).
        
        Returns:
            self
        """
        self.pipeline.fit(tangent_features, labels)
        self._is_fitted = True
        logger.info(f"TangentSpaceSVM fitted on {len(labels)} samples")
        return self
    
    def predict(self, tangent_features: np.ndarray) -> np.ndarray:
        """Predict labels."""
        return self.pipeline.predict(tangent_features)
    
    def predict_proba(self, tangent_features: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.pipeline.predict_proba(tangent_features)
    
    def score(
        self,
        tangent_features: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute balanced accuracy."""
        from sklearn.metrics import balanced_accuracy_score
        predictions = self.predict(tangent_features)
        return balanced_accuracy_score(labels, predictions)


class MDMClassifier:
    """Minimum Distance to Mean (MDM) classifier on SPD manifold.
    
    Computes the Fréchet mean of each class (SZ, HC) and classifies
    new samples by finding the nearest class mean using geodesic distance.
    
    This is the simplest Riemannian classifier and serves as a
    lower bound baseline.
    """
    
    def __init__(self, regularization_lambda: float = 1e-3):
        self.regularization_lambda = regularization_lambda
        self.class_means: Dict[int, np.ndarray] = {}
        self._is_fitted = False
    
    def fit(
        self,
        fc_matrices: np.ndarray,
        labels: np.ndarray,
    ) -> "MDMClassifier":
        """Compute class Fréchet means.
        
        Args:
            fc_matrices: SPD matrices, shape (n_samples, n_rois, n_rois).
            labels: Binary labels.
        
        Returns:
            self
        """
        from ..riemannian.spd_utils import spd_frechet_mean, regularize_spd
        
        for cls in np.unique(labels):
            mask = labels == cls
            class_matrices = [
                regularize_spd(fc_matrices[i], self.regularization_lambda)
                for i in range(len(labels)) if mask[i]
            ]
            self.class_means[cls] = spd_frechet_mean(class_matrices)
        
        self._is_fitted = True
        logger.info(
            f"MDM fitted with {len(self.class_means)} classes: "
            f"{list(self.class_means.keys())}"
        )
        return self
    
    def predict(self, fc_matrices: np.ndarray) -> np.ndarray:
        """Classify by nearest class mean using geodesic distance."""
        from ..riemannian.spd_utils import geodesic_distance, regularize_spd
        
        predictions = np.zeros(len(fc_matrices), dtype=int)
        
        for i in range(len(fc_matrices)):
            mat = regularize_spd(fc_matrices[i], self.regularization_lambda)
            min_dist = float('inf')
            best_class = 0
            
            for cls, mean in self.class_means.items():
                dist = geodesic_distance(mat, mean)
                if dist < min_dist:
                    min_dist = dist
                    best_class = cls
            
            predictions[i] = best_class
        
        return predictions
    
    def score(
        self,
        fc_matrices: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute balanced accuracy."""
        from sklearn.metrics import balanced_accuracy_score
        predictions = self.predict(fc_matrices)
        return balanced_accuracy_score(labels, predictions)


class RiemannianLogisticRegression:
    """Logistic regression in tangent space.
    
    Like TangentSpaceSVM but uses logistic regression, which
    provides probability estimates naturally and is better
    calibrated for clinical applications.
    
    Args:
        n_pca_components: Number of PCA components.
        C: Regularization parameter.
    """
    
    def __init__(
        self,
        n_pca_components: int = 256,
        C: float = 1.0,
    ):
        self.n_pca_components = n_pca_components
        self.C = C
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=C, max_iter=1000, solver="lbfgs")),
        ])
        self._is_fitted = False
    
    def fit(
        self,
        tangent_features: np.ndarray,
        labels: np.ndarray,
    ) -> "RiemannianLogisticRegression":
        """Train logistic regression on tangent features."""
        self.pipeline.fit(tangent_features, labels)
        self._is_fitted = True
        return self
    
    def predict(self, tangent_features: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(tangent_features)
    
    def predict_proba(self, tangent_features: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(tangent_features)
    
    def score(
        self,
        tangent_features: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        from sklearn.metrics import balanced_accuracy_score
        return balanced_accuracy_score(labels, self.predict(tangent_features))
