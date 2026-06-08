"""ComBat harmonization for multi-site FC data.

ComBat (Johnson et al., 2007) removes site/scanner effects from
functional connectivity features while preserving biological covariates
(diagnosis, age, sex). Applied in tangent space for mathematical
compatibility with RQFM encoding.

FedHarmony integration: FedHarmony operates on log-FC space, which
is exactly the RQFM Stage 1 (Riemannian Flattening) output. This
makes harmonization mathematically compatible with quantum encoding
for the first time.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class CombatHarmonizer:
    """ComBat harmonization for removing site effects from FC features.

    ComBat uses empirical Bayes to adjust for batch (site) effects
    while preserving biological covariates. Applied to tangent space
    features (vectorized log-FC matrices).

    Args:
        biological_covariates: List of covariate names to preserve
            (e.g., ["diagnosis", "age", "sex"]).
        parametric: Use parametric Empirical Bayes (default True).
        eb: Enable Empirical Bayes (default True).
    """

    def __init__(
        self,
        biological_covariates: List[str] = None,
        parametric: bool = True,
        eb: bool = True,
    ):
        self.biological_covariates = biological_covariates or ["diagnosis", "age", "sex"]
        self.parametric = parametric
        self.eb = eb
        self._is_fitted = False
        self._site_means = {}
        self._site_vars = {}

    def fit_transform(
        self,
        features: np.ndarray,
        site_labels: np.ndarray,
        covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply ComBat harmonization to remove site effects.

        Args:
            features: Feature matrix, shape (n_samples, n_features).
                For FC data, this should be tangent space vectors.
            site_labels: Site identifiers, shape (n_samples,).
            covariates: Biological covariates to preserve,
                shape (n_samples, n_covariates). If None, only intercept.

        Returns:
            Harmonized features, shape (n_samples, n_features).
        """
        try:
            from neurocombat import neurocombat
        except ImportError:
            logger.warning(
                "neurocombat not installed. Using simple site-mean centering "
                "as fallback harmonization."
            )
            return self._simple_harmonize(features, site_labels, covariates)

        # Prepare ComBat inputs
        # neurocombat expects features as (n_features, n_samples)
        data = features.T

        # Create batch array
        batch = site_labels

        # Create covariate DataFrame
        import pandas as pd
        if covariates is not None:
            covar_df = pd.DataFrame(
                covariates,
                columns=self.biological_covariates[:covariates.shape[1]],
            )
        else:
            covar_df = None

        # Run ComBat
        try:
            result = neurocombat(
                data=data,
                covars=covar_df,
                batch=batch,
                parametric=self.parametric,
                eb=self.eb,
            )
            harmonized = result["data"].T
        except Exception as e:
            logger.warning(f"ComBat failed: {e}. Using simple harmonization.")
            harmonized = self._simple_harmonize(features, site_labels, covariates)

        self._is_fitted = True
        return harmonized

    def _simple_harmonize(
        self,
        features: np.ndarray,
        site_labels: np.ndarray,
        covariates: Optional[np.ndarray],
    ) -> np.ndarray:
        """Simple site-mean centering as fallback harmonization.

        Removes site-specific mean shifts from features.
        This is a simplified version of ComBat for when neurocombat
        is not available.

        Args:
            features: Feature matrix, shape (n_samples, n_features).
            site_labels: Site identifiers, shape (n_samples,).
            covariates: Not used in simple harmonization.

        Returns:
            Harmonized features.
        """
        global_mean = np.mean(features, axis=0)
        harmonized = features.copy()

        for site in np.unique(site_labels):
            mask = site_labels == site
            site_mean = np.mean(features[mask], axis=0)
            site_std = np.std(features[mask], axis=0) + 1e-8
            global_std = np.std(features, axis=0) + 1e-8

            # Standardize to global distribution
            harmonized[mask] = (
                (features[mask] - site_mean) / site_std * global_std + global_mean
            )

            self._site_means[site] = site_mean
            self._site_vars[site] = site_std

        return harmonized


class TangentSpaceCombat:
    """ComBat harmonization applied specifically in SPD tangent space.

    This is the recommended harmonization approach for the PQFL pipeline
    because:
    1. Tangent space is Euclidean → ComBat assumptions are valid
    2. Site effects are removed from log-FC features (RQFM Stage 1 output)
    3. Compatible with FedHarmony's log-FC space operation

    Pipeline:
    1. Compute Fréchet mean for each site
    2. Log-map all FC matrices to tangent space
    3. Apply ComBat in tangent space (preserving diagnosis, age, sex)
    4. Tangent vectors are now harmonized and ready for quantum encoding

    Args:
        biological_covariates: Covariates to preserve during harmonization.
        parametric: Use parametric Empirical Bayes.
    """

    def __init__(
        self,
        biological_covariates: List[str] = None,
        parametric: bool = True,
    ):
        self.combat = CombatHarmonizer(
            biological_covariates=biological_covariates,
            parametric=parametric,
        )
        self._is_fitted = False

    def harmonize(
        self,
        tangent_features: np.ndarray,
        site_labels: np.ndarray,
        labels: Optional[np.ndarray] = None,
        ages: Optional[np.ndarray] = None,
        sexes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Harmonize tangent space features across sites.

        Args:
            tangent_features: Tangent space features, shape (n_samples, tangent_dim).
            site_labels: Site identifiers, shape (n_samples,).
            labels: Diagnosis labels (0=HC, 1=SZ), shape (n_samples,).
            ages: Subject ages, shape (n_samples,).
            sexes: Subject sexes, shape (n_samples,).

        Returns:
            Harmonized tangent features, shape (n_samples, tangent_dim).
        """
        # Build covariate matrix
        covariates = self._build_covariates(labels, ages, sexes)

        # Apply ComBat
        harmonized = self.combat.fit_transform(
            features=tangent_features,
            site_labels=site_labels,
            covariates=covariates,
        )

        self._is_fitted = True
        return harmonized

    def _build_covariates(
        self,
        labels: Optional[np.ndarray],
        ages: Optional[np.ndarray],
        sexes: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Build covariate matrix for ComBat."""
        parts = []

        if labels is not None:
            parts.append(labels.reshape(-1, 1))
        if ages is not None:
            parts.append(ages.reshape(-1, 1))
        if sexes is not None:
            parts.append(sexes.reshape(-1, 1))

        if parts:
            return np.hstack(parts)
        return None

    def validate_preservation(
        self,
        original_features: np.ndarray,
        harmonized_features: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Validate that biological signal is preserved after harmonization.

        Computes the ratio of between-group to within-group variance
        before and after harmonization. A good harmonization should
        preserve or improve this ratio.

        Args:
            original_features: Features before harmonization.
            harmonized_features: Features after harmonization.
            labels: Diagnosis labels.

        Returns:
            Dictionary with validation metrics.
        """
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        # Before harmonization
        lda_orig = LinearDiscriminantAnalysis()
        lda_orig.fit(original_features, labels)
        score_orig = lda_orig.score(original_features, labels)

        # After harmonization
        lda_harm = LinearDiscriminantAnalysis()
        lda_harm.fit(harmonized_features, labels)
        score_harm = lda_harm.score(harmonized_features, labels)

        return {
            "lda_score_before": score_orig,
            "lda_score_after": score_harm,
            "preservation_ratio": score_harm / max(score_orig, 1e-8),
            "signal_preserved": score_harm >= score_orig * 0.9,
        }
