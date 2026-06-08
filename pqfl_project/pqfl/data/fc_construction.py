"""Functional connectivity matrix construction from fMRI time series.

Constructs symmetric positive definite (SPD) functional connectivity
matrices using Pearson correlation, with regularization to ensure
positive definiteness.

Also provides:
- Frequency-Dependent Topology (FDT) feature extraction
- Dynamic FC computation with sliding windows
- SLRCPD tensor decomposition for dynamic FC states
"""

import numpy as np
import scipy.linalg as la
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class FCConstructor:
    """Functional connectivity matrix constructor.

    Args:
        n_rois: Number of regions of interest.
        regularization_lambda: λ for C + λI regularization (default 1e-3).
        fc_method: FC computation method: "pearson", "partial", "tangent".
        dynamic_window_size: Window size for dynamic FC (in TRs). None for static only.
        dynamic_step_size: Step size for sliding window.
    """

    def __init__(
        self,
        n_rois: int = 100,
        regularization_lambda: float = 1e-3,
        fc_method: str = "pearson",
        dynamic_window_size: Optional[int] = None,
        dynamic_step_size: int = 1,
    ):
        self.n_rois = n_rois
        self.regularization_lambda = regularization_lambda
        self.fc_method = fc_method
        self.dynamic_window_size = dynamic_window_size
        self.dynamic_step_size = dynamic_step_size

    def compute_static_fc(
        self,
        time_series: np.ndarray,
    ) -> np.ndarray:
        """Compute static functional connectivity matrix.

        Args:
            time_series: ROI time series, shape (n_timepoints, n_rois).

        Returns:
            Regularized SPD correlation matrix, shape (n_rois, n_rois).
        """
        return compute_fc_matrix(
            time_series,
            method=self.fc_method,
            regularization_lambda=self.regularization_lambda,
        )

    def compute_dynamic_fc(
        self,
        time_series: np.ndarray,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Compute dynamic functional connectivity using sliding windows.

        Args:
            time_series: ROI time series, shape (n_timepoints, n_rois).

        Returns:
            Tuple of:
            - Mean FC matrix, shape (n_rois, n_rois)
            - List of windowed FC matrices, each shape (n_rois, n_rois)
        """
        if self.dynamic_window_size is None:
            raise ValueError("dynamic_window_size must be set for dynamic FC")

        n_tp = time_series.shape[0]
        window_fcs = []

        for start in range(0, n_tp - self.dynamic_window_size + 1, self.dynamic_step_size):
            window_ts = time_series[start:start + self.dynamic_window_size]
            fc = compute_fc_matrix(
                window_ts,
                method=self.fc_method,
                regularization_lambda=self.regularization_lambda,
            )
            window_fcs.append(fc)

        # Mean FC across windows
        mean_fc = np.mean(window_fcs, axis=0)
        mean_fc = 0.5 * (mean_fc + mean_fc.T)  # Ensure symmetry

        return mean_fc, window_fcs

    def compute_fdt_features(
        self,
        time_series: np.ndarray,
        n_top: int = 20,
        frequency_bands: Optional[List[Tuple[float, float]]] = None,
        tr: float = 2.0,
    ) -> np.ndarray:
        """Compute Frequency-Dependent Topology (FDT) features.

        FDT captures how connectivity patterns vary across frequency bands,
        providing complementary information to static FC.

        Args:
            time_series: ROI time series, shape (n_timepoints, n_rois).
            n_top: Number of top deviating regions to extract.
            frequency_bands: List of (low, high) frequency bands in Hz.
                Default: slow-5 (0.01-0.027), slow-4 (0.027-0.073), slow-3 (0.073-0.17).
            tr: Repetition time in seconds (for correct frequency calculation).

        Returns:
            FDT features, shape (n_top,).
        """
        if frequency_bands is None:
            frequency_bands = [
                (0.01, 0.027),   # Slow-5
                (0.027, 0.073),  # Slow-4
                (0.073, 0.17),   # Slow-3
            ]

        n_tp = time_series.shape[0]

        # Compute FC in each frequency band
        band_fcs = []
        for low, high in frequency_bands:
            try:
                # Frequency-domain filtering using FFT
                # rfftfreq returns frequencies in Hz when given sampling interval
                fft_ts = np.fft.rfft(time_series, axis=0)
                freqs = np.fft.rfftfreq(n_tp, d=tr)  # Frequencies in Hz
                band_mask = (freqs >= low) & (freqs <= high)
                filtered_fft = np.zeros_like(fft_ts)
                filtered_fft[band_mask] = fft_ts[band_mask]
                filtered_ts = np.fft.irfft(filtered_fft, n=n_tp, axis=0)

                # Check that filtered signal isn't all zeros (band out of range)
                if np.abs(filtered_ts).max() < 1e-10:
                    logger.warning(
                        f"FDT band {low}-{high} Hz produced near-zero signal. "
                        f"Using full-spectrum FC as fallback."
                    )
                    band_fcs.append(
                        compute_fc_matrix(
                            time_series,
                            method="pearson",
                            regularization_lambda=self.regularization_lambda,
                        )
                    )
                    continue

                fc = compute_fc_matrix(
                    filtered_ts,
                    method="pearson",
                    regularization_lambda=self.regularization_lambda,
                )
                band_fcs.append(fc)
            except Exception as e:
                # Fallback: use full-spectrum FC
                logger.debug(f"FDT band {low}-{high} Hz failed: {e}. Using full-spectrum FC.")
                band_fcs.append(
                    compute_fc_matrix(
                        time_series,
                        method="pearson",
                        regularization_lambda=self.regularization_lambda,
                    )
                )

        # Compute deviation from mean across bands
        mean_band_fc = np.mean(band_fcs, axis=0)

        # Sum of absolute deviations per region
        region_deviation = np.zeros(self.n_rois)
        for band_fc in band_fcs:
            deviation = np.abs(band_fc - mean_band_fc)
            region_deviation += np.sum(deviation, axis=1)

        # Select top-k deviating regions
        top_indices = np.argsort(region_deviation)[-n_top:]
        fdt_features = region_deviation[top_indices]

        return fdt_features


def compute_fc_matrix(
    time_series: np.ndarray,
    method: str = "pearson",
    regularization_lambda: float = 1e-3,
) -> np.ndarray:
    """Compute a functional connectivity matrix from ROI time series.

    Args:
        time_series: ROI time series, shape (n_timepoints, n_rois).
        method: FC computation method.
            - "pearson": Pearson correlation coefficient (standard).
            - "partial": Partial correlation (inverse covariance).
            - "covariance": Covariance matrix.
        regularization_lambda: Regularization parameter for C + λI.

    Returns:
        Regularized SPD matrix, shape (n_rois, n_rois).
    """
    if time_series.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {time_series.shape}")

    n_tp, n_rois = time_series.shape

    if method == "pearson":
        # Pearson correlation: standard FC measure
        fc = np.corrcoef(time_series.T)
    elif method == "partial":
        # Partial correlation via precision matrix
        cov = np.cov(time_series.T)
        # Regularize covariance before inversion
        cov_reg = cov + regularization_lambda * np.eye(n_rois)
        try:
            precision = la.inv(cov_reg)
            # Convert precision to partial correlation
            d = np.diag(precision)
            fc = -precision / np.sqrt(np.outer(d, d))
            np.fill_diagonal(fc, 1.0)
        except la.LinAlgError:
            logger.warning("Precision matrix computation failed, falling back to Pearson")
            fc = np.corrcoef(time_series.T)
    elif method == "covariance":
        fc = np.cov(time_series.T)
    else:
        raise ValueError(f"Unknown FC method: {method}")

    # Ensure symmetry
    fc = 0.5 * (fc + fc.T)

    # Regularize for positive definiteness
    fc = fc + regularization_lambda * np.eye(n_rois)

    # Validate
    eigenvalues = la.eigvalsh(fc)
    if np.min(eigenvalues) < 0:
        logger.warning(
            f"FC matrix not positive definite (min eig={np.min(eigenvalues):.2e}), "
            f"applying stronger regularization"
        )
        fc = fc + (abs(np.min(eigenvalues)) + regularization_lambda) * np.eye(n_rois)

    return fc
