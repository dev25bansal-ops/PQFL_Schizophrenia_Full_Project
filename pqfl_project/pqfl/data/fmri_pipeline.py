"""fMRI preprocessing pipeline using nilearn.

Provides the online feature extraction stage that follows offline
fMRIPrep preprocessing. The pipeline:
1. Loads preprocessed fMRI data and confounds
2. Applies confound regression and band-pass filtering
3. Extracts ROI time series using Schaefer parcellation
4. Optionally computes functional connectivity matrices

All preprocessing is performed once and cached; only model forward/backward
passes and federated communication happen during training.
"""

import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class FMRIConfig:
    """Configuration for fMRI preprocessing pipeline.

    Attributes:
        parcellation: Atlas name ("schaefer", "aal").
        n_rois: Number of ROIs.
        yeo_networks: Number of Yeo networks (7 or 17).
        tr: Repetition time in seconds.
        bandpass_low: Low frequency cutoff (Hz).
        bandpass_high: High frequency cutoff (Hz).
        fd_threshold: Framewise displacement threshold (mm).
        confound_strategy: Confound regression strategy.
        smoothing_fwhm: Smoothing kernel size (mm). None for no smoothing.
        standardize: Whether to standardize ROI time series.
        detrend: Whether to detrend ROI time series.
    """
    parcellation: str = "schaefer"
    n_rois: int = 100
    yeo_networks: int = 7
    tr: float = 2.0
    bandpass_low: float = 0.01
    bandpass_high: float = 0.08
    fd_threshold: float = 0.5
    confound_strategy: str = "simple"  # simple, scrubbing, acompcor
    smoothing_fwhm: Optional[float] = None
    standardize: bool = True
    detrend: bool = True


class FMRIPipeline:
    """fMRI preprocessing and feature extraction pipeline.

    Uses nilearn for online feature extraction from fMRIPrep-processed data.
    The pipeline extracts ROI time series and computes functional connectivity.

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: Optional[FMRIConfig] = None):
        self.config = config or FMRIConfig()
        self._masker = None
        self._atlas = None

    def _get_atlas(self):
        """Load the parcellation atlas using nilearn."""
        try:
            from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_atlas_aal
        except ImportError:
            logger.warning("nilearn not installed. Using synthetic data mode.")
            return None

        if self.config.parcellation == "schaefer":
            atlas = fetch_atlas_schaefer_2018(
                n_rois=self.config.n_rois,
                yeo_networks=self.config.yeo_networks,
                resolution_mm=2,
            )
        elif self.config.parcellation == "aal":
            atlas = fetch_atlas_aal()
        else:
            raise ValueError(f"Unknown parcellation: {self.config.parcellation}")

        self._atlas = atlas
        return atlas

    def _get_masker(self, disable_bandpass=False):
        """Create a NiftiLabelsMasker for ROI time series extraction.
        
        Args:
            disable_bandpass: If True, skip bandpass filtering in the masker.
                Use this for fMRIPrep preprocessed data or when time series
                is too short for the Butterworth filter.
        """
        try:
            from nilearn.maskers import NiftiLabelsMasker
        except ImportError:
            logger.warning("nilearn not installed. Using synthetic data mode.")
            return None

        atlas = self._get_atlas()
        if atlas is None:
            return None

        # For fMRIPrep data or short time series, disable bandpass in masker
        # and apply it manually afterward with proper length checks
        # Note: nilearn uses low_pass=high-freq-cutoff, high_pass=low-freq-cutoff
        low_pass = self.config.bandpass_high if not disable_bandpass else None
        high_pass = self.config.bandpass_low if not disable_bandpass else None

        # Use 'zscore_sample' instead of True (deprecated in nilearn >= 0.10)
        standardize_mode = 'zscore_sample' if self.config.standardize else False

        self._masker = NiftiLabelsMasker(
            labels_img=atlas.maps,
            standardize=standardize_mode,
            detrend=self.config.detrend,
            smoothing_fwhm=self.config.smoothing_fwhm,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=self.config.tr if not disable_bandpass else None,
            memory="nilearn_cache",
            memory_level=1,
            verbose=0,
        )

        return self._masker

    def extract_time_series(
        self,
        fmri_img,
        confounds=None,
        is_fmriprep=False,
    ) -> np.ndarray:
        """Extract ROI time series from preprocessed fMRI data.

        Args:
            fmri_img: Preprocessed BOLD image (NIfTI or nilearn image).
            confounds: Confound regressors (from fMRIPrep).
            is_fmriprep: If True, data is fMRIPrep preprocessed (skip bandpass
                in masker to avoid filter errors, apply manually after).

        Returns:
            ROI time series, shape (n_timepoints, n_rois).
        """
        # Validate that the input is 4D (not a 3D mask file)
        try:
            import nibabel as nib
            if hasattr(fmri_img, 'shape'):
                img_shape = fmri_img.shape
            else:
                img_shape = nib.load(str(fmri_img)).shape
            
            if len(img_shape) < 4:
                logger.error(
                    f"Input image is {len(img_shape)}D with shape {img_shape}. "
                    f"Expected 4D BOLD time series. "
                    f"You may have loaded a brain mask instead of a preproc file."
                )
                return np.array([]).reshape(0, self.config.n_rois)
            
            n_vols = img_shape[3] if len(img_shape) == 4 else 1
            logger.info(f"    BOLD shape: {img_shape} ({n_vols} volumes)")
            
            if n_vols < 20:
                logger.warning(
                    f"Very short BOLD run: only {n_vols} volumes. "
                    f"Skipping this subject."
                )
                return np.array([]).reshape(0, self.config.n_rois)
        except Exception as e:
            logger.debug(f"Could not validate image dimensions: {e}")

        # For fMRIPrep data, disable bandpass in masker to avoid
        # "padlen" errors, then apply bandpass manually afterward
        masker = self._get_masker(disable_bandpass=is_fmriprep)

        if masker is None:
            # Synthetic data mode for testing without nilearn
            logger.warning("Using synthetic time series (nilearn not available)")
            return np.random.randn(200, self.config.n_rois) * 0.1

        time_series = masker.fit_transform(fmri_img, confounds=confounds)
        
        n_tp = time_series.shape[0]
        logger.info(f"    Extracted time series: {time_series.shape} ({n_tp} timepoints × {time_series.shape[1]} ROIs)")
        
        # Apply bandpass filtering manually if we skipped it in the masker
        if is_fmriprep and self.config.bandpass_low is not None and self.config.bandpass_high is not None:
            time_series = self._apply_bandpass(time_series)

        return time_series

    def _apply_bandpass(self, time_series: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to time series with proper length checks.
        
        Args:
            time_series: ROI time series, shape (n_timepoints, n_rois).
            
        Returns:
            Filtered time series, same shape.
        """
        from scipy.signal import butter, filtfilt
        
        tr = self.config.tr
        n_tp = time_series.shape[0]
        
        # Nyquist frequency
        fs = 1.0 / tr
        nyq = fs / 2.0
        
        # Check if time series is long enough for the filter
        # filtfilt needs at least padlen + 1 timepoints
        # For a 3rd order Butterworth, padlen is typically ~33
        # We also need enough timepoints for meaningful spectral estimation
        # At TR=2s and 0.01 Hz, one cycle = 100s = 50 TRs, so ~50 TP minimum
        min_timepoints = 50  # Minimum for reasonable spectral content
        
        if n_tp < min_timepoints:
            logger.warning(
                f"Time series too short for bandpass filtering "
                f"({n_tp} < {min_timepoints} timepoints). "
                f"Skipping bandpass filter."
            )
            return time_series
        
        try:
            low = self.config.bandpass_low / nyq
            high = self.config.bandpass_high / nyq
            
            # Ensure frequencies are in valid range
            if low <= 0 or high <= 0 or low >= 1 or high >= 1 or low >= high:
                logger.warning(
                    f"Invalid bandpass frequencies: {self.config.bandpass_low}-{self.config.bandpass_high} Hz "
                    f"(Nyquist={nyq:.2f} Hz). Skipping filter."
                )
                return time_series
            
            b, a = butter(3, [low, high], btype='band')
            filtered = filtfilt(b, a, time_series, axis=0, padlen=min(33, n_tp - 1))
            return filtered
        except ValueError as e:
            logger.warning(f"Bandpass filtering failed: {e}. Using unfiltered time series.")
            return time_series

    def compute_fd(self, confounds: np.ndarray) -> np.ndarray:
        """Compute Framewise Displacement from motion parameters.

        Args:
            confounds: Confound array including motion parameters.

        Returns:
            Framewise displacement values, shape (n_timepoints,).
        """
        # FD = |dx| + |dy| + |dz| + |dα| + |dβ| + |dγ|
        # This is a simplified version; nilearn provides more accurate computation
        if confounds.shape[1] >= 6:
            motion = confounds[:, :6]
            fd = np.sum(np.abs(np.diff(motion, axis=0)), axis=1)
            fd = np.concatenate([[0], fd])  # First timepoint
        else:
            fd = np.zeros(confounds.shape[0])

        return fd

    def scrub_timepoints(
        self,
        time_series: np.ndarray,
        fd: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove timepoints with excessive motion.

        Args:
            time_series: ROI time series, shape (n_tp, n_rois).
            fd: Framewise displacement, shape (n_tp,).

        Returns:
            Tuple of (scrubbed_time_series, valid_mask).
        """
        valid_mask = fd < self.config.fd_threshold
        scrubbed = time_series[valid_mask]

        n_removed = (~valid_mask).sum()
        if n_removed > 0:
            logger.info(
                f"Scrubbed {n_removed}/{len(fd)} timepoints "
                f"(FD > {self.config.fd_threshold}mm)"
            )

        return scrubbed, valid_mask

    def get_network_labels(self) -> np.ndarray:
        """Get Yeo network labels for each ROI.

        Returns:
            Array of network labels, shape (n_rois,).
        """
        if self.config.parcellation == "schaefer":
            # Schaefer parcellation includes network labels
            # Default: 7 Yeo networks distributed across ROIs
            labels_per_network = self.config.n_rois // self.config.yeo_networks
            remainder = self.config.n_rois % self.config.yeo_networks
            labels = []
            for i in range(self.config.yeo_networks):
                count = labels_per_network + (1 if i < remainder else 0)
                labels.extend([i] * count)
            return np.array(labels)
        else:
            return np.arange(self.config.n_rois)

    def get_config(self) -> Dict:
        """Return pipeline configuration."""
        return {
            "parcellation": self.config.parcellation,
            "n_rois": self.config.n_rois,
            "yeo_networks": self.config.yeo_networks,
            "bandpass": f"{self.config.bandpass_low}-{self.config.bandpass_high} Hz",
            "fd_threshold": self.config.fd_threshold,
        }
