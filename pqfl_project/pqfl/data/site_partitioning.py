"""Site partitioning for multi-site federated learning simulation.

Handles splitting of combined datasets into site-specific partitions,
simulating the federated learning scenario where data is distributed
across multiple institutions.

For real federated deployment, each site would have its own data.
For simulation/development, we simulate this by partitioning
available datasets according to the 7-site strategy.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from .dataset import FCDataset, SiteFCDataset

logger = logging.getLogger(__name__)


class SitePartitioner:
    """Partition data into site-specific subsets for federated simulation.

    Supports two modes:
    1. Real data: Each site has its own dataset (load separately)
    2. Simulation: Split a combined dataset to simulate federation

    For the 7-site architecture:
    - Training: COBRE (0), FBIRN (1), MCIC (2), LA5c (3), SRPBS (4)
    - Validation: BSNIP-2 (5), TCP 2025 (6)

    Args:
        n_sites: Number of sites.
        validation_site_ids: Set of site IDs for validation.
        partition_strategy: How to partition data:
            - "natural": Use pre-existing site labels
            - "random": Random partition
            - "label_skew": Create non-IID partition by diagnosis ratio
    """

    SITE_NAMES = {
        0: "COBRE",
        1: "FBIRN",
        2: "MCIC",
        3: "LA5c",
        4: "SRPBS",
        5: "BSNIP2",
        6: "TCP2025",
    }

    def __init__(
        self,
        n_sites: int = 7,
        validation_site_ids: Optional[set] = None,
        partition_strategy: str = "natural",
    ):
        self.n_sites = n_sites
        self.validation_site_ids = validation_site_ids or {5, 6}
        self.partition_strategy = partition_strategy

    def create_site_dataset(
        self,
        fc_matrices: np.ndarray,
        labels: np.ndarray,
        site_labels: Optional[np.ndarray] = None,
        site_id: int = 0,
        tangent_features: Optional[np.ndarray] = None,
        fdt_features: Optional[np.ndarray] = None,
    ) -> SiteFCDataset:
        """Create a SiteFCDataset from arrays.

        Args:
            fc_matrices: FC matrices, shape (n_samples, n_rois, n_rois).
            labels: Binary labels, shape (n_samples,).
            site_labels: Optional site membership labels.
            site_id: Site identifier.
            tangent_features: Optional tangent features.
            fdt_features: Optional FDT features.

        Returns:
            SiteFCDataset for the specified site.
        """
        fc_dataset = FCDataset(
            fc_matrices=fc_matrices,
            labels=labels,
            tangent_features=tangent_features,
            fdt_features=fdt_features,
            site_id=site_id,
        )

        site_name = self.SITE_NAMES.get(site_id, f"Site_{site_id}")

        return SiteFCDataset(
            fc_dataset=fc_dataset,
            site_name=site_name,
            site_id=site_id,
        )

    def partition_combined_data(
        self,
        fc_matrices: np.ndarray,
        labels: np.ndarray,
        n_partitions: int = 5,
        site_labels: Optional[np.ndarray] = None,
        tangent_features: Optional[np.ndarray] = None,
        fdt_features: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> Dict[int, SiteFCDataset]:
        """Partition combined data into site-specific datasets.

        For development/simulation: splits data across virtual sites
        to simulate the federated learning scenario.

        Args:
            fc_matrices: Combined FC matrices.
            labels: Combined labels.
            n_partitions: Number of training site partitions.
            site_labels: Pre-existing site membership (for natural partitioning).
            tangent_features: Optional tangent features.
            fdt_features: Optional FDT features.
            seed: Random seed.

        Returns:
            Dictionary mapping site_id to SiteFCDataset.
        """
        rng = np.random.RandomState(seed)
        n_samples = len(labels)

        if self.partition_strategy == "natural" and site_labels is not None:
            return self._natural_partition(
                fc_matrices, labels, site_labels,
                tangent_features, fdt_features,
            )
        elif self.partition_strategy == "label_skew":
            return self._label_skew_partition(
                fc_matrices, labels, n_partitions,
                tangent_features, fdt_features, rng,
            )
        else:  # random
            return self._random_partition(
                fc_matrices, labels, n_partitions,
                tangent_features, fdt_features, rng,
            )

    def _random_partition(
        self,
        fc_matrices, labels, n_partitions,
        tangent_features, fdt_features, rng,
    ) -> Dict[int, SiteFCDataset]:
        """Random uniform partition."""
        indices = rng.permutation(len(labels))
        partition_size = len(labels) // n_partitions

        sites = {}
        for i in range(n_partitions):
            start = i * partition_size
            end = start + partition_size if i < n_partitions - 1 else len(labels)
            idx = indices[start:end]

            sites[i] = self.create_site_dataset(
                fc_matrices=fc_matrices[idx],
                labels=labels[idx],
                site_id=i,
                tangent_features=tangent_features[idx] if tangent_features is not None else None,
                fdt_features=fdt_features[idx] if fdt_features is not None else None,
            )

        return sites

    def _label_skew_partition(
        self,
        fc_matrices, labels, n_partitions,
        tangent_features, fdt_features, rng,
    ) -> Dict[int, SiteFCDataset]:
        """Create non-IID partition with varying diagnosis ratios.

        Simulates the real-world scenario where sites have different
        SZ/HC ratios (non-IID label distribution).
        """
        sz_indices = np.where(labels == 1)[0]
        hc_indices = np.where(labels == 0)[0]

        rng.shuffle(sz_indices)
        rng.shuffle(hc_indices)

        # Different SZ ratios per site (0.2 to 0.8)
        sz_ratios = np.linspace(0.2, 0.8, n_partitions)
        rng.shuffle(sz_ratios)

        sites = {}
        sz_used = 0
        hc_used = 0

        for i in range(n_partitions):
            n_site = len(labels) // n_partitions
            n_sz = int(n_site * sz_ratios[i])
            n_hc = n_site - n_sz

            # Clamp to available
            n_sz = min(n_sz, len(sz_indices) - sz_used)
            n_hc = min(n_hc, len(hc_indices) - hc_used)

            idx = np.concatenate([
                sz_indices[sz_used:sz_used + n_sz],
                hc_indices[hc_used:hc_used + n_hc],
            ])
            sz_used += n_sz
            hc_used += n_hc

            sites[i] = self.create_site_dataset(
                fc_matrices=fc_matrices[idx],
                labels=labels[idx],
                site_id=i,
                tangent_features=tangent_features[idx] if tangent_features is not None else None,
                fdt_features=fdt_features[idx] if fdt_features is not None else None,
            )

        return sites

    def _natural_partition(
        self,
        fc_matrices, labels, site_labels,
        tangent_features, fdt_features,
    ) -> Dict[int, SiteFCDataset]:
        """Partition using pre-existing site labels."""
        unique_sites = np.unique(site_labels)
        sites = {}

        for i, site_label in enumerate(unique_sites):
            mask = site_labels == site_label
            sites[i] = self.create_site_dataset(
                fc_matrices=fc_matrices[mask],
                labels=labels[mask],
                site_id=i,
                tangent_features=tangent_features[mask] if tangent_features is not None else None,
                fdt_features=fdt_features[mask] if fdt_features is not None else None,
            )

        return sites

    @staticmethod
    def generate_synthetic_site(
        site_id: int,
        n_samples: int = 100,
        n_rois: int = 100,
        sz_ratio: float = 0.5,
        signal_strength: float = 0.1,
        seed: int = 42,
    ) -> SiteFCDataset:
        """Generate synthetic FC data for a site (for testing).

        Creates realistic SPD matrices with site-specific noise patterns
        and group differences between SZ and HC.

        Args:
            site_id: Site identifier.
            n_samples: Number of subjects.
            n_rois: Number of ROIs.
            sz_ratio: Proportion of SZ patients.
            signal_strength: Effect size (group separation).
            seed: Random seed.

        Returns:
            SiteFCDataset with synthetic data.
        """
        from scipy import linalg as la

        rng = np.random.RandomState(seed + site_id)

        n_sz = int(n_samples * sz_ratio)
        n_hc = n_samples - n_sz
        labels = np.concatenate([np.ones(n_sz), np.zeros(n_hc)])

        # Generate SPD matrices
        # Base correlation matrix with block structure (network groups)
        base_corr = np.eye(n_rois)
        network_size = n_rois // 7
        for net in range(7):
            start = net * network_size
            end = min(start + network_size, n_rois)
            block = rng.randn(end - start, end - start) * 0.3
            block = block @ block.T  # Make SPD
            # Normalize to correlations
            d = np.sqrt(np.diag(block))
            block = block / np.outer(d, d)
            base_corr[start:end, start:end] = block

        # Site-specific noise
        site_noise = rng.randn(n_rois, n_rois) * 0.05
        site_noise = 0.5 * (site_noise + site_noise.T)

        fc_matrices = np.zeros((n_samples, n_rois, n_rois))

        for i in range(n_samples):
            # Subject-specific variation
            subject_noise = rng.randn(n_rois, n_rois) * 0.1
            subject_noise = 0.5 * (subject_noise + subject_noise.T)

            fc = base_corr + site_noise + subject_noise

            # Add SZ effect: reduced connectivity in certain networks
            if labels[i] == 1:
                sz_effect = np.zeros((n_rois, n_rois))
                # Reduce connectivity in default mode network (network 0)
                dm_start, dm_end = 0, network_size
                sz_effect[dm_start:dm_end, dm_start:dm_end] = -signal_strength
                fc += sz_effect

            # Make SPD
            fc = 0.5 * (fc + fc.T)
            eigenvalues = la.eigvalsh(fc)
            if np.min(eigenvalues) < 1e-6:
                fc += (abs(np.min(eigenvalues)) + 1e-3) * np.eye(n_rois)

            fc_matrices[i] = fc

        fc_dataset = FCDataset(
            fc_matrices=fc_matrices,
            labels=labels,
            site_id=site_id,
        )

        site_name = SitePartitioner.SITE_NAMES.get(site_id, f"Site_{site_id}")

        return SiteFCDataset(
            fc_dataset=fc_dataset,
            site_name=site_name,
            site_id=site_id,
        )
