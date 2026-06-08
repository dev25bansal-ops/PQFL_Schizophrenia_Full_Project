"""PyTorch datasets for functional connectivity data.

Provides dataset classes for:
- Single-site FC data (FCDataset)
- Site-partitioned FC data (SiteFCDataset)
- Multi-site federated FC data (MultiSiteDataset)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FCDataset(Dataset):
    """PyTorch dataset for functional connectivity matrices.

    Stores FC matrices and labels, with optional tangent space features
    and FDT features. Supports both raw SPD matrices and pre-processed
    tangent space features.

    Args:
        fc_matrices: FC matrices, shape (n_samples, n_rois, n_rois).
        labels: Binary labels (0=HC, 1=SZ), shape (n_samples,).
        tangent_features: Optional pre-computed tangent features, shape (n_samples, n_features).
        fdt_features: Optional FDT features, shape (n_samples, n_fdt).
        site_id: Optional site identifier for federated learning.
        subject_ids: Optional subject identifiers.
    """

    def __init__(
        self,
        fc_matrices: np.ndarray,
        labels: np.ndarray,
        tangent_features: Optional[np.ndarray] = None,
        fdt_features: Optional[np.ndarray] = None,
        site_id: Optional[int] = None,
        subject_ids: Optional[List[str]] = None,
    ):
        self.fc_matrices = fc_matrices
        self.labels = labels
        self.tangent_features = tangent_features
        self.fdt_features = fdt_features
        self.site_id = site_id
        self.subject_ids = subject_ids

        self.n_samples = len(labels)
        self.n_rois = fc_matrices.shape[1]

        # Class distribution
        self.n_sz = int(labels.sum())
        self.n_hc = self.n_samples - self.n_sz

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Returns:
            Dictionary with keys:
            - "fc_matrix": FC matrix, shape (n_rois, n_rois)
            - "label": Binary label (0=HC, 1=SZ)
            - "tangent_features": Tangent features (if available)
            - "fdt_features": FDT features (if available)
        """
        sample = {
            "fc_matrix": torch.tensor(self.fc_matrices[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

        if self.tangent_features is not None:
            sample["tangent_features"] = torch.tensor(
                self.tangent_features[idx], dtype=torch.float32
            )

        if self.fdt_features is not None:
            sample["fdt_features"] = torch.tensor(
                self.fdt_features[idx], dtype=torch.float32
            )

        return sample

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced classification.

        Returns:
            Class weights, shape (2,).
        """
        total = self.n_samples
        weights = torch.tensor([
            total / (2 * self.n_hc) if self.n_hc > 0 else 1.0,
            total / (2 * self.n_sz) if self.n_sz > 0 else 1.0,
        ])
        return weights

    def get_data_for_riemannian(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get raw FC matrices and labels for Riemannian processing.

        Returns:
            Tuple of (fc_matrices, labels).
        """
        return self.fc_matrices, self.labels

    def set_tangent_features(self, tangent_features: np.ndarray) -> None:
        """Set pre-computed tangent features."""
        self.tangent_features = tangent_features

    def set_fdt_features(self, fdt_features: np.ndarray) -> None:
        """Set FDT features."""
        self.fdt_features = fdt_features

    def split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42,
        stratified: bool = True,
    ) -> Tuple["FCDataset", "FCDataset"]:
        """Split into train and validation sets.

        Args:
            train_ratio: Fraction for training.
            seed: Random seed.
            stratified: If True, preserve class ratio in both splits.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        if stratified and self.n_sz > 0 and self.n_hc > 0:
            # Stratified split: preserve SZ/HC ratio in both splits
            rng = np.random.RandomState(seed)
            sz_indices = np.where(self.labels == 1)[0]
            hc_indices = np.where(self.labels == 0)[0]
            
            # Shuffle each class separately
            rng.shuffle(sz_indices)
            rng.shuffle(hc_indices)
            
            n_train_sz = max(1, int(len(sz_indices) * train_ratio))
            n_train_hc = max(1, int(len(hc_indices) * train_ratio))
            
            train_idx = np.concatenate([
                sz_indices[:n_train_sz],
                hc_indices[:n_train_hc],
            ])
            val_idx = np.concatenate([
                sz_indices[n_train_sz:],
                hc_indices[n_train_hc:],
            ])
            
            # Shuffle the combined indices
            rng.shuffle(train_idx)
            rng.shuffle(val_idx)
        else:
            # Random split (fallback)
            rng = np.random.RandomState(seed)
            indices = rng.permutation(self.n_samples)
            n_train = int(self.n_samples * train_ratio)
            train_idx = indices[:n_train]
            val_idx = indices[n_train:]

        train_ds = self._subset(train_idx)
        val_ds = self._subset(val_idx)

        return train_ds, val_ds

    def _subset(self, indices: np.ndarray) -> "FCDataset":
        """Create a subset dataset."""
        return FCDataset(
            fc_matrices=self.fc_matrices[indices],
            labels=self.labels[indices],
            tangent_features=(
                self.tangent_features[indices]
                if self.tangent_features is not None
                else None
            ),
            fdt_features=(
                self.fdt_features[indices]
                if self.fdt_features is not None
                else None
            ),
            site_id=self.site_id,
        )

    def get_summary(self) -> Dict:
        """Return dataset summary statistics."""
        return {
            "n_samples": self.n_samples,
            "n_rois": self.n_rois,
            "n_sz": self.n_sz,
            "n_hc": self.n_hc,
            "sz_ratio": self.n_sz / self.n_samples,
            "site_id": self.site_id,
            "has_tangent_features": self.tangent_features is not None,
            "has_fdt_features": self.fdt_features is not None,
        }


class SiteFCDataset(Dataset):
    """Dataset for a single site in the federated learning setup.

    Wraps an FCDataset with site-specific information and
    local train/val splitting.

    Args:
        fc_dataset: Base FCDataset for this site.
        site_name: Human-readable site name.
        site_id: Numeric site identifier.
    """

    def __init__(
        self,
        fc_dataset: FCDataset,
        site_name: str,
        site_id: int,
    ):
        self.dataset = fc_dataset
        self.site_name = site_name
        self.site_id = site_id
        fc_dataset.site_id = site_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.dataset[idx]

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """Create a DataLoader for this site.

        Args:
            batch_size: Batch size.
            shuffle: Whether to shuffle.
            num_workers: Number of data loading workers.

        Returns:
            PyTorch DataLoader.
        """
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )


class MultiSiteDataset:
    """Multi-site federated dataset manager.

    Manages data across multiple sites for federated learning,
    providing site-specific data loaders and global statistics.

    The 7-site architecture:
    - Training sites: COBRE (0), FBIRN (1), MCIC (2), LA5c (3), SRPBS (4)
    - Validation sites: BSNIP-2 (5), TCP 2025 (6)

    Args:
        sites: Dictionary mapping site_id to SiteFCDataset.
        validation_site_ids: Set of site IDs used for validation.
    """

    def __init__(
        self,
        sites: Dict[int, SiteFCDataset],
        validation_site_ids: Optional[set] = None,
    ):
        self.sites = sites
        self.validation_site_ids = validation_site_ids or set()

        self.training_site_ids = set(sites.keys()) - self.validation_site_ids

    @property
    def n_sites(self) -> int:
        return len(self.sites)

    @property
    def n_training_sites(self) -> int:
        return len(self.training_site_ids)

    @property
    def n_validation_sites(self) -> int:
        return len(self.validation_site_ids)

    def get_training_sites(self) -> Dict[int, SiteFCDataset]:
        """Get training sites only."""
        return {k: v for k, v in self.sites.items() if k in self.training_site_ids}

    def get_validation_sites(self) -> Dict[int, SiteFCDataset]:
        """Get validation sites only."""
        return {k: v for k, v in self.sites.items() if k in self.validation_site_ids}

    def get_total_samples(self) -> Dict[str, int]:
        """Get total sample counts across all sites."""
        total_sz = sum(s.dataset.n_sz for s in self.sites.values())
        total_hc = sum(s.dataset.n_hc for s in self.sites.values())
        return {
            "total_sz": total_sz,
            "total_hc": total_hc,
            "total": total_sz + total_hc,
            "n_sites": self.n_sites,
        }

    def get_site_weights(self) -> Dict[int, float]:
        """Compute per-site weights for federated aggregation.

        Weights are proportional to sample size (weighted FedAvg).

        Returns:
            Dictionary mapping site_id to weight.
        """
        sample_counts = {
            sid: len(site.dataset) for sid, site in self.sites.items()
        }
        total = sum(sample_counts.values())
        return {sid: count / total for sid, count in sample_counts.items()}

    def get_summary(self) -> Dict:
        """Get comprehensive summary across all sites."""
        return {
            "n_sites": self.n_sites,
            "n_training_sites": self.n_training_sites,
            "n_validation_sites": self.n_validation_sites,
            "total_samples": self.get_total_samples(),
            "site_summaries": {
                sid: site.dataset.get_summary()
                for sid, site in self.sites.items()
            },
        }
