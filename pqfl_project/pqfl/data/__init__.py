"""Data processing module for fMRI functional connectivity.

Handles fMRI preprocessing, functional connectivity matrix construction,
PyTorch dataset creation, and multi-site data partitioning.
"""

from .fmri_pipeline import FMRIPipeline, FMRIConfig
from .fc_construction import FCConstructor, compute_fc_matrix
from .dataset import FCDataset, SiteFCDataset, MultiSiteDataset
from .site_partitioning import SitePartitioner

__all__ = [
    "FMRIPipeline",
    "FMRIConfig",
    "FCConstructor",
    "compute_fc_matrix",
    "FCDataset",
    "SiteFCDataset",
    "MultiSiteDataset",
    "SitePartitioner",
]
