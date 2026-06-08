"""Harmonization module for multi-site fMRI data.

Provides ComBat and CovBat harmonization for removing site effects
from functional connectivity features while preserving biological variance.
"""

from .combat import CombatHarmonizer, TangentSpaceCombat

__all__ = [
    "CombatHarmonizer",
    "TangentSpaceCombat",
]
