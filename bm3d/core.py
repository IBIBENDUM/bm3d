"""
Implementation of BM3D method for removing noise from images.
The algorithm consists of two stages: basic and final.
"""

import numpy as np

from .profile import BM3DProfile
from .blockmatching import findSimilarGroups
from .filtration import applyFilterInTransformDomain
from .agregation import agregationBasic


def bm3d(noisyImage: np.ndarray, noiseVariance: float,
         profile: BM3DProfile) -> np.ndarray:
    """
    Apply BM3D method to denoise image
    """

    basicImage: np.ndarray = _bm3dBasic(noisyImage, noiseVariance, profile)

    return basicImage


def _bm3dBasic(noisyImage: np.ndarray, noiseVariance: float,
               profile: BM3DProfile) -> np.ndarray:
    """
    Perform basic step of the BM3D
    """
    groupsCoords, groups = findSimilarGroups(noisyImage, profile)

    filteredGroups, weights = applyFilterInTransformDomain(groups, noiseVariance, profile)

    imageBasic = agregationBasic(noisyImage.shape, filteredGroups,
                                 groupsCoords, weights, profile)
    return imageBasic

