"""
Implementation of BM3D method for removing noise from images.
The algorithm consists of two stages: basic and final.
"""

from typing import List
import numpy as np

from .profile import BM3DProfile
from .blockmatching import findSimilarGroups, getGroupsFromCoords
from .filtration import applyFilterHt, applyFilterWie
from .agregation import agregation


def bm3d(noisyImage: np.ndarray, noiseVariance: float,
         profile: BM3DProfile) -> np.ndarray:
    """
    Apply BM3D method to denoise image
    """

    basicEstimate: np.ndarray = bm3dBasic(noisyImage, noiseVariance, profile)
    finalEstimate: np.ndarray = bm3dFinal(noisyImage, basicEstimate,
                                          noiseVariance, profile)

    return finalEstimate


def bm3dBasic(noisyImage: np.ndarray, noiseVariance: float,
              profile: BM3DProfile) -> np.ndarray:
    """
    Perform basic step of the BM3D with hard-threshold filter
    """
    groupsCoords, groups = findSimilarGroups(noisyImage, profile)

    filteredGroups, weights = applyFilterHt(groups, noiseVariance, profile)

    imageEstimate = agregation(noisyImage.shape, filteredGroups,
                               groupsCoords, weights, profile)
    return imageEstimate


def bm3dFinal(basicEstimate: np.ndarray, noisyImage: np.ndarray,
              noiseVariance: float, profile: BM3DProfile) -> np.ndarray:
    """
    Perform final step of the BM3D with wiener filter
    """
    groupsCoords, groupsEstimate = findSimilarGroups(basicEstimate, profile)
    groupsImage: List[np.ndarray] = getGroupsFromCoords(noisyImage, groupsCoords,
                                                        profile)

    filteredGroups, weights = applyFilterWie(groupsEstimate, groupsImage,
                                             noiseVariance)

    imageEstimate = agregation(noisyImage.shape, filteredGroups,
                               groupsCoords, weights, profile)
    return imageEstimate


