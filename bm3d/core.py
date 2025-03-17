"""
Implementation of BM3D method for removing noise from images.
The algorithm consists of two stages: basic and final.
"""

from typing import List
import numpy as np

from .profile import BM3DProfile, BM3DStages
from .blockmatching import findSimilarGroups, getGroupsFromCoords
from .filtration import applyFilterHt, applyFilterWie
from .agregation import agregation


def bm3d(noisyImage: np.ndarray, noiseVariance: float,
         profile: BM3DProfile) -> np.ndarray:
    """
    Apply BM3D method to denoise image
    """

    estimate: np.ndarray = bm3dBasic(noisyImage, noiseVariance, profile)

    if profile.stages == BM3DStages.BOTH_STAGES:
        estimate = bm3dFinal(noisyImage, estimate,
                             noiseVariance, profile)

    return estimate


def bm3dBasic(noisyImage: np.ndarray, noiseVariance: float,
              profile: BM3DProfile) -> np.ndarray:
    """
    Perform basic step of the BM3D with hard-threshold filter
    """

    groupsCoords, groups = findSimilarGroups(noisyImage, profile)

    filteredGroups, weights = applyFilterHt(groups, noiseVariance, profile)

    imageEstimate = agregation(noisyImage.shape, filteredGroups,
                               groupsCoords, weights, profile)
    clippedEstimate = np.clip(imageEstimate, 0, 255).astype(np.uint8)

    return clippedEstimate


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

    clippedEstimate = np.clip(imageEstimate, 0, 255).astype(np.uint8)

    return clippedEstimate


