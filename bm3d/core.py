"""
Implementation of BM3D method for removing noise from images.
The algorithm consists of two stages: basic and final.
"""

from typing import List
import numpy as np
import os

from .profile import BM3DProfile, BM3DStages
from .blockmatching import findSimilarGroups, getGroupsFromCoords
from .filtration import applyFilterHt, applyFilterWie
from .agregation import globalAgregation


def unlockAllCores():
    coresNumber = os.cpu_count() or 1
    os.sched_getaffinity(coresNumber)


def bm3d(noisyImage: np.ndarray, noiseVariance: float,
         profile: BM3DProfile) -> np.ndarray:
    """
    Apply BM3D method to denoise image
    """

    # Reset task affinity so that all cores are used
    # unlockAllCores()

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

    blocks: np.ndarray = np.lib.stride_tricks.sliding_window_view(
        noisyImage, (profile.blockSize, profile.blockSize)
        )[:: profile.blockStep, :: profile.blockStep]

    groupsCoords, groups = findSimilarGroups(blocks, profile)

    filteredGroups, weights = applyFilterHt(blocks, groups, groupsCoords, noiseVariance, profile)

    imageEstimate = globalAgregation(noisyImage.shape, filteredGroups,
                               groupsCoords, weights, profile)

    clippedEstimate = np.clip(imageEstimate, 0, 255).astype(np.uint8)
                  
    return clippedEstimate


def bm3dFinal(basicEstimate: np.ndarray, noisyImage: np.ndarray,
              noiseVariance: float, profile: BM3DProfile) -> np.ndarray:
    """
    Perform final step of the BM3D with wiener filter
    """

    blocks: np.ndarray = np.lib.stride_tricks.sliding_window_view(
        noisyImage, (profile.blockSize, profile.blockSize)
    )[:: profile.blockStep, :: profile.blockStep]

    groupsCoords, groupsEstimate = findSimilarGroups(basicEstimate, profile)
    groupsImage: List[np.ndarray] = getGroupsFromCoords(noisyImage, groupsCoords,
                                                        profile)

    filteredGroups, weights = applyFilterWie(groupsEstimate, groupsImage,
                                             noiseVariance)

    imageEstimate = globalAgregation(noisyImage.shape, filteredGroups,
                               groupsCoords, weights, profile)

    clippedEstimate = np.clip(imageEstimate, 0, 255).astype(np.uint8)

    return clippedEstimate


