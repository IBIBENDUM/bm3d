"""
Functions for applying filtering using transform domain 
"""

import numpy as np
from typing import List, Tuple
from .profile import BM3DProfile
from .transforms import *


def applyHTtoGroups(groups: List[np.ndarray], profile: BM3DProfile) -> List[np.ndarray]:
    """
    Apply hard-threshold filter to list of groups
    """

    filteredCoeffs: List[np.ndarray] = []
    for group in groups:
        filteredCoeffs.append(applyFilterToHaar(group, profile))

    return filteredCoeffs


def applyFilterToHaar(group: np.ndarray, profile: BM3DProfile) -> np.ndarray:
    """
    Apply filter to Haar coefficients
    """

    approximationCoeffs: np.ndarray = group[:, 0, :]
    detailingCoeffs: np.ndarray = group[:, 1, :]    

    filteredDetailingCoeffs: np.ndarray = applyHTtoSignal(detailingCoeffs, profile)

    filtered_group: np.ndarray = np.empty_like(group)
    filtered_group[:, 0, :] = approximationCoeffs 
    filtered_group[:, 1, :] = filteredDetailingCoeffs 

    return filtered_group


def applyHTtoSignal(array: np.ndarray, profile: BM3DProfile) -> np.ndarray:
    """
    Apply hard-threshold filter to array
    """

    return np.where(abs(array) < profile.filterThreshold, 0, array)

    
def calculateBlocksWeights(groups: List[np.ndarray], noiseVariance: float) -> List[float]:
    """
    Calculate weights for each group based on number of non-zero coefficients
    in transform domain for reference block
    """

    groupsWeights: List[float] = []
    for group in groups:
        detailingCoeffs: np.ndarray = group[:, 1, 0]
        nonZeroCoeffs: int = np.count_nonzero(detailingCoeffs)
        groupWeight: float = 0.0
        if nonZeroCoeffs != 0:
            groupWeight = 1.0 / (nonZeroCoeffs * noiseVariance ** 2)
        groupsWeights.append(groupWeight)

    return groupsWeights


def applyFilterInTransformDomain(groups: List, noiseVariance: float,
                                 profile: BM3DProfile) -> Tuple[List, List]:
    """
    Apply 3D Filtering to groups using transform domain

    Args:
        groups: List of groups
        noiseVariance: Variance of the noise used for weight
        profile: BM3D properties

    Return:
        List of filtered groups 
        List of weight for each group
    """

    transformedGroups2D = applyToGroups2DCT(groups)
    transformedCoeffs1D = applyToGroups1DTransform(transformedGroups2D)

    filteredCoeffs1D = applyHTtoGroups(transformedCoeffs1D, profile)
    weights = calculateBlocksWeights(filteredCoeffs1D, noiseVariance)

    filteredCoeffs2D = applyToGroups1DInverseTransform(filteredCoeffs1D, groups)
    filteredGroups = applyToGroupsInverse2DCT(filteredCoeffs2D)

    return filteredGroups, weights
