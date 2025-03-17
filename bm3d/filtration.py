"""
Functions for applying filtering using transform domain 
"""

import numpy as np
from typing import List, Tuple
from .profile import BM3DProfile
from .transforms import *


def applyHtToGroups(groups: List[np.ndarray], noiseVariance: float, profile: BM3DProfile) -> List[np.ndarray]:
    """
    Apply hard-threshold filter to list of groups
    """

    filteredCoeffs: List[np.ndarray] = []
    for group in groups:
        filteredCoeffs.append(applyFilterToHaar(group, noiseVariance, profile))

    return filteredCoeffs

def applyWieToGroups(groupsEstimate: List[np.ndarray], groupsImage: List[np.ndarray],
                     noiseVariance: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Apply hard-threshold filter to list of groups
    """
    filteredCoeffs: List[np.ndarray] = []
    weights: List = []
    for groupEstimate, groupImage in zip(groupsEstimate, groupsImage):
        estimateEnergy = np.power(groupEstimate, 2)
        wienerCoeffs = estimateEnergy / (estimateEnergy + noiseVariance ** 2)
        filteredCoeffs.append(wienerCoeffs * groupImage)
        weights.append(np.sum(wienerCoeffs))

    return filteredCoeffs, weights

def applyFilterToHaar(group: np.ndarray, noiseVariance: float,
                      profile: BM3DProfile) -> np.ndarray:
    """
    Apply filter to Haar coefficients
    """

    approximationCoeffs: np.ndarray = group[:, 0, :]
    detailingCoeffs: np.ndarray = group[:, 1, :]    

    filteredDetailingCoeffs: np.ndarray = applyHTtoSignal(detailingCoeffs, 
                                                          noiseVariance, profile)

    filtered_group: np.ndarray = np.empty_like(group)
    filtered_group[:, 0, :] = approximationCoeffs 
    filtered_group[:, 1, :] = filteredDetailingCoeffs 

    return filtered_group


def applyHTtoSignal(array: np.ndarray, noiseVariance: float,
                    profile: BM3DProfile) -> np.ndarray:
    """
    Apply hard-threshold filter to array
    """

    return np.where(abs(array) < noiseVariance * profile.filterThreshold, 0, array)

    
def calculateBlocksWeights(groups: List[np.ndarray],
                           noiseVariance: float) -> List[float]:
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


def applyFilterHt(groups: List[np.ndarray], noiseVariance: float,
                  profile: BM3DProfile) -> Tuple[List, List]:
    """
    Apply Hard-threshold filter to groups using transform domain

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

    filteredCoeffs1D = applyHtToGroups(transformedCoeffs1D, noiseVariance, profile)
    weights = calculateBlocksWeights(filteredCoeffs1D, noiseVariance)

    filteredCoeffs2D = applyToGroups1DInverseTransform(filteredCoeffs1D, groups)
    filteredGroups = applyToGroupsInverse2DCT(filteredCoeffs2D)

    return filteredGroups, weights


def applyFilterWie(groupsEstimate: List[np.ndarray],
                   groupsImage: List[np.ndarray],
                   noiseVariance: float) -> Tuple[List, List]:
    """
    Apply Wiener filter to groups using transform domain

    Args:
        groupsEstimate: List of groups for basic estimate
        groupsImage: List of groups for noisy image
        noiseVariance: Variance of the noise used for weight
        profile: BM3D properties

    Return:
        List of filtered groups 
        List of weight for each group
    """

    transformedCoeffs: List[List[np.ndarray]] = []
    for groups in [groupsEstimate, groupsImage]:
        transformedGroups2D = applyToGroups2DCT(groups)
        transformedCoeffs1D = applyToGroups1DTransform(transformedGroups2D)
        transformedCoeffs.append(transformedCoeffs1D)

    filteredCoeffs1D, weights = applyWieToGroups(transformedCoeffs[0], transformedCoeffs[1], noiseVariance)

    filteredCoeffs2D = applyToGroups1DInverseTransform(filteredCoeffs1D, groupsEstimate)
    filteredGroups = applyToGroupsInverse2DCT(filteredCoeffs2D)

    return filteredGroups, weights

