"""
Functions for applying filtering using transform domain 
"""

import numpy as np
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
from .profile import BM3DProfile
from .blockmatching import getGroupsFromCoords
from .transforms import *


# def applyHtToGroups(groups: List[np.ndarray], noiseVariance: float, profile: BM3DProfile) -> List[np.ndarray]:
#     """
#     Apply hard-threshold filter to list of groups
#     """
#
#     filteredCoeffs: List[np.ndarray] = []
#     for group in groups:
#         filteredCoeffs.append(applyHtToGroup(group, noiseVariance, profile))
#
#     return filteredCoeffs

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

def applyHtToGroup(group: np.ndarray, noiseVariance: float,
                      profile: BM3DProfile) -> tuple[np.ndarray, float]:
    """
    Apply filter to Haar coefficients
    """

    approximationCoeffs: np.ndarray = group[:, 0, :]
    detailingCoeffs: np.ndarray = group[:, 1, :]    

    weight = calculateGroupWeight(detailingCoeffs, noiseVariance)
    filteredDetailingCoeffs: np.ndarray = applyHTtoSignal(detailingCoeffs, 
                                                          noiseVariance, profile)

    filtered_group: np.ndarray = np.empty_like(group)
    filtered_group[:, 0, :] = approximationCoeffs 
    filtered_group[:, 1, :] = filteredDetailingCoeffs 

    return filtered_group, weight


def applyHTtoSignal(array: np.ndarray, noiseVariance: float,
                    profile: BM3DProfile) -> np.ndarray:
    """
    Apply hard-threshold filter to array
    """

    return np.where(abs(array) < noiseVariance * profile.filterThreshold, 0, array)

    
def calculateGroupWeight(detailingCoeffs: np.ndarray,
                           noiseVariance: float) -> float:
    """
    Calculate weights for each group based on number of non-zero coefficients
    in transform domain for reference block
    """

    nonZeroCoeffs: int = np.count_nonzero(detailingCoeffs)
    groupWeight: float = 0.0
    if nonZeroCoeffs != 0:
        groupWeight = 1.0 / (nonZeroCoeffs * noiseVariance ** 2)

    return groupWeight

def filterGroupHt(transformedBlocks: np.ndarray, group: np.ndarray, groupsCoord: np.ndarray,
                  noiseVariance: float, profile: BM3DProfile) -> Tuple[np.ndarray, float]:
    indices = groupsCoord // profile.blockStep
    transformedGroup2D = transformedBlocks[indices[:, 0], indices[:, 1]]

    transformedGroup1D = applyToGroup1dTransform(transformedGroup2D)

    filteredGroup1D, weight = applyHtToGroup(transformedGroup1D, noiseVariance, profile)

    filteredGroup2D = applyToGroups1DInverseTransform(filteredGroup1D, group)

    filteredGroup = applyToGroupInverse2DCT(filteredGroup2D)

    return filteredGroup, weight


def applyFilterHt(blocks: np.ndarray, groups: List[np.ndarray], groupsCoords: List[np.ndarray],
                  noiseVariance: float, profile: BM3DProfile) -> Tuple[List, List]:
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
    
    transformedBlocks = applyToBlocks2dDct(blocks)

    args = [
        (transformedBlocks, group, groupCoords, noiseVariance, profile)
        for group, groupCoords in zip(groups, groupsCoords)
    ]

    with Pool(processes=profile.cores) as p:
        results = p.starmap(filterGroupHt, args)

    filteredGroups = []
    weights = []

    for group, weight in results:
        filteredGroups.append(group)
        weights.append(weight)

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

