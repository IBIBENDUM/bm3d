"""
Functions for applying filtering using transform domain 
"""

import numpy as np
from typing import List, Tuple
from multiprocessing import Pool
from .profile import BM3DProfile
from .transforms import *

def applyHtToGroup(group: np.ndarray, noiseVariance: float,
                   profile: BM3DProfile) -> tuple[np.ndarray, float]:
    """
    Apply hard-threshold filter to Haar coefficients in group
    """

    filteredGroup = group.copy()
    detailingCoeffs: np.ndarray = filteredGroup[:, 1, :]    

    detailingCoeffs[:] = applyHTtoSignal(detailingCoeffs, noiseVariance, profile.filterThreshold)
    weight = calculateGroupWeightHt(detailingCoeffs, noiseVariance)

    return filteredGroup, weight


def applyHTtoSignal(array: np.ndarray, noiseVariance: float,
                    filterThreshold: float) -> np.ndarray:
    """
    Apply hard-threshold filter to array
    """

    return np.where(abs(array) < noiseVariance * filterThreshold, 0, array)


def calculateGroupWeightHt(detailingCoeffs: np.ndarray, noiseVariance: float) -> float:
    """
    Calculate weights for each group based on number of non-zero coefficients
    in transform domain for reference block
    """

    nonZeroCoeffs: int = np.count_nonzero(detailingCoeffs)
    groupWeight: float = 0.0
    if nonZeroCoeffs != 0:
        groupWeight = 1.0 / (nonZeroCoeffs * noiseVariance ** 2)

    return groupWeight

def filterGroupHt(transformedBlocks: np.ndarray, group: np.ndarray, groupCoord: np.ndarray,
                  noiseVariance: float, profile: BM3DProfile) -> Tuple[np.ndarray, float]:
    """
    Filter group with hard-threshold filter
    """
    indices = groupCoord // profile.blockStep
    transformedGroup2D = transformedBlocks[indices[:, 0], indices[:, 1]]

    transformedGroup1D = applyToGroup1dTransform(transformedGroup2D)

    filteredGroup1D, weight = applyHtToGroup(transformedGroup1D, noiseVariance, profile)

    filteredGroup2D = applyToGroups1DInverseTransform(filteredGroup1D, group)

    filteredGroup = applyToGroupInverse2DCT(filteredGroup2D)

    return filteredGroup, weight


def applyFilterHt(blocks: np.ndarray, groups: List[np.ndarray], groupsCoords: List[np.ndarray],
                  noiseVariance: float, profile: BM3DProfile) -> Tuple[List, List]:
    """
    Apply Hard-threshold filtering to groups using transform domain
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


def applyWieToGroup(
    estimateGroup: np.ndarray, noisyGroup: np.ndarray, noiseVariance: float
) -> Tuple[np.ndarray, float]:
    """
    Apply Wiener filtering to group
    """
    estimateEnergy = np.power(estimateGroup, 2)
    wienerCoeffs = estimateEnergy / (estimateEnergy + noiseVariance**2)
    filteredCoeff = wienerCoeffs * noisyGroup
    weight = 1.0 / (np.sum(wienerCoeffs**2) + noiseVariance**2)

    return filteredCoeff, weight


def filterGroupWie(
    estimateTransformedBlocks: np.ndarray,
    noisyTransformedBlocks: np.ndarray,
    groupEstimate: np.ndarray,
    groupCoord: np.ndarray,
    noiseVariance: float,
    profile: BM3DProfile,
) -> Tuple[np.ndarray, float]:
    """
    Apply Wiener filtering to a group in the transform domain
    """

    indices = groupCoord // profile.blockStep
    estimateTransformedGroup2D = estimateTransformedBlocks[indices[:, 0], indices[:, 1]]
    noisyTransformedGroup2D = noisyTransformedBlocks[indices[:, 0], indices[:, 1]]

    estimateTransformedGroup1D = applyToGroup1dTransform(estimateTransformedGroup2D)
    noisyTransformedGroup1D = applyToGroup1dTransform(noisyTransformedGroup2D)

    filteredGroup1D, weight = applyWieToGroup(
        estimateTransformedGroup1D, noisyTransformedGroup1D, noiseVariance
    )

    filteredGroup2D = applyToGroups1DInverseTransform(filteredGroup1D, groupEstimate)

    filteredGroup = applyToGroupInverse2DCT(filteredGroup2D)

    return filteredGroup, weight


def applyFilterWie(
    estimateBlocks: np.ndarray,
    noisyBlocks: np.ndarray,
    estimateGroups: List[np.ndarray],
    groupsCoords: List[np.ndarray],
    noiseVariance: float,
    profile: BM3DProfile,
) -> Tuple[List, List]:
    """
    Apply Wiener filter to groups using transform domain
    """

    estimateTransformedBlocks = applyToBlocks2dDct(estimateBlocks)
    noisyTransformedBlocks = applyToBlocks2dDct(noisyBlocks)

    filteredGroups = []
    weights = []

    args = [
        (
            estimateTransformedBlocks,
            noisyTransformedBlocks,
            groupEstimate,
            groupCoords,
            noiseVariance,
            profile,
        )
        for groupEstimate, groupCoords in zip(estimateGroups, groupsCoords)
    ]

    with Pool(processes=profile.cores) as p:
        results = p.starmap(filterGroupWie, args)

    filteredGroups, weights = zip(*results)
    return list(filteredGroups), list(weights)


