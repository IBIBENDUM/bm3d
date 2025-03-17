import numpy as np
from typing import List, Tuple
from .profile import BM3DProfile
from .transforms import *

def applyHTtoGroups(groups: List, profile: BM3DProfile) -> List:
    filteredCoeffs = []
    for group in groups:
        filteredCoeffs.append(applyFilterToHaar(group, profile))

    return filteredCoeffs

def applyHTtoSignal(signal: np.ndarray, profile: BM3DProfile) -> np.ndarray:
    return np.where(abs(signal) < profile.filterThreshold, 0, signal)

def applyFilterToHaar(group: np.ndarray, profile: BM3DProfile) -> np.ndarray:

    approximationCoeffs = group[:, 0, :]
    detailingCoeffs = group[:, 1, :]    

    filteredDetailingCoeffs = applyHTtoSignal(detailingCoeffs, profile)

    filtered_group = np.empty_like(group)
    filtered_group[:, 0, :] = approximationCoeffs 
    filtered_group[:, 1, :] = filteredDetailingCoeffs 

    return filtered_group
    
def calculateBlocksWeights(groups: List, noiseVariance: float) -> List:
    groupsWeights = []
    for group in groups:
        detailingCoeffs = group[:, 1, 0]
        nonZeroCoeffs = np.count_nonzero(detailingCoeffs)
        # TODO: Add check for zero division
        groupWeight = 0.0
        if nonZeroCoeffs != 0:
            groupWeight = 1.0 / (nonZeroCoeffs * noiseVariance ** 2)
        groupsWeights.append(groupWeight)

    return groupsWeights

def applyFilterInTransformDomain(groups: List, noiseVariance: float,
                                 profile: BM3DProfile) -> Tuple[List, List]:
    transformedGroups2D = applyToGroups2DCT(groups)
    transformedCoeffs1D = applyToGroups1DTransform(transformedGroups2D)

    filteredCoeffs1D = applyHTtoGroups(transformedCoeffs1D, profile)
    weights = calculateBlocksWeights(filteredCoeffs1D, noiseVariance)

    filteredCoeffs2D = applyToGroups1DInverseTransform(filteredCoeffs1D, groups)
    filteredGroups = applyToGroupsInverse2DCT(filteredCoeffs2D)

    return filteredGroups, weights










