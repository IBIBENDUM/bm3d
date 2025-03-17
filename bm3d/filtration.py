import numpy as np
from typing import List
from .profile import BM3DProfile

def applyHTtoGroups(groups: List, profile: BM3DProfile) -> List:
    filteredCoeffs = []
    for group in groups:
        filteredCoeffs.append(applyFilterToHaar(group, profile))

    return filteredCoeffs

def applyHTtoSignal(signal: np.ndarray, profile: BM3DProfile) -> np.ndarray:
    return np.where(abs(signal) < profile.filterThreshold, signal, 0)

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
        # detailingCoeffs = group[:, 1, :].reshape(16, 16, 8)
        # nonZeroCoeffs = np.count_nonzero(detailingCoeffs, axis=(1, 2))
        detailingCoeffs = group[:, 1, :]
        nonZeroCoeffs = np.count_nonzero(detailingCoeffs)
        # TODO: Add check for zero division
        groupWeights = 1.0 / (nonZeroCoeffs * noiseVariance ** 2)
        groupsWeights.append(groupWeights)

    return groupsWeights












