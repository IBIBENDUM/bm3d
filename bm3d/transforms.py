from scipy.fftpack import dctn, idctn
from typing import List, Tuple
import pywt
import numpy as np

def applyToGroups2DCT(groups: List) -> List:
    transformedGroups = []
    for group in groups:
        transformedGroup = np.empty_like(group, dtype=np.float64)
        for i, block in enumerate(group):
            transformedGroup[i] = dctn(block, norm='ortho')
        transformedGroups.append(transformedGroup)

    return transformedGroups

def applyToGroupsInverse2DCT(groups: List) -> List:
    transformedGroups = []
    for group in groups:
        transformedGroup = np.empty_like(group, dtype=np.float64)
        for i, block in enumerate(group):
            transformedGroup[i] = idctn(block, norm='ortho')
        transformedGroups.append(transformedGroup)
    return transformedGroups

def applyHaar(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return pywt.dwt(signal, 'haar')

def applyInverseHaar(signal: np.ndarray) -> List:
    return pywt.idwt(signal[0], signal[1], 'haar')

def applyToGroups1DInverseTransform(groups: List, oldGroups: List) -> List:
    inversedGroups = []
    for i, group in enumerate(groups):
        cA = group[:, 0, :]
        cD = group[:, 1, :]
        restoredSignals = pywt.idwt(cA, cD, 'haar')
        reshapedCoeffs = restoredSignals.T.flatten().reshape(oldGroups[i].shape)
        inversedGroups.append(reshapedCoeffs)
    return inversedGroups

def applyToGroups1DTransform(groups: List) -> List:
    transformedGroups = []
    for group in groups:
        groupedCoeffs = np.transpose(group, (1, 2, 0)).reshape(-1, group.shape[0])
        transformedGroup = np.apply_along_axis(applyHaar, axis=1, arr=groupedCoeffs)
        transformedGroups.append(transformedGroup)

    return transformedGroups



