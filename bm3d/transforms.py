"""
Functions for applying transformations used in BM3D
"""

from scipy.fftpack import dctn, idctn
from typing import List, Tuple
import numpy as np
import pywt


def applyToGroups2DCT(groups: List[np.ndarray]) -> List[np.ndarray]:
    """
    Apply 2D DCT to each block in list of groups

    Args:
        groups: list of groups
    
    Return:
        List of transformed groups
    """

    transformedGroups: List[np.ndarray] = []
    for group in groups:
        transformedGroup: np.ndarray = np.empty_like(group, dtype=np.float64)
        for i, block in enumerate(group):
            transformedGroup[i] = dctn(block, norm='ortho')
        transformedGroups.append(transformedGroup)

    return transformedGroups


def applyToGroupsInverse2DCT(groups: List[np.ndarray]) -> List[np.ndarray]:
    """
    Apply inverse 2D DCT to each block in list of groups

    Args:
        groups: list of groups
    
    Return:
        List of inverse-transformed groups
    """

    transformedGroups: List[np.ndarray] = []
    for group in groups:
        transformedGroup: np.ndarray = np.empty_like(group, dtype=np.float64)
        for i, block in enumerate(group):
            transformedGroup[i] = idctn(block, norm='ortho')
        transformedGroups.append(transformedGroup)

    return transformedGroups


def applyHaar(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply 1D Haar Wavelet transform
    
    Args:
        signal: array representing input signal
    
    Return:
        Tuple of approximation and detail coefficients
    """

    return pywt.dwt(signal, 'haar')


# TODO: Use this function for vectorizing
def applyInverseHaar(signal: np.ndarray) -> np.ndarray:
    """
    Apply inverse 1D Haar Wavelet transform
    
    Args:
        signal: tuple of approximation and detail coefficients 
    
    Return:
        Array representing reconstructed signal
    """
    return pywt.idwt(signal[0], signal[1], 'haar')


def applyToGroups1DInverseTransform(groups: List[np.ndarray],
                                    oldGroups: List[np.ndarray]) -> List[np.ndarray]:
    """
    Apply inverse 1D Transform to each group in list of groups

    Args:
        groups: list of groups with wavelet coefficients
    
    Return:
        List of inverse-transformed groups
    """

    inversedGroups: List[np.ndarray] = []
    for i, group in enumerate(groups):
        cA: np.ndarray = group[:, 0, :]
        cD: np.ndarray = group[:, 1, :]

        restoredSignals: np.ndarray = pywt.idwt(cA, cD, 'haar')

        # Reshape coefficients to match original block shape
        reshapedCoeffs = restoredSignals.T.flatten().reshape(oldGroups[i].shape)

        inversedGroups.append(reshapedCoeffs)

    return inversedGroups


def applyToGroups1DTransform(groups: List[np.ndarray]) -> List[np.ndarray]:
    """
    Apply 1D Transform to each group in list of groups

    Args:
        groups: list of groups
    
    Return:
        List of transformed groups
    """

    transformedGroups: List[np.ndarray] = []
    for group in groups:
        # Reshape group to get vertically overlapping pixels become rows
        groupedCoeffs: np.ndarray = np.transpose(group, (1, 2, 0)).reshape(-1, group.shape[0])

        # Apply Haar Transform to each vertically overlapping pixels
        transformedGroup: np.ndarray = np.apply_along_axis(applyHaar, axis=1, arr=groupedCoeffs)

        transformedGroups.append(transformedGroup)

    return transformedGroups
