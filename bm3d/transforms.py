"""
Functions for applying transformations used in BM3D
"""

import scipy
from scipy.fft import dctn, idctn
from typing import List, Tuple
import numpy as np
import pywt
# from numba import njit, prange

# @njit(nogil=True, fastmath=True, cache=True)
# def dctnJIT(x, norm=None):
#     return scipy.fft.dctn(x, norm=norm)
#
# @njit(nogil=True, fastmath=True, cache=True)
# def applyToBlocks2dDctJIT(blocks: np.ndarray) -> np.ndarray:
#     """
#     Apply 2D DCT to each block in blocks array
#
#     Args:
#         blocks: array of blocks
#     
#     Return:
#         Array of transformed blocks
#     """
#     transformedBlocks = np.empty_like(blocks, dtype=np.float32)
#
#     for i in prange(blocks.shape[0]): 
#         for j in range(blocks.shape[1]):
#             block = blocks[i, j]
#             dctBlock = dctnJIT(block, norm='ortho')
#             transformedBlocks[i, j] = dctBlock
#
#     return transformedBlocks

def applyToBlocks2dDct(blocks: np.ndarray) -> np.ndarray:
    """
    Apply 2D DCT to each block in blocks array

    Args:
        blocks: array of blocks
    
    Return:
        Array of transformed blocks
    """
    transformedBlocks = np.empty_like(blocks, dtype=np.float32)

    for i in range(blocks.shape[0]): 
        for j in range(blocks.shape[1]):
            block = blocks[i, j]
            dctBlock = dctn(block, norm='ortho')
            transformedBlocks[i, j] = dctBlock

    return transformedBlocks


def applyToGroupInverse2DCT(group: np.ndarray) -> np.ndarray:
    """
    Apply inverse 2D DCT to each block in list of groups

    Args:
        groups: list of groups
    
    Return:
        List of inverse-transformed groups
    """

    return np.stack([idctn(block, norm='ortho') for block in group])

def applyHaar(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply 1D Haar Wavelet transform
    
    Args:
        signal: array representing input signal
    
    Return:
        Tuple of approximation and detail coefficients
    """

    return pywt.dwt(signal, 'haar')


def applyInverseHaar(approximationCoeffs: np.ndarray,
                     detailingCoeffs: np.ndarray) -> np.ndarray:
    """
    Apply inverse 1D Haar Wavelet transform
    
    Args:
        signal: tuple of approximation and detail coefficients 
    
    Return:
        Array representing reconstructed signal
    """
    return pywt.idwt(approximationCoeffs, detailingCoeffs, 'haar')


def applyToGroups1DInverseTransform(group: np.ndarray,
                                    oldGroup: np.ndarray) -> np.ndarray:
    """
    Apply inverse 1D Transform to each group in list of groups

    Args:
        groups: list of groups with wavelet coefficients
    
    Return:
        List of inverse-transformed groups
    """

    approximationCoeffs: np.ndarray = group[:, 0, :]
    detailingCoeffs: np.ndarray = group[:, 1, :]    

    restoredSignals: np.ndarray = applyInverseHaar(approximationCoeffs, detailingCoeffs)

    # Reshape coefficients to match original block shape
    reshapedCoeffs = restoredSignals.T.flatten().reshape(oldGroup.shape)

    return reshapedCoeffs


def applyToGroup1dTransform(group: np.ndarray) -> np.ndarray:
    """
    Apply 1D Transform to each group in list of groups

    Args:
        groups: list of groups
    
    Return:
        List of transformed groups
    """

    # Reshape group to get vertically overlapping pixels become rows
    groupedCoeffs: np.ndarray = np.transpose(group, (1, 2, 0)).reshape(-1, group.shape[0])

    # Apply Haar Transform to each vertically overlapping pixels
    transformedGroup: np.ndarray = np.apply_along_axis(applyHaar, axis=1, arr=groupedCoeffs)


    return transformedGroup
