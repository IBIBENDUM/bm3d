"""
Global estimate by aggregation for image denoising by block-wise estimates
"""

from multiprocessing import Pool, cpu_count, shared_memory
from typing import List, Tuple

import numpy as np

from .profile import BM3DProfile


def getKaiserWindow(size: int, beta: float=2) -> np.ndarray:
    """
    Create 2D Kaiser Window
    Used for aggregation to reduce boundary effects

    Args:
        size: Size of window
        beta: Shape parameter

    Return:
        2D Kaiser window of shape (size, size)
    """

    kaiser = np.kaiser(size, beta)
    kaiser2d = kaiser[:, np.newaxis] @ kaiser[np.newaxis, :]

    return kaiser2d


def agregateGroup(numeratorSharedName, denominatorSharedName,
                  weight: float, group: np.ndarray, groupCoords: np.ndarray,
                  blockSize: int, kaiserWindow: np.ndarray,
                  imageShape,) -> None:
    """
    Aggregates group of blocks using shared memory
    """
    numeratorShared = shared_memory.SharedMemory(name=numeratorSharedName)
    denominatorShared = shared_memory.SharedMemory(name=denominatorSharedName)

    numerator = np.ndarray(imageShape, dtype=np.float64, buffer=numeratorShared.buf)
    denominator = np.ndarray(imageShape, dtype=np.float64, buffer=denominatorShared.buf)

    for block, (y, x) in zip(group, groupCoords):
        numerator[y : y + blockSize, x : x + blockSize] += block * weight * kaiserWindow
        denominator[y : y + blockSize, x : x + blockSize] += weight * kaiserWindow

    # epsilon = 1e-8  # Очень маленькое значение, чтобы избежать деления на ноль
    #
    # safe_denominator = np.where(denominator == 0, epsilon, denominator)  # Заменить нули на epsilon

    numeratorShared.close()
    denominatorShared.close()


def globalAgregation(imageShape: Tuple[int, int], groups: List[np.ndarray],
                     groupsCoords: List[np.ndarray], weights: List[float],
                     profile: BM3DProfile) -> np.ndarray:
    """
    Aggregates blocks using weighted averaging

    Args:
        imageShape: Shape of the output image (height, width)
        groups: List of groups of similar blocks after processing
        groupsCoords: List of coordinates for each block in groups
        weights: Weight coefficients for each group
        profile: BM3D parameters

    Return:
        np.ndarray: Aggregated denoised image
    """
    
    numeratorShared = shared_memory.SharedMemory(
        create=True, size=int(np.prod(imageShape) * 8)
    )
    denominatorShared = shared_memory.SharedMemory(
        create=True, size=int(np.prod(imageShape) * 8)
    )

    numerator = np.ndarray(imageShape, dtype=np.float64, buffer=numeratorShared.buf)
    denominator = np.ndarray(imageShape, dtype=np.float64, buffer=denominatorShared.buf)
    numerator.fill(0)
    denominator.fill(0)

    kaiserWindow: np.ndarray = getKaiserWindow(profile.blockSize, profile.kaiserShape)

    blockSize: int = profile.blockSize

    args = [
        (
            numeratorShared.name,
            denominatorShared.name,
            weight,
            group,
            groupCoords,
            blockSize,
            kaiserWindow,
            imageShape,
        )
        for weight, group, groupCoords in zip(weights, groups, groupsCoords)
    ]

    with Pool(processes=cpu_count()) as p:
        p.starmap(agregateGroup, args)

 
    epsilon = np.finfo(np.float64).eps 
    safe_denominator = np.where(denominator == 0, epsilon, denominator)
    result = numerator / safe_denominator
    # result = numerator / denominator

    numeratorShared.close()
    numeratorShared.unlink()
    denominatorShared.close()
    denominatorShared.unlink()

    return result
