"""
Global estimate by aggregation for image denoising by block-wise estimates
"""

import numpy as np
from typing import Tuple, List
from .profile import BM3DProfile


def agregation(imageShape: Tuple[int, int], groups: List[np.ndarray],
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
    # Accumulates weighted pixel sums
    numerator: np.ndarray = np.zeros(imageShape, dtype=np.float64)
    # Accumulates weights for normalization
    denominator: np.ndarray = np.zeros(imageShape, dtype=np.float64)

    blockSize: int = profile.blockSize

    for weight, group, groupCoords in zip(weights, groups, groupsCoords):
        for block, (y, x) in zip(group, groupCoords):
            numerator[y:y + blockSize, x:x + blockSize] += block * weight
            denominator[y:y + blockSize, x:x + blockSize] += weight

    return numerator / denominator
