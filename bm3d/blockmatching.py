"""
Functions for finding groups of similar blocks in image
"""

import os
import time
import numpy as np
from typing import Tuple, List
from .profile import BM3DProfile
from multiprocessing import Pool, cpu_count


def findSimilarBlocksIndices(refBlock: np.ndarray, matchingBlocks: np.ndarray,
                             profile: BM3DProfile) -> np.ndarray:
    """
    Find blocks in matchingBlocks similar to refBlock

    Args: 
        refBlock: Reference block
        matchingBlocks: Array of all possible blocks to search in
        profile: BM3D parameters

    Return:
        Indices of similar blocks sorted by increasing distance 
    """
    # Calculate distance between reference blocks and all other
    distance: np.ndarray = np.sum((matchingBlocks - refBlock) ** 2,
                                  axis=(2, 3), dtype=np.int64)
 
    # Get indices where distance below threshold
    indices: np.ndarray = np.argwhere(distance < profile.distanceThreshold *
                                      profile.blockSize ** 2)

    # Get distance values for sorting
    diffValues: np.ndarray = distance[indices[:, 0], indices[:, 1]]

    # Sort indices by distance increasing
    sortedIndices: np.ndarray = np.argsort(diffValues)

    return indices[sortedIndices]

def getSearchWindow(xRef: int, yRef: int, blocks: np.ndarray,
                    profile: BM3DProfile) -> tuple[np.ndarray, np.ndarray]:
    """
    Get window to search for similar blocks

    Args:
        xRef: Reference block x coordinate
        yRef: Reference block y coordinate
        blocks: Array of all possible blocks
        profile: BM3D properties
    
    Return:
        View of blocks limited by search window 
        Array of matching window coordinates
    """

    if profile.searchWindow == 0:
        return blocks, np.array([0, 0])

    startY = max(0, yRef - profile.searchWindow)
    endY = min(blocks.shape[0], yRef + profile.searchWindow + 1)
    startX = max(0, xRef - profile.searchWindow)
    endX = min(blocks.shape[1], xRef + profile.searchWindow + 1)

    return blocks[startY:endY, startX:endX], np.array([startY, startX])


def processBlock(args: Tuple[int, int, np.ndarray, BM3DProfile]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single block to find similar blocks.

    Args:
        args: Tuple containing (y, x, blocks, profile)

    Return:
        Tuple of (similarBlocksCoords, group)
    """
    y, x, blocks, profile = args
    refBlock = blocks[y, x]
    searchWindow, searchWindowCoords = getSearchWindow(x, y, blocks, profile)
    indices = findSimilarBlocksIndices(refBlock, searchWindow, profile)

    # Ensure even number of blocks
    if indices.shape[0] % 2 != 0:
        indices = indices[:-1]

    # Limit group size if specified
    if profile.groupMaxSize != 0:
        if indices.shape[0] > profile.groupMaxSize:
            indices = indices[:profile.groupMaxSize]

    # Scale indices to get coordinates
    similarBlocksCoords = (indices + searchWindowCoords) * profile.blockStep

    # Extract group of similar blocks
    group = searchWindow[indices[:, 0], indices[:, 1]]

    return similarBlocksCoords, group


def findSimilarGroups(blocks: np.ndarray, profile: BM3DProfile) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Find groups of similar block in image

    Args:
        image: Input image
        profile: BM3D properties
    
    Return:
        List of coordinate arrays for each group
        List of block arrays for each group
    """


    # TODO: Add check for last block
    # Get all possible blocks
    # blocks: np.ndarray = np.lib.stride_tricks.sliding_window_view(
    #     image, (profile.blockSize, profile.blockSize)
    # )[:: profile.blockStep, :: profile.blockStep]

    args = [
        (y, x, blocks, profile)
        for y in range(blocks.shape[0])
        for x in range(blocks.shape[1])
    ]

    with Pool(processes=cpu_count()) as p:
        results = p.map(processBlock, args)

    similarBlocksCoords: List[np.ndarray] = []
    similarGroups: List[np.ndarray] = []

    for coords, group in results:
        similarBlocksCoords.append(coords)
        similarGroups.append(group)

    return similarBlocksCoords, similarGroups


def getGroupsFromCoords(blocks: np.ndarray, groupsCoords: List[np.ndarray],
                        profile: BM3DProfile) -> List[np.ndarray]:

    groups = []
    for coords in groupsCoords:
        indices = coords // profile.blockStep
        group = blocks[indices[:, 0], indices[:, 1]]
        groups.append(group)

    return groups

