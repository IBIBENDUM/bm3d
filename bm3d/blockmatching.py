"""
Functions for finding groups of similar blocks in image
"""

import numpy as np
from typing import Tuple, List
from .profile import BM3DProfile, NumbaBM3DProfile
from multiprocessing import Pool
from numba import njit

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
    indices: np.ndarray = np.argwhere(
        distance < profile.distanceThreshold * profile.blockSize ** 2
    )

    # Get distance values for sorting
    diffValues: np.ndarray = distance[indices[:, 0], indices[:, 1]]

    # Sort indices by distance increasing
    sortedIndices: np.ndarray = np.argsort(diffValues)
    indices = indices[sortedIndices]

    # If only one block is found, add the next closest block
    if indices.shape[0] == 1:
        flat_distance = distance.flatten()
        second_min_idx_flat = np.argpartition(flat_distance, 1)[1]
        second_min_idx = np.array(
            np.unravel_index(
                second_min_idx_flat,
                distance.shape
            )
        ).reshape(1, -1)
        indices = np.concatenate([indices, second_min_idx], axis=0)
    
    return indices

def getSearchWindow(xRef: int, yRef: int, blocks: np.ndarray,
                    profile: BM3DProfile) -> tuple[np.ndarray, np.ndarray]:
    """
    Get blocks in search window

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


def processBlock(
        args: Tuple[int, int, np.ndarray, np.ndarray, BM3DProfile]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single block to find similar blocks.

    Args:
        args: Tuple containing (y, x, blocks, profile)

    Return:
        Tuple of (similarBlocksCoords, group)
    """
    y, x, blocks, blocksCoords, profile = args
    refBlock = blocks[y, x]
    searchWindow, searchWindowCoords = getSearchWindow(x, y, blocks, profile)
    indices = findSimilarBlocksIndices(refBlock, searchWindow, profile)
               
    # Limit group size if specified
    if profile.groupMaxSize != 0:
        if indices.shape[0] > profile.groupMaxSize:
            indices = indices[:profile.groupMaxSize]

    # Ensure even number of blocks
    if indices.shape[0] % 2 != 0:
        indices = indices[:-1]

    # Scale indices to get coordinates
    similarBlocksCoords = blocksCoords[(indices + searchWindowCoords)[:, 0],
                                       (indices + searchWindowCoords)[:, 1]]
    

    # Extract group of similar blocks
    group = searchWindow[indices[:, 0], indices[:, 1]]

    return similarBlocksCoords, group


def findSimilarGroups(
    blocks: np.ndarray,
    blocksCoords: np.ndarray,
    profile: BM3DProfile
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Find groups of similar block in image

    Args:
        image: Input image
        profile: BM3D properties
    
    Return:
        List of coordinate arrays for each group
        List of block arrays for each group
    """

    args = [
        (y, x, blocks, blocksCoords, profile)
        for y in range(blocks.shape[0])
        for x in range(blocks.shape[1])
    ]

    with Pool(processes=profile.cores) as p:
        results = p.map(processBlock, args)

    similarBlocksCoords: List[np.ndarray] = []
    similarGroups: List[np.ndarray] = []

    for coords, group in results:
        similarBlocksCoords.append(coords)
        similarGroups.append(group)

    return similarBlocksCoords, similarGroups


def getGroupsFromCoords(
    blocks: np.ndarray, 
    groupsCoords: List[np.ndarray], 
    profile: BM3DProfile
) -> List[np.ndarray]:
    """
    Retrieves groups of blocks from coordinates of similar blocks

    Args:
        blocks: Array of image blocks
        groupsCoords: List of coordinate arrays
        profile: BM3D properties

    Return:
        List of block arrays for each group
    """

    groups = []
    for coords in groupsCoords:
        i, j  = (coords // profile.blockStep).T
        group = blocks[i, j]
        groups.append(group)

    return groups

from numba import njit, prange
from typing import Tuple

@njit(fastmath=True, cache=True)
def computeIndices(imageLength: int, blockSize: int, blockStep: int) -> np.ndarray:
    """Вычисляет индексы начала блоков вдоль одного измерения изображения"""
    blocksAmount = max(1, (imageLength - blockSize) // blockStep + 1)
    return np.arange(0, imageLength - blockSize + 1, blockStep)

@njit(fastmath=True, cache=True, parallel=True)
def extract_blocks(noisy_image: np.ndarray, block_size: int, y_indices: np.ndarray, x_indices: np.ndarray) -> np.ndarray:
    """Формирует массив блоков изображения"""
    num_y, num_x = len(y_indices), len(x_indices)
    blocks = np.empty((num_y, num_x, block_size, block_size), dtype=noisy_image.dtype)

    for i in prange(num_y):
        for j in prange(num_x):
            blocks[i, j] = noisy_image[y_indices[i]:y_indices[i] + block_size,
                                       x_indices[j]:x_indices[j] + block_size]

    return blocks

@njit(fastmath=True, cache=True, parallel=True)
def getBlocks(noisyImage: np.ndarray, blockSize: int, blockStep: int) -> Tuple[np.ndarray, np.ndarray]:
    """Извлекает блоки из изображения и возвращает их вместе с координатами"""
    imageHeight, imageWidth = noisyImage.shape

    yIndices = computeIndices(imageHeight, blockSize, blockStep)
    xIndices = computeIndices(imageWidth, blockSize, blockStep)

    blocks = extract_blocks(noisyImage, blockSize, yIndices, xIndices)

    # Генерация координат без использования meshgrid
    num_y, num_x = len(yIndices), len(xIndices)
    coords = np.empty((num_y, num_x, 2), dtype=np.int32)

    for i in prange(num_y):
        for j in prange(num_x):
            coords[i, j, 0] = yIndices[i]
            coords[i, j, 1] = xIndices[j]

    return blocks, coords
