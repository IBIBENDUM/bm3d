import numpy as np
from typing import Tuple
from .profile import BM3DProfile

def findSimilarBlocksCoords(refBlock: np.ndarray, matchingBlocks: np.ndarray,
                      profile: BM3DProfile) -> np.ndarray:
    diff = np.sum((matchingBlocks - refBlock) ** 2, axis=(2, 3), dtype=np.int64)
 
    coords = np.argwhere(diff < profile.distanceThreshold * profile.blockSize ** 2)

    diff_values = diff[coords[:, 0], coords[:, 1]]

    sorted_indices = np.argsort(diff_values)

    return coords[sorted_indices]

def findSimilarGroups(image: np.ndarray, profile: BM3DProfile) -> Tuple:

    # Get all possible blocks
    blocks: np.ndarray = np.lib.stride_tricks.sliding_window_view(
        image, (profile.blockSize, profile.blockSize)
    )[::profile.blockStep, ::profile.blockStep]
    
    similarBlocksCoords = []
    similarGroups = []

    for y in range(blocks.shape[0]):
        for x in range(blocks.shape[1]):
            refBlock = blocks[y, x] 
            coords = findSimilarBlocksCoords(refBlock, blocks, profile)

            if coords.shape[0] % 2 != 0:
                coords = coords[:-1]

            if profile.groupMaxSize != 0:
                if coords.shape[0] > profile.groupMaxSize:
                   coords = coords[:profile.groupMaxSize]

            similarBlocksCoords.append(coords * profile.blockStep)

            group = blocks[coords[:, 0], coords[:, 1]]
            similarGroups.append(group)

    return similarBlocksCoords, similarGroups

    
