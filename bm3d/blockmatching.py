import numpy as np
from typing import Tuple
from .profile import BM3DProfile

def findSimilarBlocksCoords(refBlock: np.ndarray, matchingBlocks: np.ndarray,
                      profile: BM3DProfile) -> np.ndarray:
    diff = np.sum((matchingBlocks - refBlock) ** 2, axis=(2, 3), dtype=np.int64)
    return np.argwhere(diff < profile.distanceThreshold * profile.blockSize ** 2)


def findSimilarGroups(image: np.ndarray, profile: BM3DProfile) -> Tuple:

    # Get all possible blocks
    blocks: np.ndarray = np.lib.stride_tricks.sliding_window_view(
        image, (profile.blockSize, profile.blockSize)
    )
    
    similarBlocksCoords = []
    similarGroups = []

    for y in range(0, blocks.shape[0], 16):
        for x in range(0, blocks.shape[1], 16):
            refBlock = blocks[y, x] 
            coords = findSimilarBlocksCoords(refBlock, blocks, profile)

            if coords.shape[0] > profile.groupMaxSize:
               coords = coords[:profile.groupMaxSize]

            similarBlocksCoords.append(coords)

            group = blocks[coords[:, 0], coords[:, 1]]
            similarGroups.append(group)
        

    return similarBlocksCoords, similarGroups

    
