import numpy as np
from .profile import BM3DProfile

def findSimilarBlocks(image: np.ndarray, refBlock: np.ndarray,
                      matchingBlocks: np.ndarray, profile: BM3DProfile) -> list:
    diff = np.sum((matchingBlocks - refBlock) ** 2, axis=(2, 3), dtype=np.int64)
    similarIndices = np.argwhere(diff < profile.distanceThreshold * profile.blockSize)

    return [tuple(idx) for idx in similarIndices]

def findSimilarGroups(image: np.ndarray, profile: BM3DProfile) -> list:
    blocks: np.ndarray = np.lib.stride_tricks.sliding_window_view(
        image, (profile.blockSize, profile.blockSize)
    )
    
    groups = [] # TODO: Add max size and make with numpy
    for y in range(0, blocks.shape[0], 100):
        for x in range(0, blocks.shape[1], 100):
            refBlock = blocks[y, x] 
            similarBlocks = findSimilarBlocks(image, refBlock, blocks, profile)
            groups.append(similarBlocks)

    return groups
    
    
