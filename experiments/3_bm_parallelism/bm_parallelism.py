"""
Description: Comparison of block matching implementations on a single thread
    and on multiple threads using multiprocessing
Result: Speed increase of approximately 3 times
"""

import numpy as np
import cv2
import time
from multiprocessing import Pool, cpu_count
from typing import Callable, Any, Tuple
from bm3d import measureTime


def findSimilarSingleThread(image, refBlockCoords, blockSize=16, threshold=2000):
    """
    Single-threaded implementation of finding similar blocks.
    """
    h, w = image.shape[:2]
    yRef, xRef = refBlockCoords
    refBlock = image[yRef: yRef + blockSize, xRef: xRef + blockSize]

    similarBlocks = []
    for y in range(0, h - blockSize):
        for x in range(0, w - blockSize):
            block = image[y: y + blockSize, x: x + blockSize]
            diff = np.sum((refBlock - block) ** 2)
            if diff < threshold * blockSize:
                similarBlocks.append((y, x))

    return similarBlocks


def findSimilarMultiThread(image, refBlockCoords, blockSize=16, threshold=2000):
    """
    Multi-threaded implementation of finding similar blocks.
    """
    h, w = image.shape[:2]
    yRef, xRef = refBlockCoords
    refBlock = image[yRef: yRef + blockSize, xRef: xRef + blockSize]

    # Generate all possible (y, x) positions
    positions = [(y, x, image, refBlock, blockSize, threshold)
                 for y in range(0, h - blockSize)
                 for x in range(0, w - blockSize)]

    # Use all available CPU cores
    with Pool(cpu_count()) as pool:
        results = pool.map(process_block, positions)

    # Filter out None values
    similarBlocks = [result for result in results if result is not None]

    return similarBlocks


def process_block(args):
    """
    Helper function for parallel processing.
    """
    y, x, image, refBlock, blockSize, threshold = args
    block = image[y: y + blockSize, x: x + blockSize]
    diff = np.sum((refBlock - block) ** 2)

    return (y, x) if diff < threshold * blockSize else None

def main():

    image = cv2.imread("image.png")

    h, w = image.shape[:2]
    refBlockCoords = (h // 2, w // 2)

    # Single-threaded version
    _ , singleThreadTime = measureTime(findSimilarSingleThread,
                                                         image, refBlockCoords)
    print(f"Single-threaded time: {singleThreadTime:.4f} seconds")

    # Multi-threaded version
    _ , multiThreadTime = measureTime(findSimilarMultiThread,
                                                       image, refBlockCoords)
    print(f"Multi-threaded time: {multiThreadTime:.4f} seconds")

    # Speedup comparison
    speedup = singleThreadTime / multiThreadTime
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
