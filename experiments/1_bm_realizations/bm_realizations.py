"""
Description: OpenCV (CPU, OpenCL) and Numpy BM speed comparison of searching
    for similar to reference fragments WITHOUT optimizations
Results: For my laptop (Matebook D15 Ryzen 7 5700U) cv2.matchTemplate on CPU
    3 times faster than GPU, BM on Numpy without optimizations 10 times slower
"""

import cv2
import numpy as np
import time
from typing import Callable, Any, Tuple


def measureTime(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure the execution time of a given function.
    """
    startTime = time.perf_counter()
    result = func(*args, **kwargs) 
    endTime = time.perf_counter()  
    elapsedTime = endTime - startTime 

    return result, elapsedTime


def findSimilarCVCPU(image, refBlockCoords, blockSize=16, threshold=0.8):
    """
    Find similar blocks using OpenCV's matchTemplate
    """
    yRef, xRef = refBlockCoords
    refBlock = image[yRef : yRef + blockSize, xRef : xRef + blockSize]

    result = cv2.matchTemplate(image, refBlock, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    return list(zip(locations[0], locations[1]))


def findSimilarCVGPU(image, refBlockCoords, blockSize=16, threshold=0.8):
    """
    Find similar blocks using OpenCV with OpenCL 
    """
    imageUMat = cv2.UMat(image)

    yRef, xRef = refBlockCoords
    refBlock = imageUMat.get()[yRef : yRef + blockSize, xRef : xRef + blockSize]

    result = cv2.matchTemplate(imageUMat, refBlock, cv2.TM_CCOEFF_NORMED)
    result = result.get()
    locations = np.where(result >= threshold)

    return list(zip(locations[0], locations[1]))


def findSimilarNumpyBM(image, refBlockCoords, blockSize=16, threshold=2000):
    """
    Find similar blocks using Numpy
    """
    h, w = image.shape[:2]
    yRef, xRef = refBlockCoords
    refBlock = image[yRef : yRef + blockSize, xRef : xRef + blockSize]

    similarBlocks = []
    for y in range(0, h - blockSize):
        for x in range(0, w - blockSize):
            block = image[y : y + blockSize, x : x + blockSize]
            diff = np.sum((refBlock - block) ** 2)  
            if diff < threshold * blockSize:
                similarBlocks.append((y, x))

    return similarBlocks


def visualizeResults(image, refBlockCoords, similarBlocksList,
                     labels, blockSize=16, alpha=0.5):
    """
    Visualize the results of finding similar blocks
    """
    imageVis = image.copy()
    yRef, xRef = refBlockCoords

    # Draw the reference block
    cv2.rectangle(imageVis, (xRef, yRef),
                  (xRef + blockSize, yRef + blockSize), (0, 0, 0), 2)

    cv2.putText(imageVis, "Reference Block", (xRef, yRef - 10),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

    # Draw similar blocks for each implementation
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, (similarBlocks, label) in enumerate(zip(similarBlocksList, labels)):
        overlay = imageVis.copy()
        for y, x in similarBlocks:
            cv2.rectangle(overlay, (x, y), (x + blockSize, y + blockSize),
                          colors[i], -1)  
        cv2.putText(imageVis, label, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, colors[i], 2)
        cv2.addWeighted(overlay, alpha, imageVis, 1 - alpha, 0, imageVis)

    return imageVis


def main():
    image = cv2.imread("image.png")

    # Coordinates of the reference block (center of the image)
    h, w = image.shape[:2]
    refBlockCoords = (h // 2, w // 2)

    # Measure time for OpenCV
    similarBlocksCVCPU, opencvTime = measureTime(findSimilarCVCPU,
                                                  image, refBlockCoords)

    # Measure time for OpenCL
    similarBlocksCVGPU, openclTime = measureTime(findSimilarCVGPU,
                                                  image, refBlockCoords)

    # Measure time for Numpy
    similarBlocksNumpy, numpyTime = measureTime(findSimilarNumpyBM,
                                                image, refBlockCoords)

    # Visualize results
    labels = ["OpenCV", "OpenCL", "Numpy"]
    similarBlocksList = [similarBlocksCVCPU, similarBlocksCVGPU, similarBlocksNumpy]
    resultimage = visualizeResults(
        image, refBlockCoords, similarBlocksList, labels,
    )

    # Save and display results
    cv2.imwrite("result_comparison.png", resultimage)
    print(f"OpenCV block matching time: {opencvTime:.4f} seconds")
    print(f"OpenCL block matching time: {openclTime:.4f} seconds")
    print(f"Numpy block matching time: {numpyTime:.4f} seconds")


if __name__ == "__main__":
    main()
