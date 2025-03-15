"""
Description: Comparison of similar fragment search methods for cv2.matchTemplate
Results: Depending on the method, it is necessary to use different parameters,
    the final result and speed of work are approximately the same
"""

import cv2
import numpy as np
import time
from typing import Callable, Any, List, Tuple
from bm3d import measureTime


def blockMatchingOpenCV(image, refBlockCoords, blockSize=16, threshold=0.8, method=cv2.TM_CCOEFF_NORMED):
    """
    Find similar blocks using OpenCV's matchTemplate with a specified method.
    """
    yRef, xRef = refBlockCoords
    refBlock = image[yRef: yRef + blockSize, xRef: xRef + blockSize]

    result = cv2.matchTemplate(image, refBlock, method)
    
    # Find locations based on the method
    if method == cv2.TM_SQDIFF_NORMED:
        locations = np.where(result <= threshold)
    else:
        locations = np.where(result >= threshold)

    return list(zip(locations[0], locations[1]))


def saveResults(image, refBlockCoords, similarBlocks: List[Tuple[int, int]], outputPath: str, blockSize=16, alpha=0.5):
    """
    Save the results of block matching to a file.
    """
    imageVis = image.copy()
    yRef, xRef = refBlockCoords

    overlay = imageVis.copy()

    cv2.rectangle(overlay, (xRef, yRef),
                  (xRef + blockSize, yRef + blockSize), (0, 0, 0), 2)

    cv2.putText(overlay, "Reference Block", (xRef, yRef - 10),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
    cv2.addWeighted(overlay, alpha, imageVis, 1 - alpha, 0, imageVis)

    for y, x in similarBlocks:
        cv2.rectangle(overlay, (x, y), (x + blockSize, y + blockSize),
                      (0, 255, 0), 2)  

    cv2.addWeighted(overlay, alpha, imageVis, 1 - alpha, 0, imageVis)
    cv2.imwrite(outputPath, imageVis)
    print(f"Results saved to: {outputPath}")


def main():
    image = cv2.imread("image.png")

    # Coordinates of the reference block (center of the image)
    h, w = image.shape[:2]
    refBlockCoords = (h // 2, w // 2)
    blockSize = 16

    # Methods to compare
    methods = [
        ("TM_CCOEFF_NORMED", cv2.TM_CCOEFF_NORMED, 0.75),
        ("TM_CCORR_NORMED", cv2.TM_CCORR_NORMED, 0.95),
        ("TM_SQDIFF_NORMED", cv2.TM_SQDIFF_NORMED, 0.1),
    ]

    for method_name, method, theshold in methods:
        similarBlocks, elapsedTime = measureTime(
            blockMatchingOpenCV, image, refBlockCoords, blockSize, theshold, method)

        print(f"{method_name} block matching time: {elapsedTime:.4f} seconds")

        outputPath = f"result_{method_name}.png"
        saveResults(image, refBlockCoords, similarBlocks, outputPath, blockSize)


if __name__ == "__main__":
    main()
