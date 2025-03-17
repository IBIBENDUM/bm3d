"""
Functions for calculating PSNR (Peak Signal-to-Noise Ratio) and
SSIM (Structural Similarity Index)
"""

import numpy as np


def calculatePSNR(firstImage: np.ndarray,
                  secondImage: np.ndarray,
                  maxValue=255) -> float:
    """
    Calculate PSNR between two images.
    """

    mse = np.mean((firstImage - secondImage) ** 2)

    if mse == 0:
        return 100

    return 10 * np.log10((maxValue ** 2) / mse)
