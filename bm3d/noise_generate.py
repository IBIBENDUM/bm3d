import numpy as np
import cv2

def generateNoise(noise_variance: float, size: tuple) -> np.ndarray:
    """
    Generate Additive white Gaussian noise (AWGN)
    :param noise_variance: noise variance (sigma)
    :param size: size of resulting noise (image size)
    :return: noise as ndarray
    """
    if noise_variance < 0:
        raise ValueError("Noise variance should be greater than 0")
    

    return np.random.normal(0, noise_variance, size)


def addNoise(image: np.ndarray, noiseVariance: float) -> np.ndarray:
    """
    Add Additive white Gaussian Noise (AWGN) to image
    :param image: image for noise applying
    :param noise_variance: noise variance (sigma)
    """
    noise: np.ndarray = generateNoise(noiseVariance, image.shape)
    noisyImage: np.ndarray = image + noise

    noisyImage_uint = np.zeros(image.shape)
    cv2.normalize(noisyImage, noisyImage_uint, 0, 255, cv2.NORM_MINMAX, dtype=-1)

    return noisyImage_uint.astype(np.uint8)

