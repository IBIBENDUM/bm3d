import numpy as np

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

