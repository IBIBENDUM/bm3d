import numpy as np

def prepareImageDimensions(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2 and image.ndim != 3:
        raise ValueError("Noisy image should be 2D or 3D")

    if image.ndim == 2:
        return np.atleast_3d(image)

    return np.atleast_3d(image)

def bm3d(noisyImage: np.ndarray, noiseVariance: float) -> np.ndarray:
    # Check is image 2D or 3D
    noisyImage: np.ndarray = prepareImageDimensions(noisyImage)

    pass
