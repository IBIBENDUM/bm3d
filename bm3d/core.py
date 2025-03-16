import numpy as np

from .blockmatching import findSimilarGroups
from .profile import BM3DProfile

def prepareImageDimensions(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2 and image.ndim != 3:
        raise ValueError("Noisy image should be 2D or 3D")

    if image.ndim == 2:
        return np.atleast_3d(image)

    return np.atleast_3d(image)

def bm3d(noisyImage: np.ndarray, noiseVariance: float,
         profile: BM3DProfile) -> np.ndarray:
    # Check is image 2D or 3D
    noisyImagePrepared: np.ndarray = prepareImageDimensions(noisyImage)

    basicImage = _bm3dBasic(noisyImagePrepared, noiseVariance, profile)

    return basicImage

def _bm3dBasic(noisyImage: np.ndarray, noiseVariance: float,
         profile: BM3DProfile) -> np.ndarray:
    similarGroups = findSimilarGroups(noisyImage, noiseVariance, profile)
    return similarGroups
