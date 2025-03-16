import numpy as np

from .blockmatching import findSimilarGroups
from .profile import BM3DProfile
from .transforms import *
from .filtration import *


def bm3d(noisyImage: np.ndarray, noiseVariance: float,
         profile: BM3DProfile) -> np.ndarray:

    basicImage: np.ndarray = _bm3dBasic(noisyImage, noiseVariance, profile)

    return basicImage

def _bm3dBasic(noisyImage: np.ndarray, noiseVariance: float,
         profile: BM3DProfile) -> np.ndarray:
    similarBlocksCoords, similarGroups = findSimilarGroups(noisyImage, profile)

    transformedGroups2D = applyToGroups2DCT(similarGroups)

    transformedCoeffs1D = applyToGroups1DTransform(transformedGroups2D)

    filteredCoeffs1D = applyHardThresholdingFilter(transformedCoeffs1D, profile)

    filteredCoeffs2D = applyToGroup1DInverseTransform(filteredCoeffs1D, similarGroups.shape)

    filteredImage = applyToGroupInverse2DCT(filteredCoeffs2D)

    return filteredImage


    return similarGroups
