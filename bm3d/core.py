import numpy as np

from .blockmatching import findSimilarGroups
from .profile import BM3DProfile
from .transforms import *
from .filtration import *
from .agregation import *


def bm3d(noisyImage: np.ndarray, noiseVariance: float,
         profile: BM3DProfile) -> np.ndarray:

    basicImage: np.ndarray = _bm3dBasic(noisyImage, noiseVariance, profile)

    return basicImage

def _bm3dBasic(noisyImage: np.ndarray, noiseVariance: float,
         profile: BM3DProfile) -> np.ndarray:
    groupsCoords, groups = findSimilarGroups(noisyImage, profile)

    transformedGroups2D = applyToGroups2DCT(groups)

    transformedCoeffs1D = applyToGroups1DTransform(transformedGroups2D)

    filteredCoeffs1D = applyHTtoGroups(transformedCoeffs1D, profile)
    weights = calculateBlocksWeights(filteredCoeffs1D, noiseVariance)

    filteredCoeffs2D = applyToGroups1DInverseTransform(filteredCoeffs1D, groups)

    filteredGroups = applyToGroupsInverse2DCT(filteredCoeffs2D)

    imageBasic = agregationBasic(noisyImage.shape, filteredGroups,
                                 groupsCoords, weights)
    return imageBasic

