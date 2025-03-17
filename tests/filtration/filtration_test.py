import cv2
from bm3d.filtration import *
from bm3d.transforms import *
from bm3d.blockmatching import *
from bm3d.experiment import *

if __name__ == "__main__":
    image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
    profile = BM3DProfile()
    bmResult, bmTime = measureTime(findSimilarGroups, image, profile)

    transformedGroups2D = applyToGroups2DCT(bmResult[1])
    # print(f"transformedCoeffs2D: {bmResult[1][0][0]}")

    transformedCoeffs1D = applyToGroups1DTransform(transformedGroups2D)
    # print(transformedCoeffs1D[0])
    #
    filteredCoeffs1D = applyHtToGroups(transformedCoeffs1D, profile)
    # print(f"filteredCoeffs1D: {filteredCoeffs1D[0].shape}")
    #
    # weights = calculateBlocksWeights(filteredCoeffs1D, 10)
    # print(f"weights: {weights[0]}")
    #
    filteredCoeffs2D = applyToGroups1DInverseTransform(transformedCoeffs1D, bmResult[1])
    # print(f"filteredCoeffs2D: {filteredCoeffs2D[0]}")

    filteredImage = applyToGroupsInverse2DCT(transformedGroups2D)
    # print(f"filteredImage: {filteredImage[0][0]}")


