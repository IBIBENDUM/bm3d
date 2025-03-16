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
    print(f"transformedCoeffs2D: {transformedGroups2D[0].shape}")

    transformedCoeffs1D = applyToGroups1DTransform(transformedGroups2D)
    print(transformedCoeffs1D[0].shape)

    filteredCoeffs1D = applyHTtoGroups(transformedCoeffs1D, profile)
    print(f"filteredCoeffs1D: {filteredCoeffs1D[0].shape}")

    filteredCoeffs2D = applyToGroups1DInverseTransform(filteredCoeffs1D, bmResult[1])
    print(f"filteredCoeffs2D: {filteredCoeffs2D[0].shape}")

    filteredImage = applyToGroupsInverse2DCT(filteredCoeffs2D)
    print(f"filteredImage: {filteredImage[0].shape}")


