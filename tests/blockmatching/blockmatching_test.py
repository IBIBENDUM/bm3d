import cv2
from bm3d import findSimilarGroups, measureTime, BM3DProfile

def visualizeBM(image, group, blockSize=16, color=(255,0,0), alpha=0.5):
    imageRGB = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    imageVis = imageRGB.copy()

    overlay = imageVis.copy()
    for y, x in group:
        cv2.rectangle(overlay, (x, y), (x + blockSize, y + blockSize), color, -1)  
    cv2.addWeighted(overlay, alpha, imageVis, 1 - alpha, 0, imageVis)

    return imageVis

if __name__ == "__main__":
    image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
    profile = BM3DProfile()
    bmResult, bmTime = measureTime(findSimilarGroups, image, profile)
    
    print(bmResult[0])
    # cv2.imwrite("result.png", visualizeBM(image, bmResult[0][0], profile.blockSize))
    
