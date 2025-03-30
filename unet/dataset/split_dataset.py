from pathlib import Path
import cv2
import imagesize
from typing import Optional

from sklearn.model_selection import train_test_split

from directory_funcs import createEmptyDirectory

def cropImage(image, cropSize = (128, 128)):
    height, width = image.shape[:2]
    startX, startY = (width - cropSize[0]) // 2, \
                     (height - cropSize[1]) // 2
    return image[startY:startY + cropSize[1],\
                 startX:startX + cropSize[0]]

def getValidImagePaths(sourceDir, minSize=(512,512)):
    sourcePath = Path(sourceDir)
    validImages = []
    for file in sourcePath.iterdir():
        if file.suffix.lower() in ('.jpg', '.png', '.jpeg'):
            width, height = imagesize.get(str(file))
            if width >= minSize[0] and height >= minSize[1]:
                validImages.append(file.name)

    return validImages

def processAndSaveImage(srcPath, dstPath, cropSize=None):
    image = cv2.imread(srcPath, cv2.IMREAD_GRAYSCALE)
    if cropSize is not None:
        image = cropImage(image, cropSize)
    cv2.imwrite(dstPath, image)


def splitDataset(sourceDir: str="original_dataset", 
                 outputDir: str="split_dataset",
                 testSize: float=0.2,
                 maxImages: Optional[int]=None,
                 cropSize: Optional[tuple]=None):

    imageFiles = getValidImagePaths(sourceDir, cropSize) 
    if maxImages is not None and len(imageFiles) > maxImages:
        imageFiles = imageFiles[:maxImages]

    trainFiles, testFiles = train_test_split(imageFiles,
                                             test_size=testSize,
                                             random_state=0)

    outputPath = Path(outputDir)
    sourcePath = Path(sourceDir)
    createEmptyDirectory(str(outputPath / 'train'))
    createEmptyDirectory(str(outputPath / 'test'))

    for dataType, files in [('train', trainFiles), ('test', testFiles)]:
        destPath = outputPath / dataType
        for file in files:
            src = sourcePath / file
            dst = destPath / file
            processAndSaveImage(src, dst, cropSize)

    print(f"Dataset split completed: {len(trainFiles)} train, {len(testFiles)} test images")
    if cropSize:
        print(f"All images center-cropped to {cropSize}")


if __name__ == "__main__":
    splitDataset(maxImages=None, cropSize=(128, 128))
