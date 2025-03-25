import os
import shutil 
import cv2
import imagesize
from typing import Optional

from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split

import numpy as np

def addNoise(image: np.ndarray, noiseVariance: float) -> np.ndarray:

    gaussianNoise = np.random.normal(0, noiseVariance, image.shape)
    gaussianNoise = gaussianNoise.reshape(image.shape)
    noisyImage = image + gaussianNoise
    noisyImage = np.clip(noisyImage, 0, 255)
    noisyImage = noisyImage.astype(np.uint8)

    return noisyImage


def downloadDataset(datasetName: str="meriemelkhal/random",
                    downloadDir: str="original_dataset"):

    if os.path.exists(downloadDir):
        print(f"Dataset already downloaded")
        return

    os.makedirs(downloadDir)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(datasetName,
                               path=downloadDir,
                               unzip=True)

    print(f"Dataset {datasetName} downloaded to {downloadDir}")


def cropImage(image, cropSize = (128, 128)):
    height, width = image.shape[:2]
    startX, startY = (width - cropSize[0]) // 2, \
                     (height - cropSize[1]) // 2
    return image[startY:startY + cropSize[1],\
                 startX:startX + cropSize[0]]

def createEmptyDirectory(dirPath):
    if os.path.exists(dirPath):
        shutil.rmtree(dirPath) 
    os.makedirs(dirPath)


def getValidImagePaths(sourceDir, minSize=(512,512)):
    minSize = minSize or (0, 0)
    validImages = [
        f for f in os.listdir(sourceDir)
        if f.endswith(('.jpg', '.png', '.jpeg')) and
           imagesize.get(os.path.join(sourceDir, f)) >= minSize
    ]

    return validImages

def processAndSaveImages(imageFiles, sourceDir, cleanDir,
                         noisyDirs: dict, cropSize):
    for file in imageFiles:
        sourceImagePath = os.path.join(sourceDir, file)
        image = cv2.imread(sourceImagePath, cv2.IMREAD_GRAYSCALE)

        if cropSize is not None:
            image = cropImage(image, cropSize)

        cv2.imwrite(os.path.join(cleanDir, file), image)

        for variance, noisyDir in noisyDirs.items():
            noisyImage = addNoise(image, variance)
            cv2.imwrite(os.path.join(noisyDir, file), noisyImage)


def splitDataset(sourceDir: str="original_dataset", 
                 baseCleanDir: str="clean_dataset",
                 baseNoisyDir: str="noisy_dataset",
                 testSize: float=0.2,
                 maxImages: Optional[int]=None,
                 noiseVariances: list=[20, 30, 40],
                 cropSize=None):

    imageFiles = getValidImagePaths(sourceDir, cropSize) 
    if maxImages is not None and len(imageFiles):
        imageFiles = imageFiles[:maxImages]

    trainFiles, testFiles = train_test_split(imageFiles,
                                             test_size=testSize,
                                             random_state=0)

    cleanTrainDir = os.path.join(baseCleanDir, 'train')
    cleanTestDir = os.path.join(baseCleanDir, 'test')
    createEmptyDirectory(cleanTrainDir)
    createEmptyDirectory(cleanTestDir)

    noisyDirs = {
        variance: {
            "train": os.path.join(baseNoisyDir,
                                  f"noiseVariance{variance}",
                                  "train"),
            "test": os.path.join(baseNoisyDir,
                                 f"noiseVariance{variance}",
                                 "test"),
        }
        for variance in noiseVariances
    }

    for varianceFolders in noisyDirs.values():
        for dirPath in varianceFolders.values():
            createEmptyDirectory(dirPath)

    for dataType, files, cleanDir in [
        ("train", trainFiles, cleanTrainDir),
        ("test", testFiles, cleanTestDir),
    ]:
        processAndSaveImages(
            files,
            sourceDir,
            cleanDir,
            {v: d[dataType] for v, d in noisyDirs.items()},
            cropSize,
        )


    print("Dataset splitted")


if __name__ == "__main__":
    # downloadDataset("tarekmebrouk/cbsd68")
    splitDataset(maxImages=None, cropSize=(128, 128), noiseVariances=[20])
