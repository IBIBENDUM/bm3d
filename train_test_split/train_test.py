from skimage.metrics import peak_signal_noise_ratio as psnr
from functools import partial
from bayes_opt import BayesianOptimization
import numpy as np

import bm3d
from bm3d.profile import BM3DProfile, BM3DStages

from data_loader import prepareData

import logging


def calculatePsnr(cleanImage, denoisedImage):
    return psnr(cleanImage, denoisedImage, data_range=255)

def tensorToNumpy(imageTensor):
    imageNumpy = imageTensor.squeeze().numpy() * 255.0
    return imageNumpy.astype(np.uint8)

def processBatch(noisyImages, cleanImages, noiseVariance, profile):
    totalPsnr = 0.0

    for i in range(noisyImages.shape[0]):
        noisyImageNumpy = tensorToNumpy(noisyImages[i])
        cleanImageNumpy = tensorToNumpy(cleanImages[i])

        denoisedImageNumpy = bm3d.bm3d(
            noisyImageNumpy,
            noiseVariance,
            profile
        )

        imagePsnr = calculatePsnr(cleanImageNumpy, denoisedImageNumpy)
        totalPsnr += imagePsnr

    return totalPsnr

def processDataset(filterThreshold, distanceThreshold, dataLoader, noiseVariance):
    totalPsnr = 0.0
    profile = BM3DProfile(filterThreshold=filterThreshold,
                          distanceThreshold=int(distanceThreshold),
                          stages=BM3DStages.BASIC_STAGE)

    logging.info(f"Current filterThreshold: {filterThreshold}")                          
    logging.info(f"Current distanceThreshold: {distanceThreshold}")                          

    totalImages = len(dataLoader.dataset)
    for noisyImages, cleanImages in dataLoader:
        batchPsnr = processBatch(
            noisyImages,
            cleanImages,
            noiseVariance,
            profile
        )
        totalPsnr += batchPsnr

    averagePsnr = totalPsnr / totalImages
    logging.info(f"Average PSNR for filterThreshold={filterThreshold},"
                 f"distanceThreshold={distanceThreshold}: {averagePsnr}")

    return averagePsnr

def trainModel(noisyDir="dataset/noisy_dataset/noiseVariance20/train",
               cleanDir="dataset/clean_dataset/train",
               noiseVariance=20):
    logging.info("Loading data...")
    dataLoader = prepareData(noisyDir, cleanDir)
    logging.info("Data loaded successfully.")

    paramBounds = {
        'filterThreshold': (0, 10),
        'distanceThreshold': (10, 200)
    } 

    optimizeWithDataLoader = partial(
        processDataset,
        dataLoader=dataLoader,
        noiseVariance=noiseVariance
    )
    logging.info("Initializing Bayesian Optimization...")
    optimizer = BayesianOptimization(
        f=optimizeWithDataLoader,
        pbounds=paramBounds,
        random_state=0
    )
    logging.info("Starting optimization...")
    optimizer.maximize(init_points=1, n_iter=1)
    logging.info("Optimization completed.")

    bestParams = optimizer.max
    logging.info(f"Best parameters: {bestParams}")

    return bestParams

def testModel(params,
              noisyDir="dataset/noisy_dataset/noiseVariance20/test",
              cleanDir="dataset/clean_dataset/test",
              noiseVariance=20):
    logging.info("Loading data...")
    dataLoader = prepareData(noisyDir, cleanDir)
    logging.info("Data loaded successfully.")

    filterThreshold = params['params']['filterThreshold']
    distanceThreshold = params['params']['distanceThreshold']
    logging.info(f"Testing with filterThreshold={filterThreshold},"
                 f"distanceThreshold={distanceThreshold}")

    averagePsnr = processDataset(
        filterThreshold,
        distanceThreshold,
        dataLoader,
        noiseVariance
    )

    return averagePsnr

def setupLogging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("optimization.log"),
            logging.StreamHandler()
        ]
    )

if __name__ == "__main__":
    setupLogging()
    bestParams = trainModel()
    testModel(bestParams)
