import random
import logging
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from functools import partial
from bayes_opt import BayesianOptimization
from skimage.metrics import peak_signal_noise_ratio as psnr
from dataloader import getDataLoader
from sklearn.gaussian_process import kernels
from matplotlib import gridspec
from bayes_opt import acquisition
import pickle

import bm3d
from bm3d.profile import BM3DProfile, BM3DStages

def setSeeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def setupLogging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("optimization.log"),
            logging.StreamHandler()
        ]
    )

def calculatePsnr(cleanImage, denoisedImage):
    return psnr(cleanImage, denoisedImage, data_range=255)

def tensorToNumpy(imageTensor):
    imageNumpy = imageTensor.squeeze().numpy() * 255.0
    return imageNumpy.astype(np.uint8)

def denoiseImage(noisyImage, noiseVariance, profile):
    return bm3d.bm3d(noisyImage, noiseVariance, profile)

def processBatch(noisyImages, cleanImages, noiseVariance, profile):
    totalPsnr = 0.0

    for i in range(noisyImages.shape[0]):
        noisyImageNumpy = tensorToNumpy(noisyImages[i])
        cleanImageNumpy = tensorToNumpy(cleanImages[i])

        denoisedImageNumpy = bm3d.bm3d(noisyImageNumpy, noiseVariance, profile)
        imagePsnr = calculatePsnr(cleanImageNumpy, denoisedImageNumpy)
        totalPsnr += imagePsnr

    return totalPsnr

def processDataset(filterThreshold, dataLoader, noiseVariance):
    totalPsnr = 0.0
    profile = BM3DProfile(filterThreshold=filterThreshold / 100,
                          stages=BM3DStages.BASIC_STAGE)

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
    return averagePsnr

def posterior(optimizer, grid):
    grid = grid.reshape(-1, 1)
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plotGp(optimizer, xObs, yObs, range):
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(f'Gaussian Process and Utility Function After {steps} Steps', fontsize=30)

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    xGrid = np.linspace(range[0], range[1], 400).reshape(-1, 1)

    optimizer.acquisition_function._fit_gp(optimizer._gp, optimizer._space)
    mu, sigma = optimizer._gp.predict(xGrid, return_std=True)

    axis.plot(xGrid, mu, 'b-', lw=2, label='Mean Prediction')
    axis.fill_between(xGrid.flatten(), mu - 1.96 * sigma, mu + 1.96 * sigma, 
                      color='c', alpha=0.3, label='95% Confidence Interval')

    axis.scatter(xObs, yObs, color='red', s=60, label='Observations', zorder=3)

    axis.set_xlim(range)
    axis.set_xlabel("Filter Threshold", fontsize=20)
    axis.set_ylabel("PSNR", fontsize=20)
    axis.legend()

    utilityFunction = acquisition.UpperConfidenceBound(kappa=5)
    utility = -1 * utilityFunction._get_acq(gp=optimizer._gp)(xGrid)

    acq.plot(xGrid, utility, 'purple', lw=2, label='Utility Function')
    bestX = xGrid[np.argmax(utility)]
    acq.axvline(bestX, linestyle="--", color="gold", label="Next Best Guess")

    acq.set_xlim(range)
    acq.set_xlabel("Filter Threshold", fontsize=20)
    acq.set_ylabel("Utility", fontsize=20)
    acq.legend()

    plt.savefig("gp_plot_fixed.png", bbox_inches='tight', dpi=300)

def saveOptimizerState(optimizer, filename):
    with open(filename, 'wb') as f:
        pickle.dump(optimizer, f)
    logging.info(f"Optimizer state saved to {filename}")

def loadOptimizerState(filename):
    if Path(filename).exists():
        with open(filename, 'rb') as f:
            optimizer = pickle.load(f)
        logging.info(f"Optimizer state loaded from {filename}")
        return optimizer
    else:
        logging.info(f"{filename} not found. Starting from scratch.")
        return None

def optimizeParameters(dataLoader, noiseVariance):
    filterThresholdRange = (200, 400) 
    paramBounds = {
        'filterThreshold': filterThresholdRange
    }

    optimizeFunction = partial(processDataset, dataLoader=dataLoader, noiseVariance=noiseVariance)

    optimizer = loadOptimizerState("optimizer_state.pkl")
    
    if optimizer is None:
        optimizer = BayesianOptimization(
            f=optimizeFunction,
            pbounds=paramBounds,
            random_state=0
        )

    kernel = kernels.ConstantKernel(1.0, (10, 50)) + kernels.Matern(length_scale=100) 
    optimizer.set_gp_params(kernel=kernel, alpha=1e-5)  
    optimizer.maximize(init_points=15, n_iter=15)  

    saveOptimizerState(optimizer, "optimizer_state.pkl")

    x = np.array([[res['params']['filterThreshold']] for res in optimizer.res])  
    y = np.array([res['target'] for res in optimizer.res])
    plotGp(optimizer, x, y, filterThresholdRange)

    return optimizer.max

def trainModel(cleanDir, noisyDir, noiseVariance=25):
    logging.info("Loading data...")
    dataLoader = getDataLoader(cleanDir, noisyDir)
    logging.info("Data loaded successfully.")

    bestParams = optimizeParameters(dataLoader, noiseVariance)
    return bestParams

if __name__ == "__main__":
    setSeeds()
    setupLogging()

    cleanTrainDir = "dataset/dataset/clean"
    noisyTrainDir = "dataset/dataset/train"

    bestParams = trainModel(cleanTrainDir, noisyTrainDir, noiseVariance=25)

    filterThreshold = bestParams['params']['filterThreshold']

    logging.info(f"Testing with filterThreshold={filterThreshold}")
