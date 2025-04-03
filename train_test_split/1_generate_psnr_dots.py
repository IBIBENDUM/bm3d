import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
import bm3d
from bm3d.profile import BM3DProfile
from dataloader import getDataLoader
from tqdm import tqdm

def generateParameterSamples(nSamples=50, seed=42):
    """Генерирует точки параметров через LHS."""
    param_ranges = {
        "blockSize": [8, 16, 32],
        "blockStep": [8, 16],
        "searchWindow": [5, 10, 15],
        "distanceThreshold": [140, 160, 180, 200],
        "groupMaxSize": [4, 8, 16, 24],
        "filterThreshold": np.linspace(3.0, 5.0, 5),
        "kaiserShape": np.linspace(1.5, 3.0, 5),
    }
    return list(ParameterSampler(param_ranges, n_iter=nSamples, random_state=seed))


def evaluateBM3D(noisy, clean, params, sigma=25):
    profile = BM3DProfile(
        blockSize=params["blockSize"],
        blockStep=params["blockStep"],
        searchWindow=params["searchWindow"],
        distanceThreshold=params["distanceThreshold"],
        groupMaxSize=params["groupMaxSize"],
        filterThreshold=params["filterThreshold"],
        kaiserShape=params["kaiserShape"],
    )
    denoised = bm3d.bm3d(noisy, sigma, profile)
    
    return bm3d.calculatePSNR(clean, denoised)

def evaluateAllSamples(samples, dataloader, nImages):
    results = []
    for params in tqdm(samples, desc="Evaluating parameters"):
        psnrs = []
        for i, (cleanTensor, _) in enumerate(dataloader):
            if i >= nImages:
                break

            clean = cleanTensor.numpy()[0, 0]
            clean = (np.clip(clean, 0, 1) * 255).astype(np.uint8)
            noisy = bm3d.addNoise(clean, 25)
            psnr = evaluateBM3D(noisy, clean, params)
            psnrs.append(psnr)

        results.append({**params, "psnr": np.mean(psnrs)})
    return results

if __name__ == "__main__":
    dataloader = getDataLoader(
        sourceDir="dataset/split_dataset/train",
        numWorkers=3,
        augment=False,
        shuffle=False,
    )  

    samples = generateParameterSamples(nSamples=50)

    results = evaluateAllSamples(samples, dataloader, nImages=10)
    pd.DataFrame(results).to_csv("bm3d_evaluation.csv", index=False)

