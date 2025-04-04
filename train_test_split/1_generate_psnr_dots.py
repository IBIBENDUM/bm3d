import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
import bm3d
from bm3d.profile import BM3DProfile
from dataloader import getDataLoader
from tqdm import tqdm

def generate_single_param_samples(param_name, param_values, n_samples=50):
    """Генерирует параметры, изменяя только один параметр, остальные фиксированы"""
    base_params = {
        "distanceThreshold": 170,
        "filterThreshold": 3.0,
        "kaiserShape": 2.25
    }
    
    samples = []
    for value in param_values:
        params = base_params.copy()
        params[param_name] = value
        samples.append(params)
    
    return samples[:n_samples]

def evaluateBM3D(noisy, clean, params, sigma=25):
    profile = BM3DProfile(
        distanceThreshold=params["distanceThreshold"],
        filterThreshold=params["filterThreshold"],
        kaiserShape=params["kaiserShape"],
    )
    denoised = bm3d.bm3d(noisy, sigma, profile)
    return bm3d.calculatePSNR(clean, denoised)

def evaluate_parameter(param_name, param_values, dataloader, n_images=5):
    """Оценивает производительность для одного параметра"""
    results = []
    samples = generate_single_param_samples(param_name, param_values)
    
    for params in tqdm(samples, desc=f"Evaluating {param_name}"):
        psnrs = []
        for i, (cleanTensor, _) in enumerate(dataloader):
            if i >= n_images:
                break

            clean = cleanTensor.numpy()[0, 0]
            clean = (np.clip(clean, 0, 1) * 255).astype(np.uint8)
            noisy = bm3d.addNoise(clean, 25)
            psnr = evaluateBM3D(noisy, clean, params)
            psnrs.append(psnr)

        results.append({
            param_name: params[param_name],
            "psnr": np.mean(psnrs)
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    dataloader = getDataLoader(
        sourceDir="dataset/split_dataset/train",
        numWorkers=3,
        augment=False,
        shuffle=False,
    )
    
    # Анализируем каждый параметр по отдельности
    param_ranges = {
        "filterThreshold": np.linspace(1.0, 5.0, 20),
        "distanceThreshold": range(140, 201, 5),
        "kaiserShape": np.linspace(1.5, 3.0, 20),
    }
    
    for param_name, values in param_ranges.items():
        df = evaluate_parameter(param_name, values, dataloader)
        # Сохраняем только имя параметра и PSNR
        df[[param_name, "psnr"]].to_csv(f"bm3d_{param_name}_analysis.csv", index=False)
