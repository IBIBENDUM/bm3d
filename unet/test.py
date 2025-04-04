import torch
from model import UNet
from PIL import Image
import numpy as np
from config_manager import Config
from dataloader import getDataLoader
from tqdm import tqdm
import piq
import csv
from pathlib import Path

def loadModel(modelPath, device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(modelPath, map_location=device))
    model.eval()
    return model

def calculateImageMetrics(output, clean):
    """Calculate metrics for single image"""
    return {
        "psnr": piq.psnr(output, clean).item(),
        "ssim": piq.ssim(output, clean).item(),
        "vif": piq.vif_p(output, clean).item(),
    }

def saveDenoisedImage(outputImg, imageIndex, outputDir):
    outputImagePath = outputDir / f"denoised_{imageIndex}.png"
    denoisedImage = outputImg.squeeze(0).cpu().numpy()
    denoisedImage = np.clip(denoisedImage, 0, 1) * 255
    denoisedImage = denoisedImage.astype(np.uint8)
    denoisedImage = Image.fromarray(denoisedImage[0])

    denoisedImage.save(outputImagePath)

def saveMetricsToCsv(metrics, outputCsv):
    metricsNames = ["imageIndex", "psnr", "ssim", "vif"]
    with open(outputCsv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(metricsNames)
        writer.writerows(metrics) 

if __name__ == "__main__":
    config = Config()
    model = loadModel("weights.pth", config.train["device"])
    dataLoader = getDataLoader(
        sourceDir=config.paths["cleanValDir"],
        batchSize=config.train["batchSize"],
        numWorkers=config.train["numWorkers"],
        augment=False,
        shuffle=False,
    )

    results = []

    outputDir = Path("test_output")
    outputDir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for batchIndex, (noisy, clean) in tqdm(enumerate(dataLoader), desc="Evaluating data"):
            noisy = noisy.to(config.train["device"])
            clean = clean.to(config.train["device"])
            outputs = model(noisy)

            for i in range(noisy.size(0)):
                noisyImg = noisy[i].unsqueeze(0)
                cleanImg = clean[i].unsqueeze(0)
                outputImg = outputs[i].unsqueeze(0)

                batchMetrics = calculateImageMetrics(outputImg, cleanImg)

                imageIndex = batchIndex * config.train["batchSize"] + i

                results.append([imageIndex] + list(batchMetrics.values()))

                saveDenoisedImage(outputImg, imageIndex, outputDir)

    saveMetricsToCsv(results, "metrics.csv")
