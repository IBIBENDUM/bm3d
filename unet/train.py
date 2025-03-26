from datetime import datetime
import os
import math
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from model import UNet
from config import config
from dataloader import getDataLoader
from plots import saveExamples, saveLosses

def setupOutputDirectory():
    """Create output directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputDir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(outputDir, exist_ok=True)
    return outputDir

def calculatePsnr(img1, img2, maxValue=1.0):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(maxValue) - 10 * math.log10(mse.item())


def runEpoch(model, dataLoader, criterion, optimizer, isTraining=True):
    """ Run one epoch of training or validation """
    if isTraining:
        model.train()
        desc = "Training"
    else:
        model.eval()
        desc = "Validating"

    epochLoss = 0
    totalNoisyPsnr = 0
    totalDenoisedPsnr = 0
    numBatches = 0

    with torch.set_grad_enabled(isTraining):
        for noisy, clean in tqdm(dataLoader, desc=desc):
            noisy = noisy.to(config['device'])
            clean = clean.to(config['device'])
            
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            
            if isTraining:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
             
            # Calculate PSNRs
            batchNoisyPsnr = calculatePsnr(noisy, clean)
            batchDenoisedPsnr = calculatePsnr(outputs, clean)
            
            totalNoisyPsnr += batchNoisyPsnr
            totalDenoisedPsnr += batchDenoisedPsnr
            numBatches += 1
            
            epochLoss += loss.item() * noisy.size(0)
    
    avgNoisyPsnr = totalNoisyPsnr / numBatches
    avgDenoisedPsnr = totalDenoisedPsnr / numBatches
    
    return epochLoss / len(dataLoader.dataset), avgNoisyPsnr, avgDenoisedPsnr

def initModel(outputDir):
    """ Initialize model, loss function, and optimizer """
    model = UNet().to(config['device'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    config['model_save_path'] = os.path.join(outputDir, "denoising_model.pth")
    return model, criterion, optimizer

def evaluateModel(model, dataLoader, epoch, outputDir):
    """Evaluate the model and save example images with PSNR metrics"""
    sampleNoisy, sampleClean = next(iter(dataLoader))
    sampleNoisy = sampleNoisy.to(config['device'])
    sampleClean = sampleClean.to(config['device'])
    
    with torch.no_grad():
        sampleDenoised = model(sampleNoisy)
        
        # Calculate PSNRs for the sample batch
        noisyPsnr = calculatePsnr(sampleNoisy, sampleClean)
        denoisedPsnr = calculatePsnr(sampleDenoised, sampleClean)
        
        print(f"Sample PSNR - Noisy: {noisyPsnr:.2f} dB | Denoised: {denoisedPsnr:.2f} dB")
        saveExamples(sampleNoisy, sampleDenoised, sampleClean, epoch, outputDir)


def trainModel():
    """ Training loop that trains model and saves results """
    outputDir = setupOutputDirectory()
    print(f"Saving results to: {outputDir}")

    model, criterion, optimizer = initModel(outputDir)

    trainLoader = getDataLoader(
        cleanDir=config["clean_train_dir"],
        batchSize=config["batch_size"],
        numWorkers=config["num_workers"],
        shuffle=True,
    )

    valLoader = getDataLoader(
        cleanDir=config["clean_val_dir"],
        batchSize=config["batch_size"],
        numWorkers=config["num_workers"],
        shuffle=False,
    )

    trainLosses, valLosses = [], []
    trainNoisyPsnrs, trainDenoisedPsnrs = [], []
    valNoisyPsnrs, valDenoisedPsnrs = [], []
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Training phase
        trainLoss, trainNoisyPsnr, trainDenoisedPsnr = runEpoch(
            model, trainLoader, criterion, optimizer, isTraining=True)
        
        trainLosses.append(trainLoss)
        trainNoisyPsnrs.append(trainNoisyPsnr)
        trainDenoisedPsnrs.append(trainDenoisedPsnr)
        
        # Validation phase
        valLoss, valNoisyPsnr, valDenoisedPsnr = runEpoch(
            model, valLoader, criterion, None, isTraining=False)
        
        valLosses.append(valLoss)
        valNoisyPsnrs.append(valNoisyPsnr)
        valDenoisedPsnrs.append(valDenoisedPsnr)
        
        print(f"Train Loss: {trainLoss:.4f} | Val Loss: {valLoss:.4f}")
        print(f"Train PSNR - Noisy: {trainNoisyPsnr:.2f} dB | Denoised: {trainDenoisedPsnr:.2f} dB")
        print(f"Val PSNR - Noisy: {valNoisyPsnr:.2f} dB | Denoised: {valDenoisedPsnr:.2f} dB")
        
        evaluateModel(model, valLoader, epoch, outputDir)
    
    # Save results
    torch.save(model.state_dict(), config['model_save_path'])
    saveLosses(trainLosses, valLosses, outputDir)
    return model

if __name__ == "__main__":
    trainModel()
