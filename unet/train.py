from datetime import datetime
import os
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model import UNet, PSNRLoss, calculatePsnr
from dataloader import getDataLoader
from plots import saveExamples, saveLosses

def loadConfig(configPath="config.json"):
    with open(configPath, "r") as file:
        config = json.load(file)

    if config["device"] == "cuda":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return config


def setupOutputDirectory():
    """ Create output directory with timestamp """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputDir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(outputDir, exist_ok=True)

    return outputDir


def runEpoch(model, dataLoader, criterion, optimizer, isTraining, config):
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


def initModel(outputDir, config):
    """ Initialize model, loss function, and optimizer """
    model = UNet().to(config['device'])
    criterion = PSNRLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    config['model_save_path'] = os.path.join(outputDir, "denoising_model.pth")
    return model, criterion, optimizer


def evaluateModel(model, dataLoader, epoch, outputDir, config):
    """ Evaluate the model and save example images with PSNR metrics """
    sampleNoisy, sampleClean = next(iter(dataLoader))
    sampleNoisy = sampleNoisy.to(config['device'])
    sampleClean = sampleClean.to(config['device'])
    
    with torch.no_grad():
        sampleDenoised = model(sampleNoisy)
        saveExamples(sampleNoisy, sampleDenoised, sampleClean, epoch, outputDir)


def trainModel():
    """ Training loop that trains model and saves results """
    outputDir = setupOutputDirectory()
    print(f"Saving results to: {outputDir}")


    config = loadConfig()
    model, criterion, optimizer = initModel(outputDir, config)

    trainLoader = getDataLoader(
        sourceDir=config["clean_train_dir"],
        batchSize=config["batch_size"],
        numWorkers=config["num_workers"],
        shuffle=True,
    )

    valLoader = getDataLoader(
        sourceDir=config["clean_val_dir"],
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
            model, trainLoader, criterion, optimizer, isTraining=True, config=config)
        
        trainLosses.append(trainLoss)
        trainNoisyPsnrs.append(trainNoisyPsnr)
        trainDenoisedPsnrs.append(trainDenoisedPsnr)
        
        # Validation phase
        valLoss, valNoisyPsnr, valDenoisedPsnr = runEpoch(
            model, valLoader, criterion, None, isTraining=False, config=config)
        
        valLosses.append(valLoss)
        valNoisyPsnrs.append(valNoisyPsnr)
        valDenoisedPsnrs.append(valDenoisedPsnr)
        
        print(f"Train Loss: {trainLoss:.4f} | Val Loss: {valLoss:.4f}")
        print(f"Train PSNR - Noisy: {trainNoisyPsnr:.2f} dB | Denoised: {trainDenoisedPsnr:.2f} dB")
        
        evaluateModel(model, valLoader, epoch, outputDir, config)
    
    # Save results
    torch.save(model.state_dict(), config['model_save_path'])
    saveLosses(trainLosses, valLosses, outputDir)
    return model

if __name__ == "__main__":
    trainModel()
