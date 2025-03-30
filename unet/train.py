from datetime import datetime
from pathlib import Path 
import json

import torch
import torch.nn as nn
import torch.optim as optim
import piq
from tqdm import tqdm

from model import UNet
from dataloader import getDataLoader
from early_stopping import EarlyStopping
from plots import (
    plotAndSaveExamples,
    plotLosses,
    plotPsnrImprovements,
    saveLossesToCSV,
    savePsnrImprovementsToCSV,
)
from checkpoints import saveCheckpoint, loadCheckpoint


def loadConfig(configPath="config.json"):
    with open(configPath, "r") as file:
        config = json.load(file)

    if config["device"] == "cuda":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return config


def setupOutputDirectory():
    """ Create output directory with timestamp """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputDir = Path("results") / f"run_{timestamp}"
    outputDir.mkdir(parents=True, exist_ok=True)

    return outputDir


def runEpoch(model, dataLoader, criterion, optimizer, device, isTraining):
    """ Run one epoch of training or validation """
    if isTraining:
        model.train()
        desc = "Training"
        mode = torch.set_grad_enabled(True)
    else:
        model.eval()
        desc = "Validating"
        mode = torch.inference_mode()

    epochLoss = 0
    num_samples = 0
    total_psnr_diff = 0

    with mode:
        for noisy, clean in tqdm(dataLoader, desc=desc):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            
            if isTraining:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
             
            # Calculate PSNRs
            outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())
            batch_size = noisy.size(0)
            for i in range(batch_size):
                noisy_psnr = piq.psnr(noisy[i:i+1], clean[i:i+1])
                denoised_psnr = piq.psnr(outputs[i:i+1], clean[i:i+1])
                total_psnr_diff += (denoised_psnr - noisy_psnr).item()
            
            num_samples += batch_size
            epochLoss += loss.item() * batch_size
    
    avg_psnr_improvement = total_psnr_diff / num_samples
    
    return epochLoss / len(dataLoader.dataset), avg_psnr_improvement


def initModel(outputDir, config):
    """ Initialize model, loss function, and optimizer """
    model = UNet().to(config['device'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learningRate'])
    config['modelSavePath'] = outputDir / "denoising_model.pth"

    return model, criterion, optimizer


def evaluateModel(model, dataLoader, epoch, outputDir, device):
    """ Evaluate the model and save example images with PSNR metrics """
    sampleNoisy, sampleClean = next(iter(dataLoader))
    sampleNoisy = sampleNoisy.to(device)
    sampleClean = sampleClean.to(device)
    
    with torch.no_grad():
        sampleDenoised = model(sampleNoisy)
        plotAndSaveExamples(sampleNoisy, sampleDenoised, sampleClean, epoch, outputDir)


def trainModel():
    """ Training loop that trains model and saves results """
    outputDir = setupOutputDirectory()
    print(f"Saving results to: {outputDir}")

    config = loadConfig()
    model, criterion, optimizer = initModel(outputDir, config)
    device = config["device"]

    startEpoch = 0
    trainLosses, valLosses = [], []
    trainPsnrImps, valPsnrImps = [], []

    try:
        checkpointData = loadCheckpoint(model, optimizer)
        model = checkpointData['model']
        optimizer = checkpointData['optimizer']
        startEpoch = checkpointData['epoch'] + 1 
        trainLosses = checkpointData['train_losses']
        valLosses = checkpointData['val_losses']
        print(f"Resuming from epoch {startEpoch} with last loss {checkpointData['loss']:.4f}")

    except FileNotFoundError:
        print("No checkpoint found. Starting from scratch.")

    trainLoader = getDataLoader(
        sourceDir=config["cleanTrainDir"],
        batchSize=config["batchSize"],
        numWorkers=config["numWorkers"],
        shuffle=True,
    )

    valLoader = getDataLoader(
        sourceDir=config["cleanValDir"],
        batchSize=config["batchSize"],
        numWorkers=config["numWorkers"],
        shuffle=False,
    )


    # earlyStopping = EarlyStopping(patience=10, minDelta=0.005)
    
    for epoch in range(startEpoch, config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Training phase
        trainLoss, trainPsnrImp = runEpoch(
            model, trainLoader, criterion, optimizer, device, isTraining=True)
        
        trainLosses.append(trainLoss)
        trainPsnrImps.append(trainPsnrImp)
        
        # Validation phase
        valLoss, valPsnrImp = runEpoch(
            model, valLoader, criterion, None, device, isTraining=False)
        
        valLosses.append(valLoss)
        valPsnrImps.append(valPsnrImps)
        
        print(f"Train Loss: {trainLoss:.4f} | Val Loss: {valLoss:.4f}")
        print(f"PSNR Improvement: Train: {trainPsnrImp:.2f} dB | Val: {valPsnrImp:.2f} dB")
        
        if epoch % config.get('checkpointInterval', 10) == 0:
            evaluateModel(model, valLoader, epoch, outputDir, device)
            saveCheckpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=valLoss,
                trainLosses=trainLosses,
                valLosses=valLosses
            )

        # earlyStopping(valLoss)
        # if earlyStopping.earlyStop:
        #     print("Early stopping triggered!")
        #     break
    
    # Save results
    torch.save(model.state_dict(), config['modelSavePath'])
    plotLosses(trainLosses, valLosses, str(outputDir))
    saveLossesToCSV(trainLosses, valLosses, str(outputDir))
    plotPsnrImprovements(trainPsnrImps, trainPsnrImps, str(outputDir))
    savePsnrImprovementsToCSV(trainPsnrImps, trainPsnrImps, str(outputDir))

    return model

if __name__ == "__main__":
    trainModel()
