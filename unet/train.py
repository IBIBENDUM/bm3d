import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from model import UNet
from config import config
from dataloader import getDataLoader
from plots import saveExamples, saveLosses

def runEpoch(model, dataLoader, criterion, optimizer, isTraining=True):
    """ Run one epoch of training or validation """
    if isTraining:
        model.train()
        desc = "Training"
    else:
        model.eval()
        desc = "Validating"

    epochLoss = 0
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
            print(f"Pixel value from noisy image at (0, 0): {noisy[0, 0, 0, 0].item()}")
             
            epochLoss += loss.item() * noisy.size(0)
    
    return epochLoss / len(dataLoader.dataset)

def psnrLoss(image1, image2, maxValue=1.0):
    """ Calculate Peak Signal-to-Noise Ratio (PSNR) between two images """
    mse = nn.functional.mse_loss(image1, image2)
    psnr = 10 * torch.log10(maxValue**2 / mse)
    return psnr

def initModel():
    """ Initialize model, loss function, and optimizer """
    model = UNet().to(config['device'])
    criterion = psnrLoss
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    return model, criterion, optimizer

def evaluateModel(model, dataLoader, epoch):
    """ Evaluate the model on the validation set and saves example images """
    sampleNoisy, sampleClean = next(iter(dataLoader))
    sampleNoisy = sampleNoisy.to(config['device'])
    
    with torch.no_grad():
        sampleDenoised = model(sampleNoisy)
        saveExamples(sampleNoisy, sampleDenoised, sampleClean, epoch)


def trainModel():
    """ Training loop that trains model and saves results """
    model, criterion, optimizer = initModel()

    trainLoader = getDataLoader(
        noisyDir=config["noisy_train_dir"],
        cleanDir=config["clean_train_dir"],
        batchSize=config["batch_size"],
        numWorkers=config["num_workers"],
        shuffle=True,
    )

    valLoader = getDataLoader(
        noisyDir=config["noisy_val_dir"],
        cleanDir=config["clean_val_dir"],
        batchSize=config["batch_size"],
        numWorkers=config["num_workers"],
        shuffle=False,
    )

    trainLosses, valLosses = [], []
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Training phase
        trainLoss = runEpoch(model, trainLoader, criterion, optimizer, isTraining=True)
        trainLosses.append(trainLoss)
        
        # Validation phase
        valLoss = runEpoch(model, valLoader, criterion, None, isTraining=False)
        valLosses.append(valLoss)
        
        print(f"Train Loss: {trainLoss:.4f} | Val Loss: {valLoss:.4f}")
        evaluateModel(model, valLoader, epoch)
    
    # Save results
    torch.save(model.state_dict(), config['model_save_path'])
    saveLosses(trainLosses, valLosses)
    return model

if __name__ == "__main__":
    trainModel()
