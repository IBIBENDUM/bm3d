from datetime import datetime
from pathlib import Path 

import torch
import torch.nn as nn
import torch.optim as optim
import piq
from tqdm import tqdm

from model import UNet
from checkpoint_manager import CheckpointManager
from config_manager import ConfigManager
from dataloader import getDataLoader
from plots import (
    plotAndSaveExamples,
    plotLosses,
    plotPsnrImprovements,
    saveLossesToCSV,
    savePsnrImprovementsToCSV,
)


class ModelTrainer:
    def __init__(self):
        self.outputDir = self.setupOutputDirectory()

        self.config = ConfigManager().config
        if self.config.enableCheckpoints:
            self.checkpointManager = CheckpointManager()

        self.model = UNet().to(self.config.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.learningRate
        )

        self.initDataloaders()
        self.trainLosses = []
        self.valLosses = []
        self.trainPsnrImps = []
        self.valPsnrImps = []


    def setupOutputDirectory(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outputDir = Path("results") / f"run_{timestamp}"
        outputDir.mkdir(parents=True, exist_ok=True)

        print(f"Saving results to: {outputDir}")

        return outputDir


    def initDataloaders(self):
        self.trainLoader = getDataLoader(
            sourceDir=self.config.cleanTrainDir,
            batchSize=self.config.batchSize,
            numWorkers=self.config.numWorkers,
            shuffle=True,
        )

        self.valLoader = getDataLoader(
            sourceDir=self.config.cleanValDir,
            batchSize=self.config.batchSize,
            numWorkers=self.config.numWorkers,
            shuffle=False,
        )


    def runEpoch(self, dataLoader, isTraining):
        if isTraining:
            self.model.train()
            desc = "Training"
            mode = torch.set_grad_enabled(True)
        else:
            self.model.eval()
            desc = "Validating"
            mode = torch.inference_mode()

        epochLoss = 0
        totalPsnrDiff = 0

        with mode:
            for noisy, clean in tqdm(dataLoader, desc=desc):
                noisy = noisy.to(self.config.device)
                clean = clean.to(self.config.device)
                outputs = self.model(noisy)
                loss = self.criterion(outputs, clean)

                if isTraining:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                psnrDiff = self.calculatePsnrDiff(noisy, clean, outputs)
                epochLoss += loss.item() * noisy.size(0)
                totalPsnrDiff += psnrDiff

        return {
            'loss': epochLoss / len(dataLoader.dataset),
            'psnrImp': totalPsnrDiff / len(dataLoader.dataset)
        }

    
    def calculatePsnrDiff(self, noisy, clean, outputs):
        outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())
        psnrDiff = 0
        
        for i in range(noisy.size(0)):
            noisyPsnr = piq.psnr(noisy[i:i+1], clean[i:i+1])
            denoisedPsnr = piq.psnr(outputs[i:i+1], clean[i:i+1])
            psnrDiff += (denoisedPsnr - noisyPsnr).item()
        
        return psnrDiff


    def evaluateModel(self, epoch):
        sampleNoisy, sampleClean = next(iter(self.valLoader))
        sampleNoisy = sampleNoisy.to(self.config.device)
        sampleClean = sampleClean.to(self.config.device)
        
        with torch.inference_mode():
            sampleDenoised = self.model(sampleNoisy)
            plotAndSaveExamples(
                sampleNoisy, 
                sampleDenoised, 
                sampleClean, 
                epoch, 
                str(self.outputDir)
            )


    def saveResults(self):
        modelSavePath = self.outputDir / "denoising_model.pth"
        torch.save(self.model.state_dict(), modelSavePath)
        
        plotLosses(self.trainLosses, self.valLosses, str(self.outputDir))
        saveLossesToCSV(self.trainLosses, self.valLosses, str(self.outputDir))
        plotPsnrImprovements(self.trainPsnrImps, self.valPsnrImps, str(self.outputDir))
        savePsnrImprovementsToCSV(self.trainPsnrImps, self.valPsnrImps, str(self.outputDir))


    def loadCheckpoint(self):
        try:
            checkpointData = self.checkpointManager.load(self.model, self.optimizer)
            self.model = checkpointData['model']
            self.optimizer = checkpointData['optimizer']
            self.trainLosses = checkpointData['trainLosses']
            self.valLosses = checkpointData['valLosses']
            print(f"Resuming from epoch {checkpointData['epoch'] + 1}")
            return checkpointData['epoch'] + 1

        except FileNotFoundError:
            print("No checkpoint found. Starting from scratch.")
            return 0


    def train(self):
        startEpoch = self.loadCheckpoint() if self.config.enableCheckpoints else 0

        for epoch in range(startEpoch, self.config["epochs"]):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            trainMetrics = self.runEpoch(self.trainLoader, isTraining=True)
            valMetrics = self.runEpoch(self.valLoader, isTraining=False)
            
            self.trainLosses.append(trainMetrics['loss'])
            self.trainPsnrImps.append(trainMetrics['psnrImp'])
            self.valLosses.append(valMetrics['loss'])
            self.valPsnrImps.append(valMetrics['psnrImp'])
            
            print(
                f"Train Loss: {trainMetrics['loss']:.4f} | "
                f"Val Loss: {valMetrics['loss']:.4f}"
            )
            print(
                f"PSNR Improvement: Train: {trainMetrics['psnrImp']:.2f} dB | "
                f"Val: {valMetrics['psnrImp']:.2f} dB"
            )
            
            if epoch % self.config.checkpointInterval == 0:
                self.evaluateModel(epoch)
                if self.config.enableCheckpoints:
                    self.checkpointManager.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        loss=valMetrics['loss'],
                        trainLosses=self.trainLosses,
                        valLosses=self.valLosses
                    )

        self.saveResults()


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
