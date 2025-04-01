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
from logger import setupLogger
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
        self.logger = setupLogger(self.outputDir)
        self.logger.info(f"Saving results to: {self.outputDir}")

        self.config = ConfigManager().config
        self.logger.info(f"Used config: {self.config}")
        if self.config.enableCheckpoints:
            self.checkpointManager = CheckpointManager(self.config.checkpointDir)
            self.logger.info(f"Checkpoints directory: {self.checkpointManager.checkpointDir}/")


        self.model = UNet().to(self.config.device)
        self.criterion = self.initLossFunction()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.learningRate
        )

        self.initDataloaders()
        self.trainLosses = []
        self.valLosses = []
        self.trainPsnrImps = []
        self.valPsnrImps = []


    def initLossFunction(self):
        match self.config.loss.lower():
            case "l1" | "mae":
                return nn.L1Loss()
            case "l2" | "mse":
                return nn.MSELoss()
            case "smooth_l1":
                return nn.SmoothL1Loss()
            case "ssim":
                return piq.SSIMLoss()
            case _:
                raise ValueError(f"Unknown loss function: {self.config.loss}")

    def setupOutputDirectory(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outputDir = Path("results") / f"run_{timestamp}"
        outputDir.mkdir(parents=True, exist_ok=True)

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
            augment=False,
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
            self.logger.info(f"Resuming from epoch {checkpointData['epoch'] + 1}")
            return checkpointData['epoch'] + 1

        except FileNotFoundError:
            self.logger.info("No checkpoint found. Starting from scratch.")
            return 0

        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            return 0


    def train(self):
        startEpoch = self.loadCheckpoint() if self.config.enableCheckpoints else 0

        for epoch in range(startEpoch, self.config["epochs"]):
            self.logger.info(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            trainMetrics = self.runEpoch(self.trainLoader, isTraining=True)
            valMetrics = self.runEpoch(self.valLoader, isTraining=False)
            
            self.trainLosses.append(trainMetrics['loss'])
            self.trainPsnrImps.append(trainMetrics['psnrImp'])
            self.valLosses.append(valMetrics['loss'])
            self.valPsnrImps.append(valMetrics['psnrImp'])
            
            self.logger.info(
                f"Train Loss: {trainMetrics['loss']:.5f} | "
                f"Val Loss: {valMetrics['loss']:.5f}"
            )
            self.logger.info(
                f"PSNR Improvement: Train: {trainMetrics['psnrImp']:.2f} dB | "
                f"Val: {valMetrics['psnrImp']:.2f} dB"
            )
            
            if epoch+1 % self.config.checkpointInterval == 0:
                self.logger.debug(f"Saving checkpoint at epoch {epoch+1}")
                self.evaluateModel(epoch)
                if self.config.enableCheckpoints:
                    self.checkpointManager.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        logger=self.logger,
                        epoch=epoch,
                        loss=valMetrics['loss'],
                        trainLosses=self.trainLosses,
                        valLosses=self.valLosses
                    )

        self.logger.info(f"Final model and plots saved to {self.outputDir}")
        self.saveResults()


if __name__ == "__main__":
    trainer = ModelTrainer()
    # trainer.train()
