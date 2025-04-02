from datetime import datetime
from pathlib import Path 

import torch
import torch.nn as nn
import torch.optim as optim
import piq 
import random
import numpy as np
from tqdm import tqdm

from model import UNet
from checkpoint_manager import CheckpointManager
from config_manager import Config
from dataloader import getDataLoader
from logger import setupLogger
from plots import plotAndSaveExamples, plotAndSaveData, saveDataToCSV


class ModelTrainer:
    metricsNames = ["loss", "psnrDiff", "psnr", "ssim", "vif", "fsim"]

    def __init__(self):
        self.outputDir = self.setupOutputDirectory()
        self.logger = setupLogger(self.outputDir)
        self.logger.info(f"Saving results to: {self.outputDir}")

        self.config = Config()
        self.logger.info(f"Used config: {self.config}")

        self.setupModel()

        self.trainLoader, self.valLoader = self.setupDataloaders()
        self.metrics = self.setupMetricsStorage()

    def setupCheckpoints(self):
        """Initialize checkpoint manager if enabled"""
        if self.config.checkpoints["enableCheckpoints"]:
            self.checkpointManager = CheckpointManager(self.config.paths["checkpointDir"])
            self.logger.info(
                f"Checkpoints directory: {self.checkpointManager.checkpointDir}/"
            )

    def setupModel(self):
        """Initialize model, loss, optimizer and checkpoint manager"""
        self.setupCheckpoints()
        self.setupRandomSeed()

        self.model = UNet().to(self.config.train["device"])
        self.criterion = self.setupLossFunction()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.optimizer["learningRate"]
        )
        self.scheduler = self.setupLrScheduler()

    def setupLrScheduler(self):
        """Initialize learning rate scheduler"""
        return optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.optimizer["scheduler"]["stepsize"],
            gamma=self.config.optimizer["scheduler"]["gamma"],
        )

    def setupRandomSeed(self):
        """Set random seed if enabled"""
        if self.config.random["fixSeed"]:
            self.logger.info(f"Setting random seed to {self.config.random["seed"]}")
            random.seed(self.config.random["seed"])
            np.random.seed(self.config.random["seed"])
            torch.manual_seed(self.config.random["seed"])
            torch.cuda.manual_seed(self.config.random["seed"])
            torch.cuda.manual_seed_all(self.config.random["seed"])

    def setupMetricsStorage(self):
        """Initialize metrics storage structure"""
        return {
            phase: {metric: [] for metric in self.metricsNames}
            for phase in ["train", "val"]
        }


    def setupLossFunction(self):
        """Initialize loss function based on config"""
        match self.config.optimizer["loss"].lower():
            case "l1" | "mae":
                return nn.L1Loss()
            case "l2" | "mse":
                return nn.MSELoss()
            case "smooth_l1":
                return nn.SmoothL1Loss()
            case "ssim":
                return piq.SSIMLoss()
            case _:
                raise ValueError(f"Unknown loss function: {self.config.optimizer["loss"]}")

    def setupOutputDirectory(self):
        """Initialize output directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outputDir = Path("results") / f"run_{timestamp}"
        outputDir.mkdir(parents=True, exist_ok=True)

        return outputDir

    def setupDataloaders(self):
        """Initialize data loaders for training and validation"""
        trainLoader = getDataLoader(
            sourceDir=self.config.paths["cleanTrainDir"],
            batchSize=self.config.train["batchSize"],
            numWorkers=self.config.train["numWorkers"],
            augment=True,
            shuffle=True,
        )

        valLoader = getDataLoader(
            sourceDir=self.config.paths["cleanValDir"],
            batchSize=self.config.train["batchSize"],
            numWorkers=self.config.train["numWorkers"],
            augment=False,
            shuffle=False,
        )

        return trainLoader, valLoader

    def evaluateModel(self, epoch):
        """Save example predictions"""
        sampleNoisy, sampleClean = next(iter(self.valLoader))
        sampleNoisy = sampleNoisy.to(self.config.train["device"])
        sampleClean = sampleClean.to(self.config.train["device"])
        
        with torch.inference_mode():
            sampleDenoised = self.model(sampleNoisy)
            plotAndSaveExamples(
                sampleNoisy, 
                sampleDenoised, 
                sampleClean, 
                epoch, 
                str(self.outputDir)
            )

    def calculateBatchMetrics(self, noisy, clean, outputs):
        """Calculate metrics for single batch"""
        outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())

        noisyPsnr = piq.psnr(noisy, clean)
        denoisedPsnr = piq.psnr(outputs, clean)

        return {
            "psnrDiff": (denoisedPsnr - noisyPsnr).item(),
            "psnr": denoisedPsnr.item(),
            "ssim": piq.ssim(outputs, clean)[0].item(),
            "vif": piq.vif_p(outputs, clean).item(),
            "fsim": piq.fsim(outputs, clean).item(),
        }

    def saveMetrics(self):
        """Save metrics to csv files and plots"""
        for metric in self.metricsNames:
            trainValues = self.metrics["train"][metric]
            valValues =  self.metrics["val"][metric]

            plotAndSaveData(
                yValues=[trainValues, valValues],
                labels=[f"Train {metric}", f"Validation {metric}"],
                yLabel=metric,
                xLabel="Epoch",
                title=f"Training and Validation {metric}",
                outputDir=self.outputDir,
                filename=f"{metric}_plot.png",
            )

            saveDataToCSV(
                data=[
                    (epoch + 1, train, val)
                    for epoch, (train, val) in enumerate(zip(trainValues, valValues))
                ],
                headers=["epoch", f"train_{metric}", f"val_{metric}"],
                outputDir=str(self.outputDir),
                filename=f"{metric}.csv",
            )

        self.logger.info(f"All raw metrics saved to {self.outputDir}")

    def loadCheckpoint(self):
        """Load checkpoint if enabled and available"""
        if not self.config.checkpoints["enableCheckpoints"]:
            return 0

        try:
            checkpointData = self.checkpointManager.load(self.model, self.optimizer)
            self.model = checkpointData["model"]
            self.optimizer = checkpointData["optimizer"]
            self.metrics = checkpointData["metrics"]
            self.logger.info(f"Resuming from epoch {checkpointData["epoch"] + 1}")
            return checkpointData["epoch"] + 1

        except FileNotFoundError:
            self.logger.info("No checkpoint found. Starting from scratch.")
            return 0

        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            return 0

    def saveCheckpoint(self, epoch):
        """Save checkpoint"""
        if not self.config.checkpoints["enableCheckpoints"]:
            return

        self.logger.debug(f"Saving checkpoint at epoch {epoch+1}")
        self.checkpointManager.save(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            metrics=self.metrics,
        )

    def shouldSaveCheckpoint(self, epoch):
        """Check if should save checkpoint"""
        return (epoch + 1) % self.config.checkpoints["checkpointInterval"] == 0

    def saveResults(self):
        """Save model weights and metrics"""
        modelSavePath = self.outputDir / "denoising_model.pth"
        torch.save(self.model.state_dict(), modelSavePath)
        self.logger.info(f"Model weights saved to {str(modelSavePath)}")
        self.saveMetrics()

    def updateMetrics(self, stage, epochMetrics):
        """Update metrics storage with new epoch results"""
        for metric in epochMetrics:
            self.metrics[stage][metric].append(epochMetrics[metric])

    def train(self):
        """Main training loop"""
        startEpoch = self.loadCheckpoint()

        for epoch in range(startEpoch, self.config.train["epochs"]):
            self.logger.info(f"\nEpoch {epoch+1}/{self.config.train["epochs"]}")
            
            # Training stage
            trainMetrics = self.runEpoch(self.trainLoader, isTraining=True)
            self.updateMetrics("train", trainMetrics)

            # Validation stage
            valMetrics =self.runEpoch(self.valLoader, isTraining=False)
            self.updateMetrics("val", valMetrics)
            
            # Log metrics
            for metric in self.metricsNames:
                self.logger.info(f"{metric}: {trainMetrics[metric]:.4f} (train) | "
                                 f" {valMetrics[metric]:.4f} (val)")

            # Periodic save data
            if self.shouldSaveCheckpoint:
                self.saveCheckpoint(epoch)
                self.evaluateModel(epoch)
                self.saveResults()

        # Final save
        self.logger.info(f"Final model and plots saved to {self.outputDir}")
        self.saveResults()

    def runEpoch(self, dataLoader, isTraining):
        """Run training or validation epoch"""
        if isTraining:
            self.model.train()
            desc = "Training"
            mode = torch.enable_grad()
        else:
            self.model.eval()
            desc = "Validating"
            mode = torch.inference_mode()

        epochMetrics = {metric: 0.0 for metric in self.metricsNames}

        with mode:
            for noisy, clean in tqdm(dataLoader, desc=desc):
                noisy = noisy.to(self.config.train["device"])
                clean = clean.to(self.config.train["device"])
                outputs = self.model(noisy)
                loss = self.criterion(outputs, clean)

                if isTraining:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                batchMetrics = self.calculateBatchMetrics(noisy, clean, outputs)
                batchMetrics["loss"] = loss.item()

                for metric in epochMetrics:
                    epochMetrics[metric] += batchMetrics[metric] * noisy.size(0)

        epochMetrics = {k: v / len(dataLoader.dataset) for k, v in epochMetrics.items()}
        
        return epochMetrics


if __name__ == "__main__":
    trainer = ModelTrainer()
    # trainer.train()
