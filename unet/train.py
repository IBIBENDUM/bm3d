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
from config_manager import ConfigManager
from dataloader import getDataLoader
from logger import setupLogger
from plots import (
    plotAndSaveExamples,
    plotAndSaveData,
    saveDataToCSV,
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

        if self.config.fixSeed:
            self.setRandomSeed(self.config.seed)

        self.model = UNet().to(self.config.device)
        self.criterion = self.initLossFunction()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.learningRate
        )

        self.initDataloaders()
        self.initMetrics()

    def setRandomSeed(self, seed):
        self.logger.info(f"Setting random seed to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def initMetrics(self):
        self.metrics = {
            "train": {
                "loss": [],
                "psnrDiff": [],
                "psnr": [],
                "ssim": [],
                "vif": [],
                "fsim": [],
            },
            "val": {
                "loss": [],
                "psnrDiff": [],
                "psnr": [],
                "ssim": [],
                "vif": [],
                "fsim": [],
            },
        }

    def calculateBatchMetrics(self, noisy, clean, outputs):
        outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())

        noisyPsnr = piq.psnr(noisy, clean)
        denoisedPsnr = piq.psnr(outputs, clean)

        return {
            "psnrDiff": (denoisedPsnr - noisyPsnr).item(),
            "psnr": denoisedPsnr.item(),
            "ssim": piq.ssim(outputs, clean).item(),
            "vif": piq.vif_p(outputs, clean).item(),
            "fsim": piq.fsim(outputs, clean).item(),
        }

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
            augment=True,
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

        epochMetrics = {
            k: 0.0 for k in ["loss", "psnr_imp", "psnr", "ssim", "vif", "fsim"]
        }

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

                batchMetrics = self.calculateBatchMetrics(noisy, clean, outputs)

                for k in epochMetrics:
                    epochMetrics[k] += batchMetrics[k] * noisy.size(0)

        epochMetrics = {k: v / len(dataLoader.dataset) for k, v in epochMetrics.items()}
        phaseMetrics = self.metrics["train" if isTraining else "val"]
        for k in epochMetrics:
            phaseMetrics[k].append(epochMetrics[k])

        return epochMetrics

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
        
        for metric in self.metrics["train"].keys():
            plotAndSaveData(
                yValues=[self.metrics["train"][metric],
                         self.metrics["val"][metric]],
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
                    for epoch, (train, val) in enumerate(
                        zip(self.metrics["train"][metric], self.metrics["val"][metric])
                    )
                ],
                headers=["epoch", f"train_{metric}", f"val_{metric}"],
                outputDir=str(self.outputDir),
                filename=f"{metric}.csv",
            )

        self.logger.info(f"All raw metrics saved to {self.outputDir}")


    def loadCheckpoint(self):
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


    def train(self):
        startEpoch = self.loadCheckpoint() if self.config.enableCheckpoints else 0

        for epoch in range(startEpoch, self.config["epochs"]):
            self.logger.info(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            self.runEpoch(self.trainLoader, isTraining=True)
            self.runEpoch(self.valLoader, isTraining=False)
            
            for metric in self.metrics["train"].keys():
                self.logger.info(f"{metric}: {self.metrics["train"][metric]:.4f} (train) | "
                                 f" {self.metrics["val"][metric]:.4f} (val)")

            if (epoch + 1) % self.config.checkpointInterval == 0:
                self.evaluateModel(epoch)
                self.saveResults()
                if self.config.enableCheckpoints:
                    self.logger.debug(f"Saving checkpoint at epoch {epoch+1}")
                    self.checkpointManager.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics=self.metrics,
                    )

        self.logger.info(f"Final model and plots saved to {self.outputDir}")
        self.saveResults()


if __name__ == "__main__":
    trainer = ModelTrainer()
    # trainer.train()
