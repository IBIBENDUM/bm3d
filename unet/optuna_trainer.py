import optuna
import os
from pathlib import Path
import pandas as pd
import optuna.visualization as vis
from train import ModelTrainer

class OptunaModelTrainer(ModelTrainer):
    def __init__(self, study=None, saveInterval=5, epochs=30):
        super().__init__() 
        self.study = study
        self.saveInterval = saveInterval 
        self.config.train["epochs"] = epochs

    def suggestHyperparameters(self, trial):
        """Suggest hyperparameters for Optuna"""
        params = {
            "learning_rate": trial.suggest_float("learningRate", 1e-5, 1e-2, log=True),
            "step_size": trial.suggest_int("stepsize", 10, 50, step=5),
            "gamma": trial.suggest_float("gamma", 0.1, 0.9, log=True),
            "loss_function": trial.suggest_categorical("loss", ["l1", "l2", "smooth_l1", "ssim"])
        }

        self.logHyperparameters(trial.number, params)
        self.applyHyperparameters(params)
        self.setupLossFunction()

    def logHyperparameters(self, trialNumber, params):
        self.logger.info(f"Trial {trialNumber + 1} suggested hyperparameters:")
        for key, value in params.items():
            self.logger.info(f"{key}: {value}")

    def applyHyperparameters(self, params):
        self.config.optimizer.update({
            "learningRate": params["learning_rate"],
            "scheduler": {
                "stepsize": params["step_size"],
                "gamma": params["gamma"]
            },
            "loss": params["loss_function"]
        })

    def setupTrialDir(self, trial):
        trialDirName = f"trial_{trial.number + 1}"
        trialOutputDir = Path("optuna_results") / trialDirName
        trialOutputDir.mkdir(parents=True, exist_ok=True)
        self.outputDir = trialOutputDir
        self.logger.info(f"Trial results will be saved to: {self.outputDir}")
        return trialOutputDir

    def saveTrialParameters(self, trial, trialDir):
        """Save the hyperparameters for the trial in a text file"""
        paramsFilePath = trialDir / "parameters.txt"
        with open(paramsFilePath, 'w') as f:
            f.write("Trial Parameters:\n")
            for param, value in trial.params.items():
                f.write(f"{param}: {value}\n")
        self.logger.info(f"Trial parameters saved to {paramsFilePath}")

    def handleEpochEnd(self, epoch, trial):
        """Logging, save results, evaluate"""
        oldLr = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step()  # Update learning rate
        currentLr = self.optimizer.param_groups[0]["lr"]
        if currentLr != oldLr:
            self.logger.info(
                f"Learning rate changed from {oldLr:.2e} to {currentLr:.2e}"
            )

        # Log validation loss for Optuna
        self.logger.info(
            f"Validation Loss for trial {trial.number + 1}: "
            f'{self.metrics["val"]["loss"][-1]:.4f}'
        )

        # Save results every `saveInterval` epochs
        if (epoch + 1) % self.saveInterval == 0:
            self.evaluateModel(epoch)
            self.saveResults()

    def objective(self, trial):
        """Main function for Optuna"""
        self.suggestHyperparameters(trial)
        self.setupModel()

        # Create a folder for the trial
        self.setupTrialDir(trial)

        # Save the hyperparameters used for this trial in a text file inside the trial folder
        self.saveTrialParameters(trial, self.outputDir)

        # Training loop
        for epoch in range(self.config.train["epochs"]):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.config.train['epochs']}")
            
            # Training phase
            trainMetrics = self.runEpoch(self.trainLoader, isTraining=True)
            self.updateMetrics("train", trainMetrics)

            # Validation phase
            valMetrics = self.runEpoch(self.valLoader, isTraining=False)
            self.updateMetrics("val", valMetrics)
            self.handleEpochEnd(epoch, trial)

        # Return validation loss for optimization in Optuna
        finalLoss = self.metrics["val"]["loss"][-1]
        self.logger.info(f"Trial {trial.number + 1} completed with validation loss: {finalLoss:.4f}")
        
        # Save the final model weights at the end of the trial
        self.saveResults()

        return finalLoss

    def optimize(self, nTrials=10):
        """Optimize hyperparameters using Optuna"""
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=nTrials)

        bestTrial = study.best_trial
        self.logger.info(f"Best trial: {bestTrial.number + 1}\n"
                         f"Value: {bestTrial.value}")

        self.logHyperparameters(bestTrial.number, bestTrial.params)

        # Save optimization results
        self.saveOptimizationResults(study)

    def saveOptimizationResults(self, study):
        """Save optimization results and plots"""
        outputDir = "optuna_results"
        os.makedirs(outputDir, exist_ok=True)
        self.logger.info(f"Output directory created: {outputDir}")

        historyDf = pd.DataFrame(
            [(trial.number + 1, trial.value) for trial in study.trials],
            columns=["Trial", "Objective"],
        )
        historyCsvPath = os.path.join(outputDir, "optimization_history.csv")
        historyDf.to_csv(historyCsvPath, index=False)
        self.logger.info(f"Optimization history saved to {historyCsvPath}.")

        # Plot optimization history using Optuna's visualization library
        fig = vis.plot_optimization_history(study)
        historyImgPath = os.path.join(outputDir, "optimization_history.png")
        fig.write_image(historyImgPath)
        self.logger.info(f"Optimization history plot saved to {historyImgPath}.")

        # Plot parameter importance
        fig2 = vis.plot_param_importances(study)
        importanceImgPath = os.path.join(outputDir, "param_importance.png")
        fig2.write_image(importanceImgPath)
        self.logger.info(f"Parameter importance plot saved to {importanceImgPath}.")

if __name__ == "__main__":
    optunaTrainer = OptunaModelTrainer(saveInterval=5, epochs=30)
    optunaTrainer.optimize(nTrials=10)
