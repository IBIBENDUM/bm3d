import optuna
from train import ModelTrainer
import torch.optim as optim
import torch.nn as nn

class OptunaModelTrainer(ModelTrainer):
    def __init__(self, study=None):
        super().__init__()
        self.study = study

    def suggestHyperparameters(self, trial):
        """Suggest hyperparameters for the Optuna study."""
        # Suggest hyperparameters for batch size and learning rate
        batchSize = trial.suggest_int("batchSize", 8, 64, step=8)
        learningRate = trial.suggest_loguniform("learningRate", 1e-5, 1e-2)
        
        # Scheduler hyperparameters
        stepSize = trial.suggest_int("stepsize", 10, 50, step=5)
        gamma = trial.suggest_loguniform("gamma", 0.1, 0.9)

        # Suggest loss function
        lossFunction = trial.suggest_categorical("loss", ["l1", "l2", "smooth_l1", "ssim"])

        # Update config with suggested values
        self.config.train["batchSize"] = batchSize
        self.config.optimizer["learningRate"] = learningRate
        self.config.optimizer["scheduler"]["stepsize"] = stepSize
        self.config.optimizer["scheduler"]["gamma"] = gamma
        self.config.optimizer["loss"] = lossFunction  # Save loss function choice

        # Use fixed optimizer, e.g., Adam
        self.optimizer = optim.Adam(self.model.parameters(), lr=learningRate)

        # Set the loss function based on the suggestion
        self.setupLossFunction()

    def objective(self, trial):
        """Define the objective function for Optuna."""
        # Suggest hyperparameters
        self.suggestHyperparameters(trial)
        
        # Reinitialize model and optimizer with suggested hyperparameters
        self.setupModel()

        # Training loop
        epochMetrics = []
        for epoch in range(self.config.train["epochs"]):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.config.train['epochs']}")
            
            # Training phase
            trainMetrics = self.runEpoch(self.trainLoader, isTraining=True)
            self.updateMetrics("train", trainMetrics)

            # Validation phase
            valMetrics = self.runEpoch(self.valLoader, isTraining=False)
            self.updateMetrics("val", valMetrics)
            
            epochMetrics.append(valMetrics["loss"])

            # Log validation loss for Optuna optimization
            self.logger.info(f"Validation Loss: {valMetrics['loss']:.4f}")
        
        # Return the validation loss as the objective value for optimization
        return epochMetrics[-1]

    def optimize(self, nTrials=10):
        """Optimize the hyperparameters using Optuna."""
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=nTrials)

        # Best hyperparameters found by Optuna
        bestTrial = study.best_trial
        self.logger.info(f"Best trial: {bestTrial.number}")
        self.logger.info(f"  Value: {bestTrial.value}")
        for key, value in bestTrial.params.items():
            self.logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    # Create Optuna study and optimize hyperparameters
    optunaTrainer = OptunaModelTrainer()
    optunaTrainer.optimize(nTrials=50)
