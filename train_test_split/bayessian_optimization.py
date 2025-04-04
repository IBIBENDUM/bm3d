import os
import pickle
import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import bm3d
from bm3d.profile import BM3DProfile
from dataloader import getDataLoader
from tqdm import tqdm


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, trainX, trainY, likelihood):
        super().__init__(trainX, trainY, likelihood)
        self.meanModule = gpytorch.means.ConstantMean()
        self.covarModule = gpytorch.kernels.RBFKernel()

    def forward(self, x):
        meanX = self.meanModule(x)
        covarX = self.covarModule(x)
        return gpytorch.distributions.MultivariateNormal(meanX, covarX)


class BayesianOptimizer:
    def __init__(self, dataloader, bounds, checkpointDir="checkpoints", saveEvery=5):
        self.dataloader = dataloader
        self.bounds = bounds
        self.checkpointDir = checkpointDir
        self.saveEvery = saveEvery

        self.trainX = self._initializeTrainData()
        self.trainY = self._evaluateModel(self.trainX)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = GPModel(self.trainX, self.trainY, self.likelihood)

        self._loadCheckpoints()

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()}
        ], lr=0.1)

    def _initializeTrainData(self):
        return torch.rand(20, 2) * torch.tensor([
            self.bounds['distanceThreshold'][1] - self.bounds['distanceThreshold'][0],
            self.bounds['filterThreshold'][1] - self.bounds['filterThreshold'][0]
        ]) + torch.tensor([
            self.bounds['distanceThreshold'][0],
            self.bounds['filterThreshold'][0]
        ])

    def _evaluateModel(self, x):
        psnrs = []
        for params in x:
            paramDict = {
                "distanceThreshold": params[0].item(),
                "filterThreshold": params[1].item(),
            }
            psnrs.append(self._computePSNR(paramDict))
        return torch.tensor(psnrs).unsqueeze(-1)

    def _computePSNR(self, params):
        psnrs = []
        for cleanTensor, _ in self.dataloader:
            clean = cleanTensor.numpy()[0, 0]
            clean = (np.clip(clean, 0, 1) * 255).astype(np.uint8)
            noisy = bm3d.addNoise(clean, 25)

            profile = BM3DProfile(
                distanceThreshold=params["distanceThreshold"],
                filterThreshold=params["filterThreshold"],
            )
            denoised = bm3d.bm3d(noisy, 25, profile)
            psnr = bm3d.calculatePSNR(clean, denoised)
            psnrs.append(psnr)
        return np.mean(psnrs)

    def _loadCheckpoints(self):
        modelPath = os.path.join(self.checkpointDir, "gpModelLatest.pkl")
        likelihoodPath = os.path.join(self.checkpointDir, "gpLikelihoodLatest.pkl")

        if os.path.exists(modelPath) and os.path.exists(likelihoodPath):
            print("Загрузка чекпоинтов...")
            with open(modelPath, 'rb') as f:
                self.model = pickle.load(f)
            with open(likelihoodPath, 'rb') as f:
                self.likelihood = pickle.load(f)

    def _saveCheckpoints(self):
        if not os.path.exists(self.checkpointDir):
            os.makedirs(self.checkpointDir)

        with open(os.path.join(self.checkpointDir, "gpModelLatest.pkl"), 'wb') as f:
            pickle.dump(self.model, f)
        with open(os.path.join(self.checkpointDir, "gpLikelihoodLatest.pkl"), 'wb') as f:
            pickle.dump(self.likelihood, f)

    def _savePlot(self):
        self.model.eval()
        with torch.no_grad():
            testX = torch.rand(100, 2) * torch.tensor([
                self.bounds['distanceThreshold'][1] - self.bounds['distanceThreshold'][0],
                self.bounds['filterThreshold'][1] - self.bounds['filterThreshold'][0]
            ]) + torch.tensor([
                self.bounds['distanceThreshold'][0],
                self.bounds['filterThreshold'][0]
            ])
            testY = self.model(testX).mean.numpy()

        plt.figure(figsize=(8, 6))
        plt.scatter(testX[:, 0].numpy(), testX[:, 1].numpy(), c=testY, cmap='viridis')
        plt.colorbar(label='Predicted PSNR')
        plt.xlabel('Distance Threshold')
        plt.ylabel('Filter Threshold')
        plt.title('PSNR Optimization Progress')
        plt.savefig("psnrPlotLatest.png")
        plt.close()

    def optimize(self, nIter=20):
        self.model.train()
        self.likelihood.train()

        for i in tqdm(range(nIter), desc="Bayesian Optimization Progress"):
            self.optimizer.zero_grad()
            output = self.model(self.trainX)
            loss = -output.log_marginal_likelihood(output)
            loss.backward()
            self.optimizer.step()

            if (i + 1) % self.saveEvery == 0:
                self.model.eval()
                self.likelihood.eval()
                self._saveCheckpoints()
                self._savePlot()


if __name__ == "__main__":
    dataloader = getDataLoader(
        sourceDir="dataset/split_dataset/train",
        numWorkers=3,
        augment=False,
        shuffle=False,
    )

    bounds = {
        "distanceThreshold": (140, 200),
        "filterThreshold": (1.0, 5.0),
    }

    optimizer = BayesianOptimizer(dataloader, bounds, saveEvery=5)
    optimizer.optimize(nIter=20)

