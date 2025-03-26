import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob
import os
import random

class ImageDataset(Dataset):
    def __init__(self, sourceDir, transform=None, mode="L"):
        self.images = sorted(glob.glob(os.path.join(sourceDir, "*")))
        self.mode = mode  # "L" or "RGB"
        self.baseTransform = transforms.Compose([transforms.ToTensor()])
        self.augmentTransform = transform if transform else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        cleanImg = Image.open(self.images[idx]).convert(self.mode)
        cleanImg = self.baseTransform(cleanImg)

        if self.augmentTransform and random.random() > 0.5:
            cleanImg = self.augmentTransform(cleanImg)

        noisyImg = self.addRandomNoise(cleanImg)

        return noisyImg, cleanImg

    def addRandomNoise(self, image):
        """
        Add random noise to the image with random parameters
        """
        # Randomly select noise type
        noiseType = random.choice(["gaussian", "salt_pepper", "poisson", "speckle"])
        noisyImg = image.clone()

        if noiseType == "gaussian":
            # Random std between 0.02 and 0.3
            std = random.uniform(0.02, 0.3)
            noise = torch.randn_like(image) * std
            noisyImg = image + noise

        elif noiseType == "salt_pepper":
            # Random amount between 0.01 and 0.2
            amount = random.uniform(0.01, 0.2)
            # Random salt vs pepper ratio
            s_vs_p = random.uniform(0.3, 0.7)

            # Create salt and pepper noise
            noisyImg = image.clone()
            numSalt = int(amount * image.numel() * s_vs_p)
            saltCoords = [torch.randint(0, i, (numSalt,)) for i in image.shape]
            noisyImg[tuple(saltCoords)] = 1

            numPepper = int(amount * image.numel() * (1.0 - s_vs_p))
            pepperCoords = [torch.randint(0, i, (numPepper,)) for i in image.shape]
            noisyImg[tuple(pepperCoords)] = 0

        elif noiseType == "poisson":
            # Random lambda between 5 and 30
            lam = random.uniform(5, 30)
            noisyImg = torch.poisson(image * lam) / lam

        elif noiseType == "speckle":
            # Random std between 0.05 and 0.3
            std = random.uniform(0.05, 0.3)
            noise = torch.randn_like(image) * std
            noisyImg = image + image * noise

        # Clip to valid range
        return torch.clamp(noisyImg, 0, 1)


def getDataLoader(
    cleanDir,
    batchSize=16,
    numWorkers=4,
    shuffle=True,
    mode="L",
    pinMemory=True,
    augment=True,
):
    augmentTransform = None
    if augment:
        augmentTransform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ]
        )

    dataset = ImageDataset(sourceDir=cleanDir, mode=mode, transform=augmentTransform)

    dataLoader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=shuffle,
        num_workers=numWorkers,
        pin_memory=pinMemory,
    )

    return dataLoader
