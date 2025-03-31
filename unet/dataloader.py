import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import GaussianNoise
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random
from skimage.util import random_noise

class ImageDataset(Dataset):
    # Noise in range from 15 to 35 PSNR
    noiseParams = {
        "gaussian": (5, 50),
        "salt_pepper": (0.001, 0.1),
        "poisson": (20, 1500),
        "speckle": (0.05, 0.3),
    }

    def __init__(
        self,
        sourceDir,
        baseTransform=transforms.Compose([transforms.ToTensor()]),
        augmentTransform=None,
        mode="L",
        augmentProb=0.5,
    ):
        self.sourcePath = Path(sourceDir)
        self.images = sorted(self.sourcePath.glob("*"))
        self.mode = mode  # "L" or "RGB"
        self.baseTransform = baseTransform
        self.augmentTransform = augmentTransform if augmentTransform else None
        self.augmentProb = augmentProb

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        cleanImg = Image.open(self.images[idx]).convert(self.mode)
        cleanImg = self.baseTransform(cleanImg)

        if self.augmentTransform and random.random() < self.augmentProb:
            cleanImg = self.augmentTransform(cleanImg)

        noisyImg = self.addRandomNoise(cleanImg)

        return noisyImg, cleanImg

    def addRandomNoise(self, image):
        """
        Add random noise to the image with random parameters
        """
        # Randomly select noise type
        noiseType = random.choice(list(ImageDataset.noiseParams.keys()))
        noisyImg = image.clone()
        minVal, maxVal = ImageDataset.noiseParams[noiseType]
        noiseLevel = random.uniform(minVal, maxVal)

        match noiseType:
            case "gaussian":
                noiseTransform = GaussianNoise(
                    mean=0.0, sigma=noiseLevel / 255, clip=True
                )
                noisyImg = noiseTransform(noisyImg)

            case "salt_pepper":
                noisyImg = torch.tensor(
                    random_noise(noisyImg.numpy(), mode="salt", amount=noiseLevel)
                )

            case "poisson":
                noisyImg = torch.poisson(image * noiseLevel) / noiseLevel

            case "speckle":
                noisyImg = torch.tensor(
                    random_noise(image.numpy(), mode="speckle", var=noiseLevel / 255)
                )

        # Clip to valid range
        return torch.clamp(noisyImg, 0, 1).float()


def getDataLoader(
    sourceDir,
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
                transforms.RandomRotation(90),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(1, 1.25)
                ),
            ]
        )

    dataset = ImageDataset(
        sourceDir=sourceDir, mode=mode, augmentTransform=augmentTransform
    )

    dataLoader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=shuffle,
        num_workers=numWorkers,
        pin_memory=pinMemory,
    )

    return dataLoader
