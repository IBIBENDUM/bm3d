import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random
from skimage.util import random_noise

class ImageDataset(Dataset):
    def __init__(self, sourceDir, transform=None, mode="L"):
        self.sourcePath = Path(sourceDir)
        self.images = sorted(self.sourcePath.glob("*"))
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

        # Apply noise in range from 15 to 35 PSNR
        match noiseType:
            case "gaussian":
                # Random std between 50 and 5
                noiseLevel = random.uniform(5, 50)
                noiseTransform = transforms.GaussianNoise(
                    mean=0.0, sigma=noiseLevel / 255, clip=True
                )
                noisyImg = noiseTransform(noisyImg)

            case "salt_pepper":
                # Random amount between 0.1 and 0.001
                noiseLevel = random.uniform(0.001, 0.1)
                noisyImg = torch.tensor(
                    random_noise(noisyImg.numpy(), mode="salt", amount=noiseLevel)
                )

            case "poisson":
                # Random lambda between 20 and 1500
                noiseLevel = random.uniform(20, 1500)
                noisyImg = torch.poisson(image * noiseLevel) / noiseLevel

            case "speckle":
                # Random std between 0.05 and 0.3
                noiseLevel = random.uniform(30, 0.2)
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
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9,1.1)),
                transforms.RandomPerspective(distortion_scale=0.5)
            ]
        )

    dataset = ImageDataset(sourceDir=sourceDir, mode=mode, transform=augmentTransform)

    dataLoader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=shuffle,
        num_workers=numWorkers,
        pin_memory=pinMemory,
    )

    return dataLoader
