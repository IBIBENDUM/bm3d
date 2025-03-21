import os
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])
])


class ImageDataset(Dataset):
    def __init__(self, noisyDir, cleanDir, transform=None):
        # self.noisyDir = noisyDir
        # self.cleanDir = cleanDir
        self.noisyImages = sorted(glob.glob(os.path.join(noisyDir, '*')))
        self.cleanImages = sorted(glob.glob(os.path.join(cleanDir, '*')))
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.noisyImages)

    def __getitem__(self, idx):
        noisyImage = Image.open(self.noisyImages[idx]).convert('L')
        cleanImage = Image.open(self.cleanImages[idx]).convert('L')

        noisyImage = self.transform(noisyImage)
        cleanImage = self.transform(cleanImage)

        return noisyImage, cleanImage 


def prepareData(noisyDir, cleanDir, batchSize=16):
    dataset = ImageDataset(noisyDir, cleanDir, transform=transform)
    dataLoader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

    return dataLoader

