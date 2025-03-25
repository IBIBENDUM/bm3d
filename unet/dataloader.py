from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob
import os


class ImageDataset(Dataset):
    def __init__(self, noisyDir, cleanDir, transform=None, mode="L"):
        self.noisyImages = sorted(glob.glob(os.path.join(noisyDir, "*")))
        self.cleanImages = sorted(glob.glob(os.path.join(cleanDir, "*")))
        assert len(self.noisyImages) == len(
            self.cleanImages
        ), "Number of noisy and clean images doesn't match"

        self.transform = (
            transform if transform else transforms.Compose([transforms.ToTensor()])
        )
        self.mode = mode  # "L" or "RGB"

    def __len__(self):
        return len(self.noisyImages)

    def __getitem__(self, idx):
        noisyImg = Image.open(self.noisyImages[idx]).convert(self.mode)
        cleanImg = Image.open(self.cleanImages[idx]).convert(self.mode)

        if self.transform:
            noisyImg = self.transform(noisyImg)
            cleanImg = self.transform(cleanImg)

        return noisyImg, cleanImg


def getDataLoader(
    noisyDir,
    cleanDir,
    batchSize=16,
    numWorkers=4,
    shuffle=True,
    mode="L",
    pinMemory=True,
):
    dataset = ImageDataset(noisyDir, cleanDir, mode=mode)
    dataLoader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=shuffle,
        num_workers=numWorkers,
        pin_memory=pinMemory,
    )

    return dataLoader
