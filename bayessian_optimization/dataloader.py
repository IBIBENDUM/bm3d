from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, cleanDir, noisyDir):
        self.cleanImages = sorted(Path(cleanDir).glob("*"))
        self.noisyImages = sorted(Path(noisyDir).glob("*"))
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        return len(self.cleanImages)

    def __getitem__(self, idx):
        cleanImage = Image.open(self.cleanImages[idx]).convert('L')
        cleanImage = self.transform(cleanImage)
        
        noisyImage = Image.open(self.noisyImages[idx]).convert('L')
        noisyImage = self.transform(noisyImage)

        return noisyImage, cleanImage

def getDataLoader(cleanDir, noisyDir, batchSize=16):
    dataset = ImageDataset(cleanDir, noisyDir)
    dataLoader = DataLoader(dataset, batch_size=batchSize, shuffle=False)

    return dataLoader
