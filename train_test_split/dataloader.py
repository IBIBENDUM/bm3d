import os
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.v2 import GaussianNoise
from PIL import Image

# Преобразования для изображений
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Класс для датасета
class ImageDataset(Dataset):
    def __init__(self, cleanDir, noiseLevel=25):
        self.cleanImages = sorted(glob.glob(os.path.join(cleanDir, '*')))
        self.transform = transforms.Compose([transforms.ToTensor()])
        
        # Создание трансформера для добавления гауссовского шума
        self.noiseTransform = GaussianNoise(
            mean=0.0, sigma=noiseLevel/255, clip=True
        )

    def __len__(self):
        return len(self.cleanImages)

    def __getitem__(self, idx):
        # Загружаем чистое изображение
        cleanImage = Image.open(self.cleanImages[idx]).convert('L')
        
        # Применяем преобразования
        cleanImage = self.transform(cleanImage)
        
        # Добавляем гауссовский шум с помощью noiseTransform
        noisyImage = self.noiseTransform(cleanImage)

        return noisyImage, cleanImage

# Функция для подготовки данных
def getDataLoader(cleanDir, batchSize=16, noiseLevel=25):
    dataset = ImageDataset(cleanDir, noiseLevel=noiseLevel)
    dataLoader = DataLoader(dataset, batch_size=batchSize, shuffle=False)

    return dataLoader
