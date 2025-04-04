import logging
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from bayes_opt import BayesianOptimization
from skimage.metrics import peak_signal_noise_ratio as psnr
import bm3d
from bm3d.profile import BM3DProfile, BM3DStages
from dataloader import getDataLoader

def setupLogging():
    """Настройка логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("optimization.log"),
            logging.StreamHandler()
        ]
    )

def calculatePsnr(cleanImage, denoisedImage):
    """Вычисление PSNR между очищенным и денойзированным изображением."""
    return psnr(cleanImage, denoisedImage, data_range=255)

def tensorToNumpy(imageTensor):
    """Конвертирование тензора изображения в numpy массив."""
    imageNumpy = imageTensor.squeeze().numpy() * 255.0
    return imageNumpy.astype(np.uint8)

def denoiseImage(noisyImage, noiseVariance, profile):
    """Деноизинг изображения с использованием BM3D."""
    return bm3d.bm3d(noisyImage, noiseVariance, profile)

def processBatch(noisyImages, cleanImages, noiseVariance, profile):
    """Обработка пакета изображений и вычисление суммарного PSNR."""
    totalPsnr = 0.0

    for i in range(noisyImages.shape[0]):
        noisyImageNumpy = tensorToNumpy(noisyImages[i])
        cleanImageNumpy = tensorToNumpy(cleanImages[i])

        denoisedImageNumpy = denoiseImage(noisyImageNumpy, noiseVariance, profile)
        imagePsnr = calculatePsnr(cleanImageNumpy, denoisedImageNumpy)
        totalPsnr += imagePsnr

    return totalPsnr

def processDataset(filterThreshold, distanceThreshold, dataLoader, noiseVariance):
    """Обработка всего набора данных с вычислением среднего PSNR."""
    totalPsnr = 0.0
    profile = BM3DProfile(filterThreshold=filterThreshold,
                          distanceThreshold=int(distanceThreshold),
                          stages=BM3DStages.BASIC_STAGE)

    totalImages = len(dataLoader.dataset)
    for noisyImages, cleanImages in dataLoader:
        batchPsnr = processBatch(
            noisyImages,
            cleanImages,
            noiseVariance,
            profile
        )
        totalPsnr += batchPsnr

    averagePsnr = totalPsnr / totalImages
    return averagePsnr

def plot_gp(optimizer, x, y):
    """Функция для визуализации гауссовского процесса с доверительными интервалами."""
    mu, sigma = optimizer._gp.predict(x, return_std=True)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'r:', label='Целевая функция')
    plt.plot(x, mu, 'b-', label='Аппроксимация GP')
    plt.fill_between(x.flatten(), mu.flatten() - 1.96 * sigma.flatten(), mu.flatten() + 1.96 * sigma.flatten(),
                     color='blue', alpha=0.2, label='95% доверительный интервал')
    plt.scatter(optimizer.X[:, 0], optimizer.Y, c='red', s=50, edgecolors='black', label='Измеренные точки')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Визуализация гауссовского процесса')
    plt.legend()
    plt.show()

def optimizeParameters(dataLoader, noiseVariance):
    """Оптимизация параметров с использованием Bayesian Optimization."""
    paramBounds = {
        'filterThreshold': (0, 10),
        'distanceThreshold': (10, 200)
    }

    # Частичная функция для оптимизации
    optimizeFunction = partial(processDataset, dataLoader=dataLoader, noiseVariance=noiseVariance)

    optimizer = BayesianOptimization(
        f=optimizeFunction,
        pbounds=paramBounds,
        random_state=0
    )

    # Слушатель для записи результатов
    psnrs = []
    xVals = []

    optimizer.set_gp_params(kernel="Matern", alpha=1e-5)  # Установка параметров гауссовского процесса
    optimizer.maximize(init_points=5, n_iter=1)

    # Визуализация после завершения всех итераций
    plot_gp(optimizer, np.array(xVals).reshape(-1, 1), psnrs)

    return optimizer.max

def trainModel(cleanDir, noiseVariance=20):
    """Обучение модели с использованием данных и Bayesian Optimization."""
    logging.info("Loading data...")
    dataLoader = getDataLoader(cleanDir)
    logging.info("Data loaded successfully.")

    bestParams = optimizeParameters(dataLoader, noiseVariance)
    return bestParams

if __name__ == "__main__":
    """Основная функция для запуска обучения и тестирования."""
    setupLogging()

    cleanTrainDir = "dataset/split_dataset/train"

    # Шаги обучения и тестирования
    bestParams = trainModel(cleanTrainDir, noiseVariance=25)

    filterThreshold = bestParams['params']['filterThreshold']
    distanceThreshold = bestParams['params']['distanceThreshold']

    logging.info(f"Testing with filterThreshold={filterThreshold}, "
                 f"distanceThreshold={distanceThreshold}")
