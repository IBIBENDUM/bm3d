import numpy as np

def addNoise(image: np.ndarray, noiseVariance: float) -> np.ndarray:
    """
    Add Additive Zero Mean White Gaussian Noise (AWGN) to image
    Args:
        image: image for noise applying
        noiseVariance: noise variance (sigma)
    Return:
        Image with noise
    """
    gaussianNoise = np.random.normal(0, noiseVariance, image.shape)
    gaussianNoise = gaussianNoise.reshape(image.shape)
    noisyImage = image + gaussianNoise
    noisyImage = np.clip(noisyImage, 0, 255)
    noisyImage = noisyImage.astype(np.uint8)

    return noisyImage

