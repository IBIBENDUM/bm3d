import torch
from model import UNet
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np

def loadModel(modelPath, device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(modelPath, map_location=device))
    model.eval()

    return model

def denoiseImage(model, imagePath, device):
    image = Image.open(imagePath).convert("L")
    inputTensor = ToTensor()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputTensor = model(inputTensor)

    outputTensor = outputTensor.squeeze(0).cpu().numpy() 
    outputTensor = np.clip(outputTensor, 0, 1) 
    outputTensor = (outputTensor * 255).astype('uint8')
    denoisedImage = Image.fromarray(outputTensor.squeeze()) 
    
    return denoisedImage

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = loadModel("weights/weights1.pth", device)

    noisyImagePath = "cameraman256.jpg"
    denoisedImage = denoiseImage(model, noisyImagePath, device)
    denoisedImage.save("denoised_image.png")

