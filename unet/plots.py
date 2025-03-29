import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import catppuccin

def setGraphStyle():
    plt.style.use("latte")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'

def saveExamples(noisy, denoised, clean, epoch, outputDir="epoch_outputs", numExamples=3):
    Path(outputDir).mkdir(parents=True, exist_ok=True)

    setGraphStyle()

    noisy = noisy[:numExamples].cpu().clamp(0, 1)
    denoised = denoised[:numExamples].cpu().clamp(0, 1)
    clean = clean[:numExamples].cpu().clamp(0, 1)
    
    plt.figure(figsize=(15, 5))
    
    for i in range(numExamples):
        plt.subplot(3, numExamples, i+1)
        plt.imshow(noisy[i].squeeze(), cmap='gray')
        plt.title("Noisy")
        plt.axis('off')
        
        plt.subplot(3, numExamples, i+numExamples+1)
        plt.imshow(denoised[i].squeeze(), cmap='gray')
        plt.title("Denoised")
        plt.axis('off')
        
        plt.subplot(3, numExamples, i+2*numExamples+1)
        plt.imshow(clean[i].squeeze(), cmap='gray')
        plt.title("Clean")
        plt.axis('off')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{outputDir}/epoch_{epoch+1}_{timestamp}.png"
    plt.savefig(filename)
    plt.close()

def saveLosses(trainLosses, valLosses, outputDir="results"):
    outputPath = Path(outputDir)
    outputPath.mkdir(parents=True, exist_ok=True)

    setGraphStyle()
    
    plt.figure(figsize=(10, 5))
    plt.plot(trainLosses, label='Train Loss')
    plt.plot(valLosses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss')
    plt.legend()
    plt.grid()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = outputPath / f"losses_{timestamp}.png"
    plt.savefig(filename)
    plt.close()
