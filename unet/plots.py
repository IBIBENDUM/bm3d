import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import catppuccin
import csv
import piq

def setGraphStyle():
    plt.style.use("latte")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'


def plotAndSaveExamples(
    noisy, denoised, clean, epoch, outputDir="epoch_outputs", numExamples=3
):
    Path(outputDir).mkdir(parents=True, exist_ok=True)

    setGraphStyle()

    noisy = noisy[:numExamples].detach().cpu().clamp(0, 1)
    denoised = denoised[:numExamples].detach().cpu().clamp(0, 1)
    clean = clean[:numExamples].detach().cpu().clamp(0, 1)
    
    plt.figure(figsize=(15, 5))
    
    for i in range(numExamples):
        plt.subplot(3, numExamples, i+1)
        plt.imshow(noisy[i].squeeze(), cmap='gray')
        plt.title(f"Noisy\n PSNR: {piq.psnr(noisy[i].unsqueeze(0), clean[i].unsqueeze(0)):.2f} dB")
        plt.axis('off')
        
        plt.subplot(3, numExamples, i+numExamples+1)
        plt.imshow(denoised[i].squeeze(), cmap='gray')
        plt.title(f"Denoised\n PSNR: {piq.psnr(denoised[i].unsqueeze(0), clean[i].unsqueeze(0)):.2f} dB")
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


def plotAndSaveData(yValues, labels, yLabel, xLabel, title, outputDir, filename):
    outputPath = Path(outputDir)
    outputPath.mkdir(parents=True, exist_ok=True)

    setGraphStyle()

    plt.figure(figsize=(10, 5))
    for y, label in zip(yValues, labels):
        plt.plot(y, label=label)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.legend()
    plt.grid()

    filePath = outputPath / filename
    plt.savefig(filePath)
    plt.close()

def saveDataToCSV(data, headers, outputDir, filename):
    outputPath = Path(outputDir)
    outputPath.mkdir(parents=True, exist_ok=True)

    filePath = outputPath / filename

    with open(filePath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        
        for row in data:
            writer.writerow(row)

