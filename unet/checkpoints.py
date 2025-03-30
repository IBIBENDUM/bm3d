import torch
from datetime import datetime
from pathlib import Path
import shutil

def getCheckpointDir():
    if Path("/content/drive/My Drive").exists():
        checkpointDir = Path("/content/drive/My Drive/unet/checkpoints")
    else:
        checkpointDir = Path("checkpoints")
    
    checkpointDir.mkdir(parents=True, exist_ok=True)

    return checkpointDir


def saveCheckpoint(model, optimizer, epoch, loss, trainLosses, valLosses, filename="modelCheckpoint.pth"):
    checkpointDir = getCheckpointDir()
    checkpointPath = checkpointDir / filename
    backupPath = checkpointPath.with_suffix(".bak")

    checkpoint = {
        'epoch': epoch,
        'modelStateDict': model.state_dict(),
        'optimizerStateDict': optimizer.state_dict(),
        'loss': loss,
        'trainLosses': trainLosses,
        'valLosses': valLosses,
        'metadata': {
            'saveTime': datetime.now().isoformat(),
            'pytorchVersion': torch.__version__
        }
    }

    torch.save(checkpoint, checkpointPath)
    shutil.copy2(checkpointPath, backupPath)

    print(f"Checkpoint saved at {checkpointPath}")
    print(f"Backup checkpoint saved at {backupPath}")


def loadCheckpoint(model, optimizer, filename="modelCheckpoint.pth"):
    checkpointDir = getCheckpointDir()
    checkpointPath = checkpointDir / filename
    backupPath = checkpointPath.with_suffix(".bak")

    if checkpointPath.exists():
        checkpoint = torch.load(checkpointPath)
    elif backupPath.exists():
        checkpoint = torch.load(backupPath)
    else:
        raise FileNotFoundError("Checkpoint not found.")

    model.load_state_dict(checkpoint['modelStateDict'])
    optimizer.load_state_dict(checkpoint['optimizerStateDict'])
    
    return {
        'model': model,
        'optimizer': optimizer,
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss'],
        'trainLosses': checkpoint.get('trainLosses', []),
        'valLosses': checkpoint.get('valLosses', []),
        'metadata': checkpoint.get('metadata', {})
    }
