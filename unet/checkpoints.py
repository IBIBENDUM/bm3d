import torch
from pathlib import Path

def getCheckpointDir():
    if Path("/content/drive/My Drive").exists():
        checkpointDir = Path("/content/drive/My Drive/unet/checkpoints")
    else:
        checkpointDir = Path("checkpoints")
    
    checkpointDir.mkdir(parents=True, exist_ok=True)

    return checkpointDir

def saveCheckpoint(model, optimizer, epoch, loss, filename="modelCheckpoint.pth"):
    checkpointDir = getCheckpointDir()
    checkpointPath = checkpointDir / filename
    backupPath = checkpointPath.with_suffix(".bak")

    torch.save({
        'epoch': epoch,
        'modelStateDict': model.state_dict(),
        'optimizerStateDict': optimizer.state_dict(),
        'loss': loss
    }, checkpointPath)

    torch.save({
        'epoch': epoch,
        'modelStateDict': model.state_dict(),
        'optimizerStateDict': optimizer.state_dict(),
        'loss': loss
    }, backupPath)

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
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Checkpoint loaded from {checkpointPath}")

    return model, optimizer, epoch, loss
