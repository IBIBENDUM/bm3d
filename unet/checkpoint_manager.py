import torch
from datetime import datetime
from pathlib import Path
import shutil

class CheckpointManager:
    def __init__(self, basePath=None):
        self.checkpointDir = self._getCheckpointDir(basePath)
        self.checkpointDir.mkdir(parents=True, exist_ok=True)

    def _getCheckpointDir(self, basePath):
        if basePath:
            return Path(basePath)
        if Path("/content/drive/My Drive").exists():
            return Path("/content/drive/My Drive/unet/checkpoints")
        return Path("checkpoints")

    def _getFilePaths(self, fileName, epoch=1):
        mainPath = self.checkpointDir / fileName
        backupPath = mainPath.with_suffix(f".epoch{epoch}.bak")
        return mainPath, backupPath

    def _createMetadata(self):
        return {
            'saveTime': datetime.now().isoformat(),
            'pytorchVersion': torch.__version__
        }

    def save(
        self,
        model,
        optimizer,
        epoch,
        metrics,
        fileName="modelCheckpoint.pth",
    ):
        mainPath, backupPath = self._getFilePaths(fileName, epoch)
        
        checkpoint = {
            'epoch': epoch,
            'modelStateDict': model.state_dict(),
            'optimizerStateDict': optimizer.state_dict(),
            'metrics': metrics,
            'metadata': self._createMetadata()
        }
        torch.save(checkpoint, mainPath)
        shutil.copy2(mainPath, backupPath)

    def load(self, model, optimizer, fileName="modelCheckpoint.pth"):
        mainPath, _ = self._getFilePaths(fileName)
        
        try:
            checkpointPath = mainPath
            checkpoint = torch.load(checkpointPath, weights_only=False)
            
            model.load_state_dict(checkpoint['modelStateDict'])
            optimizer.load_state_dict(checkpoint['optimizerStateDict'])
            
            return {
                "model": model,
                "optimizer": optimizer,
                "epoch": checkpoint["epoch"],
                "metrics": checkpoint["metrics"],
                "metadata": checkpoint["metadata"]
            }
        except FileNotFoundError:
            raise FileNotFoundError("No checkpoint or backup found")
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint: {str(e)}")
