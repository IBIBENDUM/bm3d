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

    def _getFilePaths(self, fileName):
        mainPath = self.checkpointDir / fileName
        backupPath = mainPath.with_suffix(".bak")
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
        logger,
        epoch,
        loss,
        trainLosses,
        valLosses,
        fileName="modelCheckpoint.pth",
    ):
        mainPath, backupPath = self._getFilePaths(fileName)
        
        checkpoint = {
            'epoch': epoch,
            'modelStateDict': model.state_dict(),
            'optimizerStateDict': optimizer.state_dict(),
            'loss': loss,
            'trainLosses': trainLosses,
            'valLosses': valLosses,
            'metadata': self._createMetadata()
        }

        torch.save(checkpoint, mainPath)
        shutil.copy2(mainPath, backupPath)
        
        logger.info(f"Checkpoint saved at {mainPath}")
        logger.info(f"Backup saved at {backupPath}")

    def load(self, model, optimizer, fileName="modelCheckpoint.pth"):
        mainPath, backupPath = self._getFilePaths(fileName)
        
        try:
            checkpointPath = mainPath if mainPath.exists() else backupPath
            checkpoint = torch.load(checkpointPath)
            
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
        except FileNotFoundError:
            raise FileNotFoundError("No checkpoint or backup found")
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint: {str(e)}")
