import json
from pathlib import Path
import torch
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class Config:
    device: str
    learningRate: float
    epochs: int
    batchSize: int
    numWorkers: int
    enableCheckpoints: bool
    checkpointInterval: int
    checkpointDir: str
    cleanTrainDir: str
    loss: str
    cleanValDir: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

    def __getitem__(self, key):
        return self.extra.get(key, getattr(self, key, None))


class ConfigManager:
    def __init__(self, configPath="config.json"):
        self._configPath = Path(configPath)
        self._rawConfig = self._loadConfig()
        self._validateConfig()
        self.config = self._create_config()

    def _create_config(self):
        main_params = {
            'cleanTrainDir': self._rawConfig['cleanTrainDir'],
            'cleanValDir': self._rawConfig['cleanValDir'],
            'checkpointDir': self._rawConfig['checkpointDir'],
            'batchSize': self._rawConfig['batchSize'],
            'numWorkers': self._rawConfig['numWorkers'],
            'learningRate': self._rawConfig['learningRate'],
            'epochs': self._rawConfig['epochs'],
            'device': self._rawConfig['device'],
            'enableCheckpoints': self._rawConfig['enableCheckpoints'],
            'checkpointInterval': self._rawConfig['checkpointInterval'],
            'loss': self._rawConfig['loss']
        }
        
        extra_params = {
            k: v for k, v in self._rawConfig.items()
            if k not in main_params
        }
        
        return Config(**main_params, extra=extra_params)


    def _loadConfig(self):
        with open(self._configPath, "r") as file:
            return json.load(file)

    def _validateConfig(self):
        required_keys = [
            "device",
            "learningRate",
            "epochs",
            "batchSize",
            "numWorkers",
            "checkpointInterval",
            "cleanTrainDir",
            "cleanValDir",
            "checkpointDir",
            "enableCheckpoints",
            "loss",
        ]
        for key in required_keys:
            if key not in self._rawConfig:
                raise ValueError(f"Missing required config key: {key}")

    def __getattr__(self, name):
        return getattr(self.config, name, None)

    def __repr__(self):
        return f"ConfigManager(configPath={self._configPath},\n config={self.config})"
