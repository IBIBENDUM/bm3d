import json
from pathlib import Path
import torch
from dataclasses import dataclass
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
    cleanTrainDir: str
    cleanValDir: str
    extra: Dict[str, Any] = {}

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
        self.config = Config(**self._rawConfig, extra=self._rawConfig)

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
            "enableCheckpoints",
        ]
        for key in required_keys:
            if key not in self._rawConfig:
                raise ValueError(f"Missing required config key: {key}")

    def __getattr__(self, name):
        return getattr(self.config, name, None)

    def __repr__(self):
        return f"ConfigManager(configPath={self._configPath},\n config={self.config})"
