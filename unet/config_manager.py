import json
from pathlib import Path
import torch
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class Config:
    paths: Dict[str, str] = field(default_factory=dict)
    train: Dict[str, Any] = field(default_factory=dict)
    optimizer: Dict[str, Any] = field(default_factory=dict)
    checkpoints: Dict[str, Any] = field(default_factory=dict)
    random: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        data = self._load_json("config.json")
        self.paths.update(data.get("paths", {}))
        self.train.update(data.get("train", {}))
        self.optimizer.update(data.get("optimizer", {}))
        self.checkpoints.update(data.get("checkpoints", {}))
        self.random.update(data.get("random", {}))
        self.extra.update({k: v for k, v in data.items() if k not in {"paths", "train", "optimizer", "checkpoints", "random"}})

        if self.device == "cuda" and not torch.cuda.is_available():
            self.train["device"] = "cpu"

    def __getattr__(self, key):
        for section in (self.paths, self.train, self.optimizer, self.checkpoints, self.random, self.extra):
            if key in section:
                return section[key]
        raise AttributeError(f"Config has no attribute '{key}'")

    @staticmethod
    def _load_json(config_path: str):
        with open(config_path, "r") as file:
            return json.load(file)

    def __repr__(self):
        # Генерация красивого вывода, как в JSON
        return json.dumps(self.to_dict(), indent=4)

    def to_dict(self):
        # Преобразование в словарь с игнорированием атрибутов класса
        return {
            "paths": self.paths,
            "train": self.train,
            "optimizer": self.optimizer,
            "checkpoints": self.checkpoints,
            "random": self.random,
            "extra": self.extra
        }

# Автоматическая загрузка конфига при создании объекта
config = Config()

# Вывод объекта
print(config)
