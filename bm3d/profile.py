"""
    Contains bm3d properties and predefined profiles
"""

import enum
import os
from dataclasses import dataclass

class BM3DStages(enum.Enum):
    BASIC_STAGE = 1
    BOTH_STAGES = 2

@dataclass
class BM3DProfile:
    blockSize: int = 16
    blockStep: int = 8
    searchWindow: int = 39
    distanceThreshold: int = 100
    groupMaxSize: int = 16
    filterThreshold: float = 3.0
    kaiserShape: float = 2.0
    cores: int = 1
    stages: BM3DStages = BM3DStages.BOTH_STAGES

    def __post_init__(self):
        if self.cores < 0:
            maxCores = os.cpu_count() or 1
            self.cores = maxCores + self.cores
