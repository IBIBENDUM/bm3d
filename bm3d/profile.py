"""
    Contains bm3d properties and predefined profiles
"""

import enum
import os
from dataclasses import dataclass
# from numba import int32, float32
# from numba.experimental import jitclass

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

    # def toNumbaProfile(self):
    #     """
    #     Копирует значения параметров из BM3DProfile в NumbaBM3DProfile.
    #     """
    #     return NumbaBM3DProfile(
    #         blockSize=self.blockSize,
    #         blockStep=self.blockStep,
    #         searchWindow=self.searchWindow,
    #         distanceThreshold=self.distanceThreshold,
    #         groupMaxSize=self.groupMaxSize,
    #         filterThreshold=self.filterThreshold,
    #         kaiserShape=self.kaiserShape,
    #         cores=self.cores,
    #         stages=self.stages,  # Мы передаем значение перечисления, чтобы оно было совместимо с numba
    #     )
    #

# # Структура для работы с numba
# spec = [
#     ("blockSize", int32),
#     ("blockStep", int32),
#     ("searchWindow", int32),
#     ("distanceThreshold", int32),
#     ("groupMaxSize", int32),
#     ("filterThreshold", float32),
#     ("kaiserShape", float32),
#     ("cores", int32),
#     ("stages", int32),  # Индекс для перечисления (enum)
# ]
#
#
# @jitclass(spec)
# class NumbaBM3DProfile:
#     def __init__(
#         self,
#         blockSize=16,
#         blockStep=8,
#         searchWindow=39,
#         distanceThreshold=100,
#         groupMaxSize=16,
#         filterThreshold=3.0,
#         kaiserShape=2.0,
#         cores=1,
#         stages=BM3DStages.BOTH_STAGES,
#     ):
#         self.blockSize = blockSize
#         self.blockStep = blockStep
#         self.searchWindow = searchWindow
#         self.distanceThreshold = distanceThreshold
#         self.groupMaxSize = groupMaxSize
#         self.filterThreshold = filterThreshold
#         self.kaiserShape = kaiserShape
#         self.cores = cores
#         self.stages = stages.value  # Используем value, так как numba работает с int32
