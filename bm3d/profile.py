"""
    Contains bm3d properties and predefined profiles
"""

import os
import enum


class PositiveIntDescriptor:
    def __init__(self, name):
        self.name = name

    def __get__(self, obj, objtype):
        return getattr(obj, f"_{self.name}")

    def __set__(self, obj, value):
        if not isinstance(value, int):
            raise ValueError(f"{self.name} must be an integer")
        if value <= 0:
            raise ValueError(f"{self.name} must be a positive integer")
        setattr(obj, f"_{self.name}", value)


class BM3DStages(enum.Enum):
    BASIC_STAGE = 1
    BOTH_STAGES = 2


# TODO: Add profiles fast and normal
# TODO: FIX COPY PASTE
class BM3DProfile:
    __slots__ = (
        "_blockSize",
        "_blockStep",
        "_searchWindow",
        "_distanceThreshold",
        "_filterThreshold",
        "_groupMaxSize",
        "_kaiserShape",
        "_stages",
        "_cores"
    )

    blockSize = PositiveIntDescriptor("blockSize")
    blockStep = PositiveIntDescriptor("blockStep")
    searchWindow = PositiveIntDescriptor("searchWindow")
    distanceThreshold = PositiveIntDescriptor("distanceThreshold")
    groupMaxSize = PositiveIntDescriptor("groupMaxSize")

    @property
    def filterThreshold(self) -> float:
        return self._filterThreshold

    @filterThreshold.setter
    def filterThreshold(self, value: float):
        self._filterThreshold = value

    @property
    def kaiserShape(self) -> float:
        return self._kaiserShape

    @kaiserShape.setter
    def kaiserShape(self, value: float):
        self._kaiserShape = value

    @property
    def stages(self) -> BM3DStages:
        return self._stages

    @stages.setter
    def stages(self, value: BM3DStages):
        self._stages = value

    @property
    def cores(self) -> int:
        return self._cores

    @cores.setter
    def cores(self, value: int):
        if value == 0:
            value = 1
        elif value < 1:
            cpuCount = os.cpu_count()
            value = cpuCount + value + 1 if cpuCount else 1

        self._cores = value

    def __init__(
        self,
        blockSize: int = 16,
        blockStep: int = 8,
        groupMaxSize: int = 16,
        searchWindow: int = 39,
        distanceThreshold: int = 100,
        filterThreshold: float = 3.0,
        kaiserShape: float = 2.0,
        stages: BM3DStages = BM3DStages.BOTH_STAGES,
        cores: int = 1,
    ):
        self._blockSize = blockSize
        self._blockStep = blockStep
        self._groupMaxSize = groupMaxSize
        self._searchWindow = searchWindow
        self._distanceThreshold = distanceThreshold
        self._filterThreshold = filterThreshold
        self._kaiserShape = kaiserShape
        self._stages = stages
        self.cores = cores


    def __repr__(self):
        return (
            f"BM3DProfile(blockSize={self._blockSize}, "
            f"blockStep={self._blockStep}, "
            f"groupMaxSize={self._groupMaxSize}, "
            f"searchWindow={self._searchWindow}, "
            f"distanceThreshold={self._distanceThreshold}, "
            f"filterThreshold={self._filterThreshold}, "
            f"kaiserShape={self._kaiserShape}, "
            f"stages={self._stages})"
            f"cores={self._cores})"
        )
