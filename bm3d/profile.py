"""
    Contains bm3d properties and predefined profiles
"""

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
class BM3DProfile:
    __slots__ = (
        "_blockSize",
        "_blockStep",
        "_searchWindow",
        "_distanceThreshold",
        "_filterThreshold",
        "_groupMaxSize",
        "_stages",
    )

    blockSize = PositiveIntDescriptor("blockSize")
    blockStep = PositiveIntDescriptor("blockStep")
    searchWindow = PositiveIntDescriptor("searchWindow")
    groupMaxSize = PositiveIntDescriptor("groupMaxSize")
    distanceThreshold = PositiveIntDescriptor("distanceThreshold")

    @property
    def stages(self) -> BM3DStages:
        return self._stages

    @stages.setter
    def stages(self, value: BM3DStages):
        self._stages = value

    @property
    def filterThreshold(self) -> float:
        return self._filterThreshold

    @filterThreshold.setter
    def filterThreshold(self, value: float):
        self._filterThreshold = value

    def __init__(
        self,
        blockSize: int = 16,
        blockStep: int = 8,
        groupMaxSize: int = 16,
        searchWindow: int = 39,
        distanceThreshold: int = 100,
        filterThreshold: float = 3.00,
        stages: BM3DStages = BM3DStages.BOTH_STAGES,
    ):
        self._blockSize = blockSize
        self._blockStep = blockStep
        self._groupMaxSize = groupMaxSize
        self._searchWindow = searchWindow
        self._distanceThreshold = distanceThreshold
        self._filterThreshold = filterThreshold
        self._stages = stages
