"""
    Contains bm3d properties and predefined profiles
"""

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

class PositiveFloatDescriptor:
    def __init__(self, name):
        self.name = name

    def __get__(self, obj, objtype):
        return getattr(obj, f"_{self.name}")

    def __set__(self, obj, value):
        if not isinstance(value, float):
            raise ValueError(f"{self.name} must be an float")
        if value <= 0:
            raise ValueError(f"{self.name} must be a positive float")
        setattr(obj, f"_{self.name}", value)

# TODO: Add profiles fast and normal
class BM3DProfile:
    __slots__ = (
        "_blockSize",
        "_blockStep",
        "_distanceThreshold",
        "_filterThreshold",
        "_groupMaxSize",
    )

    blockSize = PositiveIntDescriptor("blockSize")
    blockStep = PositiveIntDescriptor("blockStep")
    distanceThreshold = PositiveIntDescriptor("distanceThreshold")
    filterThreshold = PositiveFloatDescriptor("filterThreshold")
    groupMaxSize = PositiveIntDescriptor("groupMaxSize")

    def __init__(
        self,
        blockSize: int = 16,
        blockStep: int = 8,
        distanceThreshold: int = 100,
        filterThreshold: float = 30.00,
        groupMaxSize: int = 16,
    ):
        self._blockSize = blockSize
        self._blockStep = blockStep
        self._groupMaxSize = groupMaxSize
        self._distanceThreshold = distanceThreshold
        self._filterThreshold = filterThreshold

