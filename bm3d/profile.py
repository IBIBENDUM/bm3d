"""
    Contains bm3d properties and predefined profiles
"""

class PositiveIntDescriptor:
    def __init__(self, name):
        self.name = name

    def __get__(self, obj, objtype):
        return getattr(obj, f"_{self.name}")

    def __set__(self, obj, value):
        if value <= 0:
            raise ValueError(f"{self.name} must be a positive integer")
        setattr(obj, f"_{self.name}", value)

# TODO: Add profiles fast and normal
class BM3DProfile:
    __slots__ = ("_blockSize", "_distanceThreshold", "_filterThreshold", "_groupMaxSize")

    blockSize = PositiveIntDescriptor("blockSize")
    groupMaxSize = PositiveIntDescriptor("groupMaxSize")
    distanceThreshold = PositiveIntDescriptor("distanceThreshold")
    filterThreshold = PositiveIntDescriptor("filterThreshold")

    def __init__(self, blockSize: int=16, distanceThreshold: int=500, filterThreshold: float=10.00,
                 groupMaxSize: int = 16):
        self._blockSize = blockSize
        self._groupMaxSize = groupMaxSize
        self._distanceThreshold = distanceThreshold
        self._filterThreshold = filterThreshold

