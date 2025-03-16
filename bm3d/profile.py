"""
    Contains bm3d properties and predefined profiles
"""

# TODO: Add profiles fast and normal
class BM3DProfile:
    __slots__ = ('_blockSize', '_distanceThreshold')

    def __init__(self, blockSize: int=16, distanceThreshold: int = 500):
        self._blockSize = blockSize
        self._distanceThreshold = distanceThreshold

    @property
    def blockSize(self) -> int:
        return self._blockSize

    @blockSize.setter
    def blockSize(self, value: int):
        if value <= 0:
            raise ValueError("Block size must be a positive integer")
        self._blockSize = value

    @property
    def distanceThreshold(self) -> int:
        return self._distanceThreshold

    @distanceThreshold.setter
    def distanceThreshold(self, value: int):
        if value <= 0:
            raise ValueError("Distance threshold must be a positive integer")
        self._blockSize = value
