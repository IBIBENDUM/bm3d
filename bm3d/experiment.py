"""
Helper functions for experiments
"""

import time
from typing import Callable, Any, Tuple


def measureTime(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure the execution time of a given function.
    """
    startTime = time.perf_counter() 
    result = func(*args, **kwargs)  
    endTime = time.perf_counter()  
    elapsedTime = endTime - startTime

    return result, elapsedTime

