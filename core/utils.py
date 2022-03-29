"""

"""

import os
import psutil
import time
from functools import wraps
import cProfile
import pstats
from icecream import ic


def get_mem_usage(precision: int = 2) -> float:
    """Returns current process' memory usage in MB."""
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / 1000000, precision)


class Timer:
    """This context manager allows timing blocks of code."""
    def __init__(self):
        self._timer = None
        self._elapsed = None
    
    def __enter__(self) -> None:
        self._timer = time.time()
        return self

    def __exit__(self, *_: list) -> None:
        self._elapsed = time.time() - self._timer

    def __float__(self):
        return self._elapsed


def profile(sort_by='cumulative', lines_to_print=None, strip_dirs=False):
    """Decorator to profile functions."""
    # TODO fix call signature
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()

            ps = pstats.Stats(pr)
            if strip_dirs: ps.strip_dirs()
            if isinstance(sort_by, (tuple, list)): ps.sort_stats(*sort_by)
            else: ps.sort_stats(sort_by)
            ps.print_stats(lines_to_print)

            return retval

        return wrapper
    return inner