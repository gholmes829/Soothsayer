"""

"""

import os.path as osp, os
from threading import current_thread
from typing import Callable
import validators
import psutil
import time
from time import sleep
from icecream import ic


class DefaultDict(dict):
    def __init__(self, default_factory: Callable, **kwargs: dict):
        super().__init__(**kwargs)
        self.default_factory = default_factory

    def __missing__(self, _):
        return self.default_factory()


def get_mem_usage(precision: int = 2, samples: int = 25, delay: float = 0.01) -> float:
    """Returns current process' memory usage in MB."""
    data = []
    process = psutil.Process(os.getpid())
    data.append(round(process.memory_info().rss / 1000000, precision))
    for _ in range(samples - 1):
        process = psutil.Process(os.getpid())
        data.append(round(process.memory_info().rss / 1000000, precision))
        sleep(delay)
    return sum(data) / len(data)


class MemoryMonitor:
    """This context manager allows mem profiling blocks of code."""
    def __init__(self):
        self._measure = None
        self._used = None
    
    def __enter__(self) -> None:
        self._measure = get_mem_usage()
        return self

    def __exit__(self, *_: list) -> None:
        self._used = get_mem_usage() - self._measure

    def __float__(self):
        return self._used or get_mem_usage() - self._measure


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
        return self._elapsed or time.time() - self._timer


def is_path(path: str) -> bool:
    return osp.exists(osp.expanduser(path))

def is_url(url: str) -> bool:
    return validators.url(url)

def thread_id():
    return current_thread().getName()