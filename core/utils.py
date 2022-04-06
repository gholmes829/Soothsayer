"""

"""

import os.path as osp, os
from threading import current_thread
import validators
import psutil
import time
from icecream import ic
import base64


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
        return self._elapsed or time.time() - self._timer


def is_path(path: str) -> bool:
    return osp.exists(osp.expanduser(path))

def is_url(url: str) -> bool:
    return validators.url(url)

def thread_id():
    return current_thread().getName()
    

alt_chars = bytes('_-', 'utf-8')
def encode_b64(text: str) -> str:
    return base64.b64encode(bytes(text, 'utf-8'), alt_chars).decode('ascii')

def decode_b64(text: str) -> str:  
    return base64.b64decode(text.encode('ascii'), alt_chars).decode('utf-8')