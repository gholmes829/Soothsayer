"""

"""

import os.path as osp
import time
from typing import Callable
import numpy as np
from collections import defaultdict
from icecream import ic

from core.preprocess import preprocess
from core.utils import decode_b64


def noop(_): return _

class Document:
    def __init__(
            self,
            path: str,
            transform_name: Callable[[str], str] = noop,
        ) -> None:
        self.path = path
        self.name = transform_name(osp.splitext(osp.basename(self.path))[0])
        with open(path, 'r') as f:
            tokens = preprocess(f.read())
        
        self.index = defaultdict(list)  # term to locs
        
        for i, term in enumerate(tokens):
            if term: self.index[term].append(i)

def WebPage(path: str):
    return Document(path, decode_b64)

# class WebPage:
#     def __init__(self, data) -> None:
#         url, content = data
#         self.name = url
#         tokens = preprocess(content)
#         self.index = defaultdict(list)  # term to locs
        
#         for i, term in enumerate(tokens):
#             if term: self.index[term].append(i)