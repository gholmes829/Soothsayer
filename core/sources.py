"""

"""

import os.path as osp
import time
import numpy as np
from collections import defaultdict
from icecream import ic

from core.preprocess import preprocess


class Document:
    def __init__(self, path: str) -> None:
        self.path = path
        self.name = osp.splitext(osp.basename(self.path))[0]
        with open(path, 'r') as f:
            tokens = preprocess(f.read())
        
        self.index = defaultdict(list)  # term to locs
        
        for i, term in enumerate(tokens):
            if term: self.index[term].append(i)


class WebPage:
    def __init__(self, data) -> None:
        url, i, clean = data
        self.name = url
        tokens = preprocess(clean)
        print(' '.join([t for t in tokens if t]))
        self.index = defaultdict(list)  # term to locs
        
        for i, term in enumerate(tokens):
            if term: self.index[term].append(i)