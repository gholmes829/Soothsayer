"""

"""

import os.path as osp
import time
from collections import defaultdict
from typing import Any
from icecream import ic
import json

from core.preprocess import preprocess


class Document:
    def __init__(self, name: str, type: str, raw_content: str) -> None:
        self.type = type
        self.name = name
        self.index = defaultdict(list)  # term to locs
        self.created_at = time.time()
        
        for i, term in enumerate(preprocess(raw_content)):
            if term: self.index[term].append(i)

    @staticmethod
    def from_txt(fpath: str) -> 'Document':
        assert fpath.endswith('.txt')
        with open(fpath, 'r') as f:
            return Document(osp.splitext(osp.basename(fpath))[0], 'file', f.read())

    @staticmethod
    def from_json(fpath: str) -> 'Document':
        assert fpath.endswith('.json')
        with open(fpath, 'r') as f:
            data = json.load(f)
            return Document(data['name'], data['type'], data['content'])