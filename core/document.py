"""

"""

import os.path as osp
import time
from core.preprocessing import preprocess
from nltk import FreqDist
from icecream import ic


class Document:
    def __init__(self, doc_path: str) -> None:
        self.path = doc_path
        self.base_name = osp.splitext(osp.basename(self.path))[0]
        self.time_stamp = time.time()
        with open(doc_path, 'r') as f:
            self.token_dist = {
                term: {
                    'freq': freq,
                    'tf-idf': None
                } for term, freq in FreqDist(preprocess(f.read())).items()}
        self.vector = None
        self.magnitude = None