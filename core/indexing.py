"""

"""

from typing import Callable
from icecream import ic
import multiprocessing as mp
import numpy as np
from collections import defaultdict
import compress_pickle
from tqdm import tqdm
from itertools import repeat

from core.sources import Document
from core.utils import DefaultReadDict, Timer, MemoryMonitor


def forward_index_default():
    return {'weight': 0, 'locs': DefaultReadDict(list)}

def index_item_default():
    return {'idx': None, 'idf': None, 'docs': defaultdict(forward_index_default)}

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(index_item_default)
        self.docs = set()

    def update(self, gen_doc: Callable, doc_sources: set[str] = None) -> None:
        # create documents
        with mp.Pool() as p:
            documents: list[Document] = [doc for doc in tqdm(
                p.imap_unordered(gen_doc, doc_sources),
                total = len(doc_sources),
                desc = 'Processing files'
            )]

        assert documents
        for doc_to_update in documents:
            self.docs.add(doc_to_update.name)
            for token, locs in doc_to_update.index.items():
                self.index[token]['docs'][doc_to_update.name]['locs'] = locs

        # index position of tokens
        for i, token in enumerate(sorted(list(self.index.keys()))):
            self.index[token]['idx'] = i

        # compute idfs and magnitude to normalize doc vectors
        self._compute_idfs()  # compute all idfs since N changed
        self._compute_doc_wieghts()

    # def remove(self, documents: set[Document] = None) -> None:
    #     assert documents
    #     for doc_to_del in documents:
    #         del self.doc_name_to_vec[doc_to_del.name]
    #         for token in doc_to_del.index:
    #             del self.index[token]['locs'][doc_to_del.name]
    #             if not self.index[token]['locs']: self.vocab_set.remove(token)
    #             self.index[token]['df'] -= 1

    #     self._compute_idfs()  # compute all idfs since N changed
    #     for i, token in enumerate(sorted(list(self.index.keys()))): self.index[token]['idx'] = i
    #     self._compute_doc_vecs()

    def save(self, path: str) -> None:
        # with open(path, 'wb') as f:
        compress_pickle.dump(self, path, compression='gzip')

    @staticmethod
    def load(path: str) -> 'InvertedIndex':
        # with open(path, 'rb') as f:
        return compress_pickle.load(path, compression='gzip')

    def _compute_idfs(self) -> None:
        for data in self.index.values():
            data['idf'] = np.log10(len(self.docs) / len(data['docs']))

    @staticmethod
    def _compute_doc_weights(simplified_index: list, doc_name: str) -> tuple[str, float]:
        accum_norm = 0
        weights = {}
        for token, tf, idf in simplified_index:
            tf_idf = tf[doc_name] * idf
            if tf_idf:
                weights[token] = tf_idf
                accum_norm += tf_idf ** 2
        norm = np.sqrt(accum_norm)
        return doc_name, {token: weight / norm for token, weight in weights.items()}
    
    @staticmethod
    def _compute_doc_weights_star(args) -> tuple[str, float]:
        return InvertedIndex._compute_doc_weights(*args)

    @staticmethod
    def get_doc_tf(token_data: dict) -> DefaultReadDict:
        return DefaultReadDict(int, **{doc_name: len(doc_data['locs']) for doc_name, doc_data in token_data['docs'].items()})


    def _compute_doc_wieghts(self) -> None:
        simplified_index = [(token, InvertedIndex.get_doc_tf(token_data), token_data['idf']) for token, token_data in self.index.items()]
        with mp.Pool() as p:
            for doc_name, token_weights in tqdm(
                    p.imap_unordered(InvertedIndex._compute_doc_weights_star, zip(repeat(simplified_index), self.docs), chunksize=250),
                    total = len(self.docs),
                    desc = 'Generating vector data'
                ):
                for token, weight in token_weights.items():
                    self.index[token]['docs'][doc_name]['weight'] = weight


            # self.doc_magnitudes = {doc_name: magnitude for doc_name, magnitude in tqdm(
            #     p.imap_unordered(InvertedIndex._compute_doc_magnitude_star, zip(repeat(simplified_index), joined_docs), chunksize = 250),
            #     total = len(joined_docs),
            #     desc = 'Generating source data'
            # )}

         
    def __len__(self):
        return len(self.index)

    def __getitem__(self, key: str) -> dict:
        return self.index[key]

    def __iter__(self):
        return iter(self.index)

    def __contains__(self, key: str) -> bool:
        return key in self.index