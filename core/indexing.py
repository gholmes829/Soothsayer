"""

"""

from typing import Callable
from icecream import ic
from collections import defaultdict
import multiprocessing as mp
import numpy as np
import pickle
from tqdm import tqdm
from itertools import repeat

from core.sources import Document
from core.utils import DefaultDict


def index_item_default():
    return {'idf': None, 'locs': DefaultDict(list), 'idx': None}

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(index_item_default)
        self.doc_magnitudes = defaultdict(int)
        # self.doc_name_to_vec = {}

    def update(self, gen_doc: Callable, doc_sources: set[str] = None) -> None:
        with mp.Pool() as p:
            documents: list[Document] = [doc for doc in tqdm(
                p.imap_unordered(gen_doc, doc_sources),
                total = len(doc_sources),
                desc = 'Processing files'
            )]

        assert documents
        for doc_to_update in documents:
            for term, locs in doc_to_update.index.items():
                self.index[term]['locs'][doc_to_update.name] = locs
                self.index[term]['df'] = len(self[term]['locs'])

        # self._compute_idfs()  # compute all idfs since N changed
        for term, locs in self.index.items():
            self.index[term]['idf'] = np.log10(len(documents) / len(locs))

        for i, term in enumerate(sorted(list(self.index.keys()))): self.index[term]['idx'] = i

        self._compute_doc_magnitudes(documents)

        # self._compute_doc_vecs()

    # def remove(self, documents: set[Document] = None) -> None:
    #     assert documents
    #     for doc_to_del in documents:
    #         del self.doc_name_to_vec[doc_to_del.name]
    #         for term in doc_to_del.index:
    #             del self.index[term]['locs'][doc_to_del.name]
    #             if not self.index[term]['locs']: self.vocab_set.remove(term)
    #             self.index[term]['df'] -= 1

    #     self._compute_idfs()  # compute all idfs since N changed
    #     for i, term in enumerate(sorted(list(self.index.keys()))): self.index[term]['idx'] = i
    #     self._compute_doc_vecs()

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'InvertedIndex':
        with open(path, 'rb') as f:
            return pickle.load(f)

    def num_docs(self) -> int:
        return len(self.doc_magnitudes)

    def num_terms(self) -> int:
        return len(self.index)

    def get_doc_names(self) -> set[str]:
        return set(self.doc_magnitudes.keys()) # set(self.doc_name_to_vec.keys())

    # def _compute_idfs(self) -> None:
    #     for term, locs in self.index.items():
    #         self.index[term]['idf'] = np.log10(len(self.doc_names) / len(locs))

    # def _compute_doc_magnitude(self, doc: Document) -> tuple[str, float]:
    #     magnitude = 0
    #     for token in doc.index:
    #         magnitude += np.square(len(self.index[token]['locs'][doc.name]) * self.index[token]['idf'])
    #     return np.sqrt(magnitude) or 1


    # def _compute_doc_magnitudes(self, documents: list[Document]) -> None:
    #         for doc in tqdm(documents, desc = 'Generating source vectors'):
    #             self.doc_magnitudes[doc.name] = self._compute_doc_magnitude(doc)

    def _compute_doc_magnitude(self, doc_name: str) -> float:
        magnitude = 0
        for token in self.index.keys():
            magnitude += (len(self.index[token]['locs'][doc_name]) * self.index[token]['idf']) ** 2
        return doc_name, np.sqrt(magnitude)

    def _compute_doc_magnitudes(self, new_documents: list[Document]) -> None:
        joined_docs = set(self.doc_magnitudes.keys()) | set(doc.name for doc in new_documents)
        self.doc_magnitudes.clear()

        with mp.Pool() as p:
            self.doc_magnitudes = {doc: magnitude for doc, magnitude in tqdm(
                p.imap_unordered(self._compute_doc_magnitude, joined_docs, chunksize = 250),
                total = len(joined_docs),
                desc = 'Generating source data'
            )}

        # for doc_name in tqdm(joined_docs, desc = 'Generating source data'):
        #     for token in self.index:
        #         # if doc_name in self.index[token]['locs']:
        #         self.doc_magnitudes[doc_name] += (len(self.index[token]['locs'][doc_name]) * self.index[token]['idf']) ** 2

        # for doc_name, sqrd_sum in self.doc_magnitudes.items():
        #     self.doc_magnitudes[doc_name] = np.sqrt(sqrd_sum)
         
    def __len__(self):
        return len(self.index)

    def __getitem__(self, key: str) -> dict:
        return self.index[key]

    def __iter__(self):
        return iter(self.index)

    def __contains__(self, key: str) -> bool:
        return key in self.index