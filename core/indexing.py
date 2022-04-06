"""

"""

from icecream import ic
from collections import defaultdict
import numpy as np
import compress_pickle
from tqdm import tqdm

from core.sources import Document


def index_item_default():
    return {'idf': None, 'locs': {}, 'idx': None}

class InvertedIndex:
    def __init__(self, documents: set[Document] = None) -> None:
        self.index = defaultdict(index_item_default)
        self.vocab_set = set()
        self.doc_name_to_vec = {}
        if documents: self.update(documents)

    def update(self, documents: set[Document] = None) -> None:
        assert documents
        for doc_to_update in documents:
            self.doc_name_to_vec[doc_to_update.name] = None
            for term, locs in doc_to_update.index.items():
                self.index[term]['locs'][doc_to_update.name] = locs
                self.vocab_set.add(term)
                self.index[term]['df'] = len(self[term]['locs'])

        self._compute_idfs()  # compute all idfs since N changed
        for i, term in enumerate(sorted(list(self.index.keys()))): self.index[term]['idx'] = i
        self._compute_doc_vecs()

    def remove(self, documents: set[Document] = None) -> None:
        assert documents
        for doc_to_del in documents:
            del self.doc_name_to_vec[doc_to_del.name]
            for term in doc_to_del.index:
                del self.index[term]['locs'][doc_to_del.name]
                if not self.index[term]['locs']: self.vocab_set.remove(term)
                self.index[term]['df'] -= 1

        self._compute_idfs()  # compute all idfs since N changed
        for i, term in enumerate(sorted(list(self.index.keys()))): self.index[term]['idx'] = i
        self._compute_doc_vecs()

    def save(self, path: str) -> None:
        compress_pickle.dump(self, path, compression = 'gzip')

    @staticmethod
    def load(path: str) -> 'InvertedIndex':
        return compress_pickle.load(path)

    def num_docs(self) -> int:
        return len(self.doc_name_to_vec)

    def num_terms(self) -> int:
        return len(self.vocab_set)

    def get_doc_names(self) -> set[str]:
        return set(self.doc_name_to_vec.keys())

    def _compute_idfs(self) -> None:
        for term, locs in self.index.items():
            self.index[term]['idf'] = np.log10(len(self.doc_name_to_vec) / len(locs))

    def _compute_doc_vecs(self) -> None:
        # TODO parallelize
        for target_doc in tqdm(self.doc_name_to_vec, desc = 'Generating source vectors'):
            vec = np.zeros(len(self))
            for term in self.vocab_set:
                # could add a forward index to make this loop more efficient... not sure if its worth the memory tho
                try: vec[self.index[term]['idx']] = len(self.index[term]['locs'][target_doc]) * self.index[term]['idf']
                except KeyError: continue
            self.doc_name_to_vec[target_doc] = vec / (np.linalg.norm(vec) or 1)
                    
    def __len__(self):
        return len(self.vocab_set)

    def __getitem__(self, key: str) -> dict:
        return self.index[key]

    def __iter__(self):
        return iter(self.index)

    def __contains__(self, key: str) -> bool:
        return key in self.vocab_set