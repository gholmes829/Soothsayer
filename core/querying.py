"""

"""

from typing import Hashable, Tuple
import numpy as np
from pipe import where
from core.preprocess import preprocess
from cachetools import cached
from cachetools.keys import hashkey
from icecream import ic

from core.indexing import InvertedIndex

def smooth(x, smoothness: float = 1):
    return 2 * (1 / (1 + np.exp(-x / smoothness)) - 0.5)


def handle_boolean_query(index, query: str) -> list[tuple[str, float]]:
    raise NotImplementedError


def make_query_vector(index, query_tokens: list[str]) -> np.ndarray:
    query_tf = {term: query_tokens.count(term) for term in set(query_tokens)}
    query_vector = np.zeros(len(index))
    for term in query_tokens: query_vector[index[term]['idx']] = query_tf[term] * index[term]['idf']
    if not (query_norm := np.linalg.norm(query_vector)):
        return None
    normalized = query_vector / query_norm
    return normalized


def is_candidate(index: InvertedIndex, query: list[str], doc_name: str) -> bool:
    for q in query:
        if doc_name in index[q]['locs']: return True
    return False

def cache_filter(_, query: str, *args, **kwargs) -> Tuple[Hashable, ...]: return hashkey(query)
query_cache = {}
@cached(cache = query_cache, key = cache_filter)
def handle_vector_query(index: InvertedIndex, query: str, top_k: int = 10) -> list[tuple[str, float]]:
    query_tokens = [term for term in preprocess(query) if term in index]
    query_vector = make_query_vector(index, query_tokens)
    doc_names = [doc for doc in index.doc_name_to_vec if is_candidate(index, query_tokens, doc)]
    doc_matrix = np.array([index.doc_name_to_vec[doc] for doc in doc_names])
    scores = doc_matrix @ query_vector
    score_names, scores = list(zip(*list(sorted(
        [(name, score) for name, score in zip(doc_names, scores)],
        key = (lambda pair: (lambda _, score: score)(*pair)),
        reverse = True,
    ) | where(lambda pair: (lambda _, score: score and not np.isnan(score))(*pair)))))

    scaled = smooth(np.array(scores[:top_k]), 0.05)
    return tuple(zip(score_names[:top_k], scaled))

def clear_cache():
    query_cache.clear()