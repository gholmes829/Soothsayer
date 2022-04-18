"""

"""

import numpy as np
from core.preprocess import preprocess
from icecream import ic

from core.indexing import InvertedIndex

def smooth(x, smoothness: float = 1):
    return 2 * (1 / (1 + np.exp(-x / smoothness)) - 0.5)


def handle_boolean_query(index: InvertedIndex, query: str) -> list[tuple[str, float]]:
    raise NotImplementedError


def make_doc_vector(index: InvertedIndex, doc_name: str, sorted_query_tokens: list[str]):
    vector = np.zeros(len(sorted_query_tokens))
    for i, token in enumerate(sorted_query_tokens):
        # if doc_name in index[token]['locs']:
        vector[i] = len(index[token]['locs'][doc_name]) * index[token]['idf']
        # else: vector[i] = 0
    return vector / index.doc_magnitudes[doc_name]

# TODO get word synonyms
def make_query_vector(index: InvertedIndex, sorted_query_tokens: list[str]) -> np.ndarray:
    query_tf = {token: sorted_query_tokens.count(token) for token in set(sorted_query_tokens)}
    vector = np.zeros(len(sorted_query_tokens))
    for i, token in enumerate(sorted_query_tokens):
        vector[i] = query_tf[token] * index[token]['idf']
    if not (norm := np.linalg.norm(vector)): return None
    return vector / norm


def is_candidate(index: InvertedIndex, query: list[str], doc_name: str) -> bool:
    for q in query:
        if doc_name in index[q]['locs']: return True
    return False


def handle_vector_query(index: InvertedIndex, query: str, top_k: int = 10) -> list[tuple[str, float]]:
    query_tokens = [term for term in preprocess(query) if term in index]
    sorted_query_tokens = sorted(query_tokens)
    doc_names = np.array([doc for doc in index.doc_magnitudes if is_candidate(index, query_tokens, doc)])
    if not doc_names.shape[0]: return None
    doc_matrix = np.array([make_doc_vector(index, doc, sorted_query_tokens) for doc in doc_names])
    query_vector = make_query_vector(index, sorted_query_tokens)
    scores = doc_matrix @ query_vector
    sorted_idx = np.argsort(scores)[::-1]
    return tuple(zip(
        doc_names[sorted_idx][:top_k],
        smooth(np.array(scores[sorted_idx][:top_k]), 0.05)
    ))