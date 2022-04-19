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
        vector[i] = index[token]['docs'][doc_name]['weight']  # len(index[token]['locs'][doc_name]) * index[token]['idf']
    return vector

# TODO get word synonyms
def make_query_vector(index: InvertedIndex, sorted_query_tokens: list[str]) -> np.ndarray:
    query_tf = {token: sorted_query_tokens.count(token) for token in set(sorted_query_tokens)}
    vector = np.zeros(len(sorted_query_tokens))
    for i, token in enumerate(sorted_query_tokens):
        vector[i] = query_tf[token] * index[token]['idf']
    if not (norm := np.linalg.norm(vector)): return None
    return vector / norm


def make_proximity_score_vector(index: InvertedIndex, sorted_query_tokens: list[str], candidate_doc_names: list[str]):
    vector = np.ones(len(candidate_doc_names))  # TODO change this to zeros
    token_locs = [index[token]['docs'][doc]['locs'] for doc in candidate_doc_names for token in sorted_query_tokens]
    
    return vector



def is_candidate(index: InvertedIndex, query_tokens: list[str], doc_name: str) -> bool:
    for token in query_tokens:
        if doc_name in index[token]['docs']: return True
    return False


def handle_vector_query(index: InvertedIndex, query: str, top_k: int = 10) -> list[tuple[str, float]]:
    query_tokens = [token for token in preprocess(query) if token in index]
    sorted_query_tokens = sorted(query_tokens)
    candidate_doc_names = [doc for doc in index.docs if is_candidate(index, query_tokens, doc)]
    candidate_doc_names_array = np.array(candidate_doc_names)
    if not candidate_doc_names_array.shape[0]: return None
    doc_matrix = np.array([make_doc_vector(index, doc, sorted_query_tokens) for doc in candidate_doc_names_array])
    query_vector = make_query_vector(index, sorted_query_tokens)
    proximity_score_vector = make_proximity_score_vector(index, sorted_query_tokens, candidate_doc_names)
    scores = proximity_score_vector * (doc_matrix @ query_vector)
    sorted_idx = np.argsort(scores)[::-1]
    return tuple(zip(
        candidate_doc_names_array[sorted_idx][:top_k],
        smooth(np.array(scores[sorted_idx][:top_k]), 0.05)
    ))