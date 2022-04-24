"""

"""

import numpy as np
from bisect import bisect_left
from icecream import ic

from core.preprocess import preprocess
from core.indexing import InvertedIndex

def smooth(x, smoothness: float = 1):
    return 2 * (1 / (1 + np.exp(-x / smoothness)) - 0.5)


def handle_boolean_query(index: InvertedIndex, query: str) -> list[tuple[str, float]]:
    raise NotImplementedError


def make_doc_vector(index: InvertedIndex, doc_name: str, sorted_query_tokens: list[str]):
    vector = np.empty(len(sorted_query_tokens))
    for i, token in enumerate(sorted_query_tokens):
        vector[i] = index[token]['docs'][doc_name]['weight']  # len(index[token]['locs'][doc_name]) * index[token]['idf']
    return vector

# TODO get word synonyms?
def make_query_vector(index: InvertedIndex, sorted_query_tokens: list[str]) -> np.ndarray:
    query_tf = {token: sorted_query_tokens.count(token) for token in set(sorted_query_tokens)}
    vector = np.empty(len(sorted_query_tokens))
    for i, token in enumerate(sorted_query_tokens):
        vector[i] = query_tf[token] * index[token]['idf']
    if not (norm := np.linalg.norm(vector)): return None
    return vector / norm


def find_nearest(container: list[int], target: int) -> int:
    """
    Assumes container is sorted and returns closest value to target.
    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(container, target)
    if pos == 0: return container[0]
    if pos == len(container): return container[-1]
    pre = container[pos - 1]
    post = container[pos]
    return post if post - target < target - pre else pre


def make_proximity_score_vector(index: InvertedIndex, sorted_query_tokens: list[str], candidate_doc_names: list[str]):
    # return np.ones(len(candidate_doc_names))  # NOTE returning this nullifys affect of proximity
    vector = np.empty(len(candidate_doc_names))
    for i, doc in enumerate(candidate_doc_names):
        total_dist = 1  # set at one to avoid divide by zero
        comparisons = 1  # set at one to avoid divide by zero
        qt_count = 0
        for qt in sorted_query_tokens:
            if (qt_locs := index[qt]['docs'][doc]['locs']): qt_count += 1
            remaining = set(sorted_query_tokens) - {qt}
            for compare_qt in remaining:
                if not (compare_qt_locs := index[compare_qt]['docs'][doc]['locs']): break
                for qt_loc in qt_locs:
                    total_dist += np.log10(abs(qt_loc - find_nearest(compare_qt_locs, qt_loc)))  # TODO shld add log here?
                    comparisons += 1
        mean_proximity = 1 / (total_dist / comparisons)
        qt_percent_present = qt_count / len(sorted_query_tokens)
        vector[i] = qt_percent_present * mean_proximity

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