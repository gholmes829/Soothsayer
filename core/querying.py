"""

"""

import numpy as np
from pipe import where
from core.preprocess import preprocess
from icecream import ic

def smooth(x, smoothness: float = 1):
    return 2 * (1 / (1 + np.exp(-x / smoothness)) - 0.5)

def handle_boolean_query(index, query: str) -> list[tuple[str, float]]:
    pass

def make_query_vector(index, query):
    query_tokens = [term for term in preprocess(query) if term in index]
    query_tf = {term: query_tokens.count(term) for term in set(query_tokens)}
    query_vector = np.zeros(len(index))
    for term in query_tokens: query_vector[index[term]['idx']] = query_tf[term] * index[term]['idf']
    if not (query_norm := np.linalg.norm(query_vector)):
        return None
    normalized = query_vector / query_norm
    return normalized

def handle_vector_query(index, query: str, top_k: int = 10) -> list[tuple[str, float]]:
    query_vector = make_query_vector(index, query)

    doc_names = [doc_name for doc_name in index.doc_name_to_vec]
    doc_matrix = np.array([doc_vec for doc_vec in index.doc_name_to_vec.values()])
    scores = doc_matrix @ query_vector

    doc_names, scores = list(zip(*list(sorted(
        [(name, score) for name, score in zip(doc_names, scores)],
        key = (lambda pair: (lambda _, score: score)(*pair)),
        reverse = True,
    ) | where(lambda pair: (lambda _, score: score and not np.isnan(score))(*pair)))))

    scaled = smooth(np.array(scores[:top_k]), 0.05)
    return tuple(zip(doc_names[:top_k], scaled))