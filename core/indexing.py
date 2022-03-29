"""

"""

from typing import Any
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from core.document import Document

def compute_idf(df, N): return np.log10(N / df)

def index(collection: set[Document]) -> dict[str, Any]:
    N = len(collection)
    index = defaultdict(lambda: {'df': 0, 'tf': 0, 'idx': None, 'idf': None, 'posting': {}})

    for doc in tqdm(collection, total = N, desc = 'indexing docs'):
        for term, stats in doc.token_dist.items():
            index[term]['df'] += 1
            index[term]['tf'] += stats['freq']
            index[term]['posting'][doc.base_name] = stats['freq']

    T = len(index)
    sorted_terms = sorted(index.keys())
    for i, term in enumerate(sorted_terms):
        index[term]['idx'] = i

    for stats in tqdm(index.values(), desc = 'computing idf scores'):
        stats['idf'] = compute_idf(stats['df'], N)

    for doc in tqdm(collection, desc = 'vectorizing docs'):
        vector = np.zeros(T)
        for term, stats in doc.token_dist.items():
            tf_idf = stats['freq'] * index['term']['idf']
            stats['tf-idf'] = tf_idf
            vector[index[term]['idx']] = tf_idf
        magnitude = np.linalg.norm(vector)
        doc.vector, doc.magnitude = vector / magnitude, magnitude

    return dict(index)