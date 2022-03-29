"""

"""

import os.path as osp, os
import argparse
import re
from icecream import ic
from typing import Iterator
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
from nltk import FreqDist
from pipe import where
import time
import compress_pickle

from core import document, indexing, preprocessing
from core.utils import *


PATH = osp.dirname(osp.realpath(__file__))
doc_path_pattern = re.compile(r'.+[.]txt')


def make_argparser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser(description = 'generate search engine')
    argparser.add_argument('--raw_query', '-q', help = 'query to search for')
    argparser.add_argument('--collection', '-c', help = 'path to dir with *.txt files')
    argparser.add_argument('--top_k', '-k', type = int, help = 'max number of results to return')
    argparser.add_argument('--doc_sim', '-d', nargs = 2, help = 'compare two documents')
    return argparser


def make_collection(collection_path: str) -> Iterator[document.Document]:
    assert osp.exists(collection_path)
    assert osp.isdir(collection_path)
    doc_paths = {osp.join(collection_path, f_path) for f_path in os.listdir(collection_path) if doc_path_pattern.fullmatch(f_path)}

    with mp.Pool() as p:
        for doc in tqdm(
            p.imap_unordered(document.Document, doc_paths),
            total = len(doc_paths),
            desc = 'processing docs'
        ): yield (doc.base_name, doc)


def smooth(x, smoothness: float = 1): return 2 * (1 / (1 + np.exp(-x / smoothness)) - 0.5)


def should_compute(query, doc):
    for q in query:
        if q in doc: return True
    return False


def run_query(collection, index, raw_query: str, top_k):
    # TODO cache results
    N, T = len(collection), len(index)
    query = [term for term in preprocessing.preprocess(raw_query) if term in index]
    
    query_token_dist = dict(FreqDist(query))
    query_vector = np.zeros(T)
    for term in query: query_vector[index[term]['idx']] = query_token_dist[term] * index[term]['idf']
    if not (query_norm := np.linalg.norm(query_vector)):
        print('No good matches found.')
        return
    query_vector /= query_norm
    
    # try matrix multiplication, storing vectors as matrix, folder for index
    top_docs = [doc for doc in collection.values() if should_compute(query, doc.token_dist)]
    if not top_docs:
        print('No good matches found.')
        return
    doc_matrix = np.array([doc.vector for doc in top_docs])
    doc_names = [doc.base_name for doc in top_docs]
    scores = doc_matrix @ query_vector


    doc_names, scores = list(zip(*list(sorted(
        [(name, score) for name, score in zip(doc_names, scores)],
        key = (lambda pair: (lambda _, score: score)(*pair)),
        reverse = True,
    ) | where(lambda pair: (lambda _, score: score and not np.isnan(score))(*pair)))))

    # doc_names, scores = list(zip(*list(sorted(
    #     [(doc.base_name, query_vector @ doc.vector) for doc in collection.values() if should_compute(query, doc.token_dist)],
    #     key = (lambda pair: (lambda _, score: score)(*pair)),
    #     reverse = True,
    # ) | where(lambda pair: (lambda _, score: score and not np.isnan(score))(*pair)))[:top_k]))

    scaled = smooth(np.array(scores[:top_k]), 0.05)

    return tuple(zip(doc_names[:top_k], scaled))


def query(collection, index, raw_query: str, top_k):
    timer = time.time()
    res = run_query(collection, index, raw_query, top_k)
    elapsed = time.time() - timer
    for i, (name, score) in enumerate(res, 1):
        print(f'{i}) {name}: {round(100 * score, 2)}%')
    ic(round(elapsed, 3))


def run_doc_sim(collection, index, d1, d2):
    timer = time.time()
    print(f'Similarity: {collection[d1].vector @ collection[d2].vector}')
    elapsed = time.time() - timer
    ic(round(elapsed, 3))

# add ranked doc similarity
def main() -> None:
    argparser = make_argparser()
    args = argparser.parse_args()
    collection_path: str = args.collection
    raw_query: str = args.raw_query
    top_k: int = args.top_k

    if (doc_compare := args.doc_sim):
        d1, d2 = doc_compare
        d1 = osp.splitext(osp.basename(d1))[0]
        d2 = osp.splitext(osp.basename(d2))[0]

    index_path = osp.join(PATH, 'index.pkl')
    if collection_path:
        collection = dict(make_collection(collection_path))
        index = indexing.index(collection.values())
        # for doc in collection.values(): del doc.token_dist
        print('Saving to disk...')
        with open(index_path, 'wb') as f:
            compress_pickle.dump((collection, index), f, compression='gzip')
        print('Index created.')
    else:
        with open(index_path, 'rb') as f:
            collection, index = compress_pickle.load(f, compression='gzip')
    
    if doc_compare:
        run_doc_sim(collection, index, d1, d2)

    if raw_query:
        query(collection, index, raw_query, top_k)

    elif not collection_path:
        top_k = top_k or 3
        while (raw_query := input('>> ')) not in {'exit()', 'quit()'}:
            query(collection, index, raw_query, top_k)
    
    


if __name__ == '__main__':
    main()