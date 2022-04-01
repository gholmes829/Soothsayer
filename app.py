"""

"""

# system
import sys
import os.path as osp, os

# typing
from typing import Optional

# ui
import cmd2
from tqdm import tqdm

# general
import multiprocessing as mp

# debugging
from icecream import ic

# custom
from core.utils import *
from core.cmd_parsing import *
from core.document import Document
from core.indexing import InvertedIndex
from core.querying import handle_vector_query, handle_boolean_query, clear_cache

# static paths
PATH = osp.dirname(osp.realpath(__file__))


class App(cmd2.Cmd):
    def __init__(self) -> None:
        super().__init__()
        self.register_postloop_hook(self.cleanup)
        self.prompt = '>> '
        self.continuation_prompt = '... '
        self.index = InvertedIndex.load('index.gz') if osp.exists('index.gz') else InvertedIndex()


    @cmd2.with_argparser(run_update_index_parser)
    def do_update_index(self, args: argparse.Namespace) -> None:
        source, source_type = args.source
        {
            'url': self.handle_url,
            'path': self.handle_path,
        }[source_type](source)
    complete_update_index = cmd2.Cmd.path_complete


    def handle_path(self, path: str) -> None:
        doc_paths = {osp.join(path, child) for child in os.listdir(path) if child.endswith('.txt') and not osp.isdir(child)}
        with mp.Pool() as p:
            docs = [doc for doc in tqdm(
                p.imap_unordered(Document, doc_paths),
                total = len(doc_paths),
                desc = 'Processing files'
            )]
        self.index.update(docs)
        self.poutput('Saving index...')
        self.index.save('index.gz')
        self.poutput('Index saved.')
        

    def handle_url(self, url: str) -> None:
        pass


    @cmd2.with_argparser(run_query_parser)
    def do_query(self, args: argparse.Namespace) -> None:
        query_type = args.query_type
        query = ' '.join(args.query)
        self.handle_query(query_type, query, args.top_k)


    def handle_query(self, query_type: str, query: str, top_k = None) -> None:
        with Timer() as t:
            most_relevant = {
                'boolean': handle_boolean_query,
                'vector':  handle_vector_query
            }[query_type](self.index, query, top_k = top_k or 10)

        if not most_relevant: return self.poutput('No good matches.\n')
        for i, (name, score) in enumerate(most_relevant, 1):
            self.poutput(f'{i}) {name}: {round(100 * score, 2)}%')
        self.poutput(f'\nCompleted in {round(float(t), 5)} secs\n')


    @cmd2.with_argparser(run_update_index_parser)
    def do_index(self, args: argparse.Namespace) -> None:
        source, source_type = args.source
        {
            'url': self.handle_url,
            'path': self.handle_path,
        }[source_type](source)
    complete_index = cmd2.Cmd.path_complete


    def do_clear_query_cache(self, _) -> None:
        clear_cache()


    def default(self, statement: cmd2.Statement) -> Optional[bool]:
        """Treats default command as vector space query."""
        return self.handle_query('vector', statement.raw)


    def cleanup(self) -> None:
        pass


    @staticmethod
    def run(): sys.exit(App().cmdloop())