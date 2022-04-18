"""

"""

# system
import sys
import os.path as osp, os

# typing
from typing import Optional

# ui
import cmd2

# general
import multiprocessing as mp
import tempfile

# debugging
from icecream import ic

# custom
from core.utils import *
from core.cmd_parsing import *
from core.sources import Document
from core.indexing import InvertedIndex
from core.querying import handle_vector_query, handle_boolean_query
from core.crawling import Spider

# static paths
PATH = osp.dirname(osp.realpath(__file__))


class App(cmd2.Cmd):
    def __init__(self) -> None:
        super().__init__()
        self.register_postloop_hook(self.cleanup)
        self.prompt = '>> '
        self.continuation_prompt = '... '
        exists = osp.exists('index.gz')
        self.index = InvertedIndex.load('index.gz') if exists else InvertedIndex()
        self.poutput(f'{"Loaded" if exists else "Created"} index with {self.index.num_docs()} docs {self.index.num_terms()} terms.')


    @cmd2.with_argparser(run_index_local_parser)
    def do_index_local(self, args: argparse.Namespace) -> None:
        source: str = args.source
        doc_paths = {osp.join(source, child) for child in os.listdir(source) if child.endswith('.txt')}
        self.index.update(Document.from_txt, doc_paths)
        self.poutput('Saving index...')
        self.index.save('index.gz')
        self.poutput('Index saved.')
    complete_index_local = cmd2.Cmd.path_complete

    @cmd2.with_argparser(run_index_web_parser)
    def do_index_web(self, args: argparse.Namespace) -> None:
        sources: list[str] = args.sources
        dur: float = args.dur
        urls = set()
        for source in sources:
            if is_url(source):
                urls.add(source)
            elif is_path(source):
                with open(source, 'r') as f:
                    for clean_line in (line.split('#')[0].strip() for line in f):
                        if clean_line: urls.add(clean_line)
            else: raise ValueError('must be url or path')

        with tempfile.TemporaryDirectory() as tmp_dir:
            saved_web_paths = Spider.basic_crawl(tmp_dir, urls, timeout = dur)
            self.index.update(Document.from_json, saved_web_paths)

        self.poutput('Saving index...')
        self.index.save('index.gz')
        self.poutput('Index saved.')
    complete_index_web = cmd2.Cmd.path_complete


    @cmd2.with_argparser(run_query_parser)
    def do_query(self, args: argparse.Namespace) -> None:
        query_type = args.query_type
        query = ' '.join(args.query)
        self.handle_query(query_type, query, args.top_k)


    def handle_query(self, query_type: str, query: str, top_k: int = None) -> None:
        with Timer() as t:
            most_relevant = {
                'boolean': handle_boolean_query,
                'vector':  handle_vector_query
            }[query_type](self.index, query, top_k = top_k or 10)
        if not most_relevant: return self.poutput('\nNo good matches.\n')
        for i, (name, score) in enumerate(most_relevant, 1):
            self.poutput(f'{i}) {name}: {round(100 * score, 2)}%')
        self.poutput(f'\nCompleted in {round(float(t), 5)} secs\n')


    def do_clear_index(self, _) -> None:
        """Clears and overwrites existing index."""
        self.index = InvertedIndex()
        self.index.save('index.gz')


    def default(self, statement: cmd2.Statement) -> Optional[bool]:
        """Treats default command as vector space query."""
        self.handle_query('vector', statement.raw)

    def cleanup(self) -> None:
        pass


    @staticmethod
    def run(): sys.exit(App().cmdloop())