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
from contextlib import nullcontext
import tempfile

# debugging
from icecream import ic

# custom
from core.utils import *
from core.cmd_parsing import *
from core.sources import Document
from core.indexing import InvertedIndex
from core.querying import run_query
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
        self.poutput(f'{"Loaded" if exists else "Created"} index with {len(self.index.docs)} docs {len(self.index)} terms.')


    @cmd2.with_argparser(run_index_local_parser)
    def do_index_local(self, args: argparse.Namespace) -> None:
        source: str = args.source
        doc_paths = {osp.join(source, child) for child in os.listdir(source) if child.split('.')[-1] in {'json', 'txt'}}
        self.index.update(Document.make, doc_paths)
        self.poutput('Saving index...')
        self.index.save('index.gz')
        self.poutput('Index saved.')
    complete_index_local = cmd2.Cmd.path_complete


    @cmd2.with_argparser(run_index_web_parser)
    def do_crawl_web(self, args: argparse.Namespace) -> None:
        sources: list[str] = args.sources
        dur: float = args.dur
        output: str = args.output
        urls = set()
        for source in sources:
            if is_url(source):
                urls.add(source)
            elif is_path(source):
                with open(source, 'r') as f:
                    for clean_line in (line.split('#')[0].strip() for line in f):
                        if clean_line: urls.add(clean_line)
            else: raise ValueError('must be url or path')

        dir_context = nullcontext(output) if output else tempfile.TemporaryDirectory()
        with dir_context as tmp_dir:
            saved_web_paths = Spider.basic_crawl(tmp_dir, urls, timeout = dur)
            sleep(0.1)
            self.index.update(Document.from_json, saved_web_paths)

        self.poutput('Saving index...')
        self.index.save('index.gz')
        self.poutput('Index saved.')
    complete_crawl_web = cmd2.Cmd.path_complete


    @cmd2.with_argparser(run_query_parser)
    def do_query(self, args: argparse.Namespace) -> None:
        self.handle_query(' '.join(args.query), args.top_k)


    def handle_query(self, query: str, top_k: int = None) -> None:
        ic(query, top_k)
        with Timer() as t:
            most_relevant = run_query(self.index, query, top_k = top_k or 10)
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
        self.handle_query(statement.raw)

    def cleanup(self) -> None:
        pass


    @staticmethod
    def run(): sys.exit(App().cmdloop())