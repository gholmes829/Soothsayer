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
from threading import Thread
from queue import Queue, Empty
from time import sleep
import tempfile

# debugging
from icecream import ic

# custom
from core.utils import *
from core.cmd_parsing import *
from core.sources import Document, WebPage
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
        self.index = InvertedIndex.load('index.pkl') if osp.exists('index.pkl') else InvertedIndex()
        self.poutput(f'Loaded index with {self.index.num_docs()} docs {self.index.num_terms()} terms.')


    @cmd2.with_argparser(run_index_local_parser)
    def do_index_local(self, args: argparse.Namespace) -> None:
        self.handle_path(args.source)
    complete_index_local = cmd2.Cmd.path_complete

    @cmd2.with_argparser(run_index_web_parser)
    def do_index_web(self, args: argparse.Namespace) -> None:
        urls = set()
        for source in args.sources:
            if is_url(source): urls.add(source)
            elif is_path(source):
                with open(source, 'r') as f:
                    for line in f:
                        cleaned = line.strip()
                        if cleaned and not cleaned.startswith('#'):
                            assert is_url(cleaned)
                            urls.add(cleaned)
            else: raise ValueError('must be url or path')
        self.handle_url(urls, args.dur)
    complete_index_web = cmd2.Cmd.path_complete

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
        self.index.save('index.pkl')
        self.poutput('Index saved.')
        

    def handle_url(self, urls: list[str], crawl_duration: float = None) -> None:
        spider = Spider()
        data = Queue()
        def on_content(url, content, latency):
            data.put((url, content))
            print(f'{time.time()}: with l={latency}, {thread_id()} collected "{url}"')

        
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                t = Thread(target = lambda: spider.crawl(urls, on_content = on_content, timeout = crawl_duration), daemon = True)
                t.start()
                while data.empty(): sleep(0.1)
                while spider.is_crawling:
                    try:
                        url, content = data.get(timeout = 1)
                        try:
                            with open(osp.join(tmp_dir, encode_b64(url) + '.txt'), 'w') as f: f.write(content)
                        except OSError: pass  # TODO file name could be too long
                    except Empty: pass
                t.join()
            except KeyboardInterrupt: spider.signal_stop_crawl()

            while not data.empty():
                url, content = data.get()
                with open(osp.join(tmp_dir, encode_b64(url) + '.txt'), 'w') as f: f.write(content)
            
            web_pages = set(osp.join(tmp_dir, rel_path) for rel_path in os.listdir(tmp_dir))

            with mp.Pool() as p:
                pages = [doc for doc in tqdm(
                    p.imap_unordered(WebPage, web_pages),
                    total = len(web_pages),
                    desc = 'Processing pages'
                )]
        self.index.update(pages)
        self.poutput('Saving index...')
        self.index.save('index.pkl')
        self.poutput('Index saved.')


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
        self.index.save('index.pkl')

    def default(self, statement: cmd2.Statement) -> Optional[bool]:
        """Treats default command as vector space query."""
        self.handle_query('vector', statement.raw)

    def cleanup(self) -> None:
        pass


    @staticmethod
    def run(): sys.exit(App().cmdloop())