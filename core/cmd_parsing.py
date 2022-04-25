"""

"""

import argparse
import cmd2

from core.utils import is_path, is_url

def validate_source(source: str) -> tuple:
    assert is_url or is_path(source)
    return source

def validate_path(path: str) -> str:
    assert is_path(path)
    return path

def validate_url(url: str) -> str:
    assert is_url(url)
    return url

# INDEX
run_index_web_parser = cmd2.Cmd2ArgumentParser(description = 'evaluate urls and update index')
run_index_web_parser.add_argument(
    'sources',
    nargs = '+',
    type = validate_source,
    help = 'crawlable urls',
)

run_index_web_parser.add_argument(
    '--dur',
    '-d',
    type = float,
    help = 'how long to crawl for in secs',
)

run_index_web_parser.add_argument(
    '--output',
    '-o',
    type = validate_path,
    help = 'dump crawled data to persistent dir',
)

run_index_local_parser = cmd2.Cmd2ArgumentParser(description = 'evaluate source and update index')
run_index_local_parser.add_argument(
    'source',
    type = validate_path,
    help = 'path to *.txt collection',
)

# QUERY
run_query_parser = cmd2.Cmd2ArgumentParser(description='run a query')
run_query_parser.add_argument(
    'query',
    nargs = '+',
    help = 'query to return relevant sources for'
)
run_query_parser.add_argument(
    '--top_k',
    '-k',
    type = int,
    help = 'number of top results to return'
)