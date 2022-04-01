"""

"""

import os.path as osp, os
import argparse
import validators
import cmd2

def resolve_source(source: str) -> tuple:
    if validators.url(source): return source, 'url'
    elif osp.exists(osp.expanduser(source)): return source, 'path'
    else: raise argparse.ArgumentTypeError('source is not a path or url')

# INDEX
run_update_index_parser = cmd2.Cmd2ArgumentParser(description = 'evaluate source and update index')
run_update_index_parser.add_argument(
    'source',
    type = resolve_source,
    help = 'crawlable url or path to *.txt collection',
)

# QUERY
run_query_parser = cmd2.Cmd2ArgumentParser(description='run a query')
run_query_parser.add_argument(
    'query_type',
    choices = {'vector', 'boolean'},
    help = 'run query and return relevant sources'
)
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