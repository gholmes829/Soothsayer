"""

"""

from threading import Thread
import requests
from bs4 import BeautifulSoup
import re
from time import time, sleep
from icecream import ic

from core.utils import Timer


url_ptn = re.compile('^https://')

class Spider:
    def __init__(self, seeds: set[str]) -> None:
        self.seeds = seeds
        self.frontier = list(seeds)
        self.searched = set()
        self.buffer = []

    def crawl(self, timeout: float = None, delay = 0.1):
        with Timer() as t:
            i = 0
            while self.frontier and float(t) < timeout:
                ic(i)
                target = self.frontier.pop()
                res = requests.get(target)
                if res.status_code != 200:
                    continue
                    # raise ConnectionError(res.status_code)
                self.searched.add(target)

                soup = BeautifulSoup(res.text, 'html.parser')
                links = {link for t in soup.find_all('a', attrs = {'href': url_ptn}) if (link := t.get('href')) not in self.searched}
                for link in links: self.frontier.append(link)
                

                for data in soup(['style', 'script']): data.decompose()
                filtered = ' '.join(soup.stripped_strings)

                self.buffer.append((target, i, filtered))

                sleep(delay)
                i += 1

    def clear_buffer(self):
        self.buffer.clear()