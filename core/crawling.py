"""
TODO could add option to prevent search going to parent of seed path
"""

from threading import Thread, Event, Timer as TimerTrigger, Condition
from queue import Queue
from typing import Any, Callable
import requests
from bs4 import BeautifulSoup
import time
from math import ceil
from icecream import ic
from urllib.parse import urlparse, urljoin
import random

from core.utils import *



class Spider:
    def __init__(self):
        # timing/ sync
        self.interrupt = Event()
        self.frontier_cv = Condition()

        # collection
        self.collected: set[str] = set()
        self.front_queues: list[Queue[str]] = []  # a front queue for each priority level
        self.back_queues: dict[str, Queue[str]] = {}  # will have back queue for each domain
        self.ready_queue: Queue[str] = Queue()  # lowest = highest priority, iterates thru item to handle ties
        self.domain_profiles: dict[str: dict[str, Any]] = {}

        # state
        self.num_threads = None
        self.is_crawling = False

    def crawl(
            self,
            seeds: set[str],
            on_content: Callable[[str, str], None],
            url_priority: Callable[[str], int] = lambda _: 0,
            k_weights: tuple[float] = (1,),
            timeout: float = None,
        ) -> None:
        self._initialize_crawl(seeds, k_weights, url_priority)
        workers = [self._make_worker(on_content, url_priority, k_weights) for _ in range(self.num_threads)]
        self.is_crawling = True
        for t in workers: t.start()
        try: self.interrupt.wait(timeout = timeout)
        except KeyboardInterrupt: pass
        self.signal_stop_crawl()
        for t in workers: t.join()
        self._reset()

    
    def should_crawl(self) -> bool:
        return not self.interrupt.is_set()

    def signal_stop_crawl(self) -> None:
        self.interrupt.set()


    def _crawl_worker(
            self,
            on_content: Callable[[str, str], None],
            url_priority: Callable[[str], int],
            k_weights: tuple[int]
        ) -> None:
        urls_crawled = 0  # number of urls crawled
        errs = 0  # number of errors during scraping
        latency_sum = 0
        ic(f'starting thread {thread_id()}')

        with Timer() as t:
            while self.should_crawl():
                ready_time, target_domain = self.ready_queue.get()
                latency = time.time() - ready_time
                latency_sum += latency
                target_queue = self.back_queues[target_domain]
                target_url = target_queue.get()
                
                try:
                    resp = requests.get(target_url)
                    fetched_at = time.time()
                    if resp.status_code != 200: raise ConnectionError('received non-200 code')
                    content, urls = Spider._parse_html(resp.text, target_domain)
                    self._queue_new_urls(urls, url_priority)
                    if content: on_content(target_url, content, latency)
                except (requests.exceptions.ConnectionError, ConnectionError):
                    fetched_at = time.time()
                    errs += 1
                urls_crawled += 1
                
                if target_queue.empty(): self._recycle_back_queue(target_domain, target_queue, fetched_at, k_weights)
                else: self._queue_domain(fetched_at, target_domain)
                   
        ic(thread_id(), errs, urls_crawled, latency_sum, float(t))


    def _initialize_crawl(self, seeds: set[str], k_weights: tuple[int], url_priority: Callable[[str], int]) -> None:
        self.front_queues = [Queue() for _ in range(len(k_weights))]
        for seed in seeds:
            self.collected.add(seed)
            domain = Spider.get_domain(seed)
            self._add_to_frontier(seed, url_priority(seed))
            if domain not in self.back_queues: self._initialize_new_domain(domain, Queue())
            self.back_queues[domain].put(seed)
        self.num_threads = min(ceil(len(self.back_queues) / 2), 64)  # per mercator reccomendations


    def _add_to_frontier(self, url: str, priority: int) -> None:
        with self.frontier_cv:
            self.front_queues[priority].put(url)
            self.frontier_cv.notify_all()


    def _pull_from_frontier(self, k_weights: tuple[int]) -> str:
        with self.frontier_cv:
            while not (available := [i for i, queue in enumerate(self.front_queues) if not queue.empty()]):
                self.frontier_cv.wait()
            weights = [k_weights[i] for i in available]
            return self.front_queues[random.choices(available, weights=weights, k=1).pop()].get()


    def _profile_domain(self, url: str) -> dict:
        return {
            'crawl_delay': 1,  # TODO update with actual robot values
        }


    def _make_worker(self, *args, **kwargs):
        return Thread(target = self._crawl_worker, args = args, kwargs = kwargs, daemon = True)


    def _queue_new_urls(self, urls, url_priority):
        for new_url in urls - self.collected:
            self.collected.add(new_url)
            if Spider.get_domain(new_url) in self.back_queues:
                self._add_to_frontier(new_url, url_priority(new_url))

    def _queue_domain(self, fetched_at, domain):
        ready_at = fetched_at + self.domain_profiles[domain]['crawl_delay']
        wait_time = ready_at - time.time()
        if wait_time > 0: TimerTrigger(wait_time, lambda: self.ready_queue.put((ready_at, domain))).start()
        else: self.ready_queue.put((ready_at, domain))


    def _initialize_new_domain(self, new_domain, target_queue):
        self.domain_profiles[new_domain] = self._profile_domain(new_domain)
        self.back_queues[new_domain] = target_queue
        self.ready_queue.put((time.time(), new_domain))


    def _recycle_back_queue(
            self,
            target_domain: str,
            target_queue: Queue[str],
            fetched_at: float,
            k_weights: tuple[int]
        ) -> None:
        del self.back_queues[target_domain]
        while self.should_crawl():
            new_url = self._pull_from_frontier(k_weights)
            new_domain = Spider.get_domain(new_url)
            if new_domain in self.back_queues:  # currently active
                self.back_queues[new_domain].put(new_url)
            elif new_domain in self.domain_profiles:  # seen before but not active
                target_queue.put(new_url)
                self.back_queues[new_domain] = target_queue
                self._queue_domain(fetched_at, new_domain)
                break
            else:  # never seen before
                target_queue.put(new_url)
                self._initialize_new_domain(new_domain, target_queue)
                break

        
    @staticmethod
    def _parse_html(text: str, target_domain: str) -> tuple[str, set[str]]:
        soup = BeautifulSoup(text, 'html.parser')
        urls = {urljoin(target_domain, url.get('href')) for url in soup.findAll('a')}
        content = ' '.join(p.text for p in soup('p'))
        return content, urls


    def _reset(self) -> None:
        self.collected.clear()
        self.front_queues.clear()
        self.back_queues.clear()
        self.ready_queue.queue.clear()
        self.domain_profiles.clear()
        self.interrupt = Event()
        self.frontier_cv = Condition()
        self.num_threads = None
        self.is_crawling = False


    @staticmethod
    def get_domain(url: str) -> str:
        return '{uri.scheme}://{uri.netloc}/'.format(uri=urlparse(url))