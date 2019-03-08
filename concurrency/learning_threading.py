"""
Synchronous Version, Not Threading
- https://realpython.com/python-concurrency/

The requests.Session object allows requests to do some fancy networking tricks that speeds things up...

Threading
Thread -
Pool - pool of threads
Executor - controls when each thread runs

queue.Queue is a threadsafe data structure

"""

import requests
import time
import threading
import concurrent.futures



def download_site_syn(url, session):
    with session.get(url) as response:
        print(f'read {len(response.content)} from {url}')
        return response.content


def download_all_sites_syn(sites):
    result = []
    with requests.Session() as session:
        for url in sites:
            result.append(download_site_syn(url, session))
    return result


thread_local = threading.local()


def get_session():
    if not getattr(thread_local, 'session', None):
        thread_local.session = requests.Session()
    return thread_local.session


def download_site_thr(url):
    session = get_session()
    with session.get(url) as response:
        print(f'read {len(response.content)} from {url}')
        return response.content


def download_all_sites_thr(sites):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        result = executor.map(download_site_thr, sites)
    return result




if __name__ == '__main__':

    # either 'syn', 'thr', or 'both'
    type_to_run = 'both'
    sites = ['http://www.jython.org', 'http://olympus.realpython.org/dice'] * 80

    if type_to_run in ('both', 'syn'):
        print('running syn')
        start_time = time.time()
        syn_res = download_all_sites_syn(sites)
        duration = time.time() - start_time
        print(f'synchronous version: downloaded {len(sites)} in {duration} seconds')

    if type_to_run in ('both', 'thr'):
        print('running thr')
        start_time = time.time()
        thr_res = list(download_all_sites_thr(sites))
        duration = time.time() - start_time
        print(f'threaded version: downloaded {len(sites)} in {duration} seconds')


