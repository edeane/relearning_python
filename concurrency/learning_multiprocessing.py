"""
multiprocessing

- creates new instance of python on each CPU

"""

import requests
import multiprocessing
import time

session = None

def set_global_session():
    global session
    if not session:
        session = requests.Session()


def download_site(url):
    with session.get(url) as response:
        name = multiprocessing.current_process().name
        print(f'{name}: read {len(response.content)} from {url}')
        return response.content


def download_all_sites(sites):
    with multiprocessing.Pool(initializer=set_global_session, processes=4) as pool:
        result = pool.map(download_site, sites)
        return result

if __name__ == '__main__':

    sites = ['http://www.jython.org', 'http://olympus.realpython.org/dice'] * 80
    start_time = time.time()
    mul_res = download_all_sites(sites)
    duration = time.time() - start_time
    print(f'downloaded {len(sites)} in {duration} seconds')


