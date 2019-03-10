"""
asycio

- Event loop places tasks in "ready" list or "waiting" list
- FIFO
- Unlike threading tasks never give up control without intentionally doing so
- Resources can be shared more easily
- "await" allows the task to hand control back to the event loop
- "async" is like a flag that tell Python that the function uses await
- in order to use await you must have async

"""

import asyncio
import time
import aiohttp


async def download_site(session, url):
    async with session.get(url) as response:
        print(f'read {response.content_length} from {url}')


async def download_all_sites(sites):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in sites:
            task = asyncio.ensure_future(download_site(session, url))
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == '__main__':

    sites = ['http://www.jython.org', 'http://olympus.realpython.org/dice'] * 80
    start_time = time.time()
    asy_res = asyncio.run(download_all_sites(sites))
    duration = time.time() - start_time
    print(f'downloaded {len(sites)} sites in {duration} seconds')

