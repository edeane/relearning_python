"""
Concurrent Execution
https://docs.python.org/3.7/library/concurrency.html

X https://realpython.com/python-gil/
- https://realpython.com/python-concurrency/

"""

import sys
import time
from threading import Thread
from multiprocessing import Pool


# sys get ref count
def sys_get_ref_count():
    a = []
    b = a
    print('sys get ref count of a:', sys.getrefcount(a))

# countdown function
def countdown(n):
    while n > 0:
        n -=1


# single thread
def single_count(count):
    start = time.time()
    countdown(count)
    end = time.time()
    print('time taken in seconds for single thread:', end - start)


# multithread
def multi_thread_count(count):
    t1 = Thread(target=countdown, args=(25_000_000,))
    t2 = Thread(target=countdown, args=(25_000_000,))
    start = time.time()
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    end = time.time()
    print('time taken in seconds for 2 threads:', end - start)


# multiprocessing
def multi_process_count(count):
    pool = Pool(processes=2)
    start = time.time()
    r1 = pool.apply_async(countdown, [count//2])
    r2 = pool.apply_async(countdown, [count // 2])
    pool.close()
    pool.join()
    end = time.time()
    print('tame taken in seconds for 2 multiprocess:', end - start)



if __name__ == '__main__':

    # input('count:')
    count_in = 50_000_000
    print('count in __main__ countdown:', count_in)

    single_count(count_in)

    multi_thread_count(count_in)

    multi_process_count(count_in)
