"""
Concurrent Execution
https://docs.python.org/3.7/library/concurrency.html

X https://realpython.com/python-gil/
- https://realpython.com/python-concurrency/

Works with 3.7.1 and not 3.7.2 for some reason or another...

"""

import sys
import time
from threading import Thread
from multiprocessing import Pool
import numpy as np
import pandas as pd


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
    t1 = Thread(target=countdown, args=(count // 4,))
    t2 = Thread(target=countdown, args=(count // 4,))
    t3 = Thread(target=countdown, args=(count // 4,))
    t4 = Thread(target=countdown, args=(count // 4,))
    start = time.time()
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    end = time.time()
    print('time taken in seconds for 4 threads:', end - start)


# multiprocess
def multi_process_count(count):
    pool = Pool(processes=4)
    start = time.time()
    r1 = pool.apply_async(countdown, [count // 4])
    r2 = pool.apply_async(countdown, [count // 4])
    r3 = pool.apply_async(countdown, [count // 4])
    r4 = pool.apply_async(countdown, [count // 4])
    pool.close()
    pool.join()
    end = time.time()
    print('time taken in seconds for 4 multi process:', end - start)


# https://realpython.com/python-concurrency/
# create function that converts pandas dataframe to the below code and takes n rows and p columns
# multivariate stats

def print_df():
    concurrent = pd.DataFrame({'Concurrency Type': ['Pre-emptive multitasking (threading)',
                                                    'Cooperative multitasking (asyncio)',
                                                    'Multiprocessing (multiprocessing)'],
                               'Switching Decision': [
                                   'The operating system decides when to switch tasks external to Python.',
                                   'The tasks decide when to give up control.',
                                   'The processes all run at the same time on different processors.'],
                               'Number of Processors': ['1', '1', 'Many']})

    concurrent = pd.concat([pd.DataFrame(np.array([concurrent.columns]), columns=concurrent.columns),
                            concurrent], ignore_index=True, axis=0)

    # get items length
    c_c_len = concurrent.select_dtypes('object') \
        .apply(lambda x: x.str.len(), axis=1)

    c_c_max_len = c_c_len.max(axis=0)

    for idx, c_c in enumerate(concurrent.iterrows()):
        a0 = c_c[1]['Concurrency Type']
        a1 = c_c[1]['Switching Decision']
        a2 = c_c[1]['Number of Processors']
        m0 = c_c_max_len['Concurrency Type'] + 2
        m1 = c_c_max_len['Switching Decision'] + 2
        m2 = c_c_max_len['Number of Processors'] + 2
        if idx == 1:
            print('|', '-'*m0, '|', '-'*m1, '|', '-'*m2, '|', sep='')
        if idx == 0:
            print(f'|{a0:^{m0}}|{a1:^{m1}}|{a2:^{m2}}|', end='\n')
        else:
            print(f'|{a0:<{m0}}|{a1:<{m1}}|{a2:<{m2}}|', end='\n')




if __name__ == '__main__':

    print_df()

    # input('count:')
    count_in = 100_000_000

    print('running single thread')
    single_count(count_in)

    print('running multi thread')
    multi_thread_count(count_in)

    print('running multi process')
    multi_process_count(count_in)
