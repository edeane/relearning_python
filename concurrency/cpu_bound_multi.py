"""
cpu bound problem

"""
import time
import multiprocessing
import numpy as np


def sum_of_i_by_i(up_to):
    return sum(i * i for i in range(up_to))

def sum_of_i_squ(up_to):
    return sum(i ** 2 for i in range(up_to))

def sum_of_np_squ(up_to):
    return sum(np.square(i) for i in np.arange(0, up_to, dtype=np.float))


def sum_of_squares_mul(up_tos):
    with multiprocessing.Pool() as pool:
        return pool.map(sum_of_i_by_i, up_tos)


if __name__ == '__main__':

    numbers = [5_000_000 + x for x in range(20)]
    # 'syn', 'mul', or 'both'
    run_type = 'syn'

    if run_type in ['syn', 'both']:
        print('running syn')
        sum_funs = [
            sum_of_i_by_i,
            sum_of_i_squ,
            sum_of_np_squ,
        ]
        for sum_fun in sum_funs:
            fun_name = sum_fun.__name__
            print(f'running {fun_name}')
            start_time = time.time()
            syn_res = [sum_fun(i) for i in numbers]
            duration = time.time() - start_time
            print(f'synchronous {fun_name} duration {duration} seconds')

    if run_type in ['mul', 'both']:
        print('running mul')
        start_time = time.time()
        mul_res = sum_of_squares_mul(numbers)
        duration = time.time() - start_time
        print(f'multiprocessing duration {duration} seconds')

















