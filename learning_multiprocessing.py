
import multiprocessing
import time


def myfunc(t):
    print("{} starts".format(t))
    time.sleep(1)
    print("{} ends".format(t))


def main():
    tasks = ['1.txt', '2.txt', '3.txt', '4.txt', '5.txt']
    pool = multiprocessing.Pool(processes=2)
    pool.map(myfunc, tasks, chunksize=1)
    pool.close()
    pool.join()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
