import concurrent.futures
import functools
import itertools
import multiprocessing

import pylib


def run_parallels(work_fn, iterable, max_workers=None, chunksize=1, processing_bar=True, backend_executor=multiprocessing.Pool, debug=False):
    if not debug:
        with backend_executor(max_workers) as executor:
            try:
                works = executor.imap(work_fn, iterable, chunksize=chunksize)  # for multiprocessing.Pool
            except:
                works = executor.map(work_fn, iterable, chunksize=chunksize)

            if processing_bar:
                try:
                    import tqdm
                    try:
                        total = len(iterable)
                    except:
                        total = None
                    works = tqdm.tqdm(works, total=total)
                except ImportError:
                    print('`import tqdm` fails! Run without processing bar!')

            results = list(works)
    else:
        results = [work_fn(i) for i in iterable]
    return results

run_parallels_mp = run_parallels
run_parallels_cfprocess = functools.partial(run_parallels, backend_executor=concurrent.futures.ProcessPoolExecutor)
run_parallels_cfthread = functools.partial(run_parallels, backend_executor=concurrent.futures.ThreadPoolExecutor)


def map_reduce(map_fn,
               reduce_fn,
               map_inputs,
               map_max_workers=None,
               map_chunksize=1,
               reduce_max_workers=None,
               reduce_chunksize=1,
               processing_bar=True,
               backend_executor=multiprocessing.Pool,
               debug=False):
    """A simple map-reduce wrapper.

    Parameters
    ----------
        map_fn :
            `map_fn` should return key-value pairs, i.e., [(k1, v1), (k2, v2), ..., (kn, vn)].
        reduce_fn :
            `reduce_fn` will accept key-value pairs grouped by a same key, i.e., [(k, v1), (k, v2), ..., (k, vn)].
            `reduce_fn` should return a list of records, i.e., [r1, r2, ...].
    Notes
    -----
        map -> group (by key returned from mapper) -> reduce (accepts a group of key-value pairs).
    """
    # map
    list_of_key_values = run_parallels(map_fn,
                                       map_inputs,
                                       max_workers=map_max_workers,
                                       chunksize=map_chunksize,
                                       processing_bar=processing_bar,
                                       backend_executor=backend_executor,
                                       debug=debug)
    key_values = itertools.chain.from_iterable(list_of_key_values)

    # group
    reduce_inputs = (list(g) for _, g in pylib.groupby_unsorted(key_values, key=lambda x: x[0]))

    # reduce
    list_of_reduce_outputs = run_parallels(reduce_fn,
                                           reduce_inputs,
                                           max_workers=reduce_max_workers,
                                           chunksize=reduce_chunksize,
                                           processing_bar=processing_bar,
                                           backend_executor=backend_executor,
                                           debug=debug)
    reduce_outputs = list(itertools.chain.from_iterable(list_of_reduce_outputs))
    return reduce_outputs


if __name__ == '__main__':
    # run_parallels
    import time

    def work(i):
        time.sleep(0.0001)
        i**i
        return i

    t = time.time()
    results = run_parallels_mp(work, range(10000), max_workers=2, chunksize=1, processing_bar=True, debug=False)
    for i in results:
        print(i)
    print(time.time() - t)

    # map_reduce
    import collections

    def map_fn(sentence):
        words = collections.Counter(sentence.split(' '))
        return list(words.items())

    def reduce_fn(word_cnts):
        word_cnts = word_cnts
        return [(word_cnts[0][0], sum(cnt for _, cnt in word_cnts))]

    map_inputs = ['Hello World Hello hello', 'ssbb World ssbb', 'Hadoop Hello']

    reduce_output = map_reduce(map_fn, reduce_fn, map_inputs, debug=False)

    print(reduce_output)
