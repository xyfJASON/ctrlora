import collections
import itertools


# ==============================================================================
# =                                   utils                                    =
# ==============================================================================

def distinct_sorted(iterable, *, key=None, reverse=False):
    lst = sorted(iterable, key=key, reverse=reverse)
    lst = [next(g) for _, g in itertools.groupby(lst, key=key)]
    return lst


def groupby_unsorted(iterable, key=None):
    key = (lambda x: x) if key is None else key
    groups = collections.OrderedDict()
    for x in iterable:
        groups.setdefault(key(x), []).append(x)
    for k, g in groups.items():
        yield k, (x for x in g)


def index_sorted(iterable, *, key=None, reverse=False):
    key = (lambda x: x) if key is None else key
    index, result = zip(*sorted(enumerate(iterable), key=lambda i_x: key(i_x[1]), reverse=reverse))
    return list(index), list(result)


def index_distinct_sorted(iterable, *, key=None, reverse=False):
    key = (lambda x: x) if key is None else key
    index, result = zip(*distinct_sorted(enumerate(iterable), key=lambda i_x: key(i_x[1]), reverse=reverse))
    return list(index), list(result)


argsort = index_sorted
arg_distinct_sort = index_distinct_sorted


# ==============================================================================
# =                                enhance iter                                =
# ==============================================================================

class IterHeadPrevious():

    def __init__(self, n=1):
        self._n = n

    def __str__(self):
        return 'HeadPrevious-%d' % self._n


class IterTailNext():

    def __init__(self, n):
        self._n = n

    def __str__(self):
        return 'IterTailNext+%d' % self._n


def check_iter_head(iterable):
    it = iter(iterable)
    try:
        yield True, next(it)
    except StopIteration:
        return
    for e in it:
        yield False, e


def check_iter_tail(iterable):
    it = iter(iterable)
    try:
        e = next(it)
    except StopIteration:
        return
    while True:
        try:
            nxt = next(it)
            yield False, e
            e = nxt
        except StopIteration:
            yield True, e
            break


def check_iter_head_tail(iterable):
    for is_start, (is_end, e) in check_iter_head(check_iter_tail(iterable)):
        yield is_start, is_end, e


check_iter_start = check_iter_head
check_iter_end = check_iter_tail
check_iter_start_end = check_iter_head_tail


def previous_iter(iterable):
    it = iter(iterable)

    try:
        previous = next(it)
    except StopIteration:
        return
    yield IterHeadPrevious(1), previous

    for current in it:
        yield previous, current
        previous = current


def next_iter(iterable):
    it = iter(iterable)

    try:
        current = next(it)
    except StopIteration:
        return
    while True:
        try:
            nxt = next(it)
            yield current, nxt
            current = nxt
        except StopIteration:
            yield current, IterTailNext(1)
            break


def triple_iter(iterable):
    it = iter(iterable)

    try:
        previous, current = IterHeadPrevious(1), next(it)
    except StopIteration:
        return

    while True:
        try:
            nxt = next(it)
            yield previous, current, nxt
            previous, current = current, nxt
        except StopIteration:
            yield previous, current, IterTailNext(1)
            break


def previous_n_list_iter(list_like_iterable, n=1):
    previous_n = itertools.tee(list_like_iterable, n + 1)
    previous_n = [itertools.chain([IterHeadPrevious(i) for i in range(l, 0, -1)], previous) for l, previous in enumerate(previous_n)]
    return zip(*previous_n[::-1])


def next_n_list_iter(list_like_iterable, n=1):
    next_n = itertools.tee(list_like_iterable, n + 1)
    next_n = [itertools.islice(itertools.chain(next, [IterTailNext(i) for i in range(1, l + 1)]), l, None) for l, next in enumerate(next_n)]
    return zip(*next_n)


def previous_n_next_m_list_iter(list_like_iterable, n=1, m=1):
    previous_n = itertools.tee(list_like_iterable, n + 1)
    previous_n = [itertools.chain([IterHeadPrevious(i) for i in range(l, 0, -1)], previous) for l, previous in enumerate(previous_n)]
    next_m = itertools.tee(list_like_iterable, m + 1)
    next_m = [itertools.islice(itertools.chain(next, [IterTailNext(i) for i in range(1, l + 1)]), l, None) for l, next in enumerate(next_m)]
    return zip(*(previous_n[::-1] + next_m[1:]))


def triple_list_iter(list_like_iterable):
    return previous_n_next_m_list_iter(list_like_iterable, n=1, m=1)
