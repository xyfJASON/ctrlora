import collections.abc as abc


class Nested:
    """Wrapper for nested structures.

    This class wraps indexable objects nested by dict and tuple, and overwrites
    the __getitem__ method to index the objects preserving the nested structure.

    The innermost elements should be indexable but cannot be dict or tuple.

    Examples
    --------
    x = Nested(({'a': [1, 2, 3], 'b': ('456', [7, 8, 9])}, [10, 11, 12]))
    x[0]
    >>> ({'a': 1, 'b': ('4', 7)}, 10)
    x[2]
    >>> ({'a': 3, 'b': ('6', 9)}, 12)

    """

    def __init__(self, nested):
        self.nested = nested
        self.len = Nested.nested_len(nested)

    def __getitem__(self, i):
        return Nested.nested_getitem(self.nested, i)

    def __len__(self):
        if self.len is None:
            raise TypeError("the object has no len()!")
        else:
            return self.len

    def __repr__(self):
        return f'Nested({repr(self.nested)})'

    @staticmethod
    def nested_len(nested):
        if isinstance(nested, dict):
            lens = [Nested.nested_len(v) for v in nested.values()]
        elif isinstance(nested, tuple):
            lens = [Nested.nested_len(v) for v in nested]
        else:
            try:
                lens = [len(nested)]
            except TypeError:
                if not hasattr(nested, '__getitem__'):
                    raise TypeError(f"'{type(nested).__name__}' object does not support indexing")
                else:
                    lens = [None]
        lens = set(lens)
        lens.discard(None)

        if len(lens) > 1:
            raise Exception('Nested elements should have the same length!')
        elif len(lens) == 1:
            return lens.pop()
        else:
            return None

    @staticmethod
    def nested_getitem(nested, i):
        if isinstance(nested, dict):
            return {k: Nested.nested_getitem(v, i) for k, v in nested.items()}
        elif isinstance(nested, tuple):
            return tuple(Nested.nested_getitem(v, i) for v in nested)
        else:
            return nested[i]


class NestedIterable(abc.Iterable):
    """Wrapper for nested iterables.

    This class wraps iterables nested by dict and tuple, and overwrites the __iter__
    and __next__ method to make each iteration item preserve the nested structure.

    """

    def __init__(self, nested_iterable):
        if not NestedIterable.check_iterable(nested_iterable):
            raise TypeError('The innermost elements should be iterable!')
        self.nested_iterable = nested_iterable

    def __iter__(self):
        # return a new object
        return NestedIterable(NestedIterable.nested_iter(self.nested_iterable))

    def __next__(self):
        try:
            return NestedIterable.nested_next(self.nested_iterable)
        except TypeError:
            raise TypeError('The object is not an iterator!')

    @staticmethod
    def check_iterable(nested_iterable):
        if isinstance(nested_iterable, dict):
            return all(NestedIterable.check_iterable(v) for v in nested_iterable.values())
        elif isinstance(nested_iterable, tuple):
            return all(NestedIterable.check_iterable(v) for v in nested_iterable)
        else:
            return hasattr(nested_iterable, '__iter__')

    @staticmethod
    def nested_iter(nested_iterable):
        if isinstance(nested_iterable, dict):
            return {k: NestedIterable.nested_iter(v) for k, v in nested_iterable.items()}
        elif isinstance(nested_iterable, tuple):
            return tuple(NestedIterable.nested_iter(v) for v in nested_iterable)
        else:
            return iter(nested_iterable)

    @staticmethod
    def nested_next(nested_iterator):
        if isinstance(nested_iterator, dict):
            return {k: NestedIterable.nested_next(v) for k, v in nested_iterator.items()}
        elif isinstance(nested_iterator, tuple):
            return tuple([NestedIterable.nested_next(v) for v in nested_iterator])
        else:
            return next(nested_iterator)


if __name__ == '__main__':
    print('test Nested')
    x = Nested(({'a': [1, 2, 3], 'b': ('456', [7, 8, 9])}, [10, 11, 12]))
    print(len(x))
    for i in x:
        print(i)
    for i in x:
        print(i)

    print('test NestedIterable')
    x = NestedIterable(({'a': [1, 2, 3], 'b': ('456', [7, 8, 9])}, [10, 11, 12]))
    for i in x:
        print(i)
    for i in x:
        print(i)

    print('test NestedIterable.__iter__')
    x = iter(x)
    print(next(x))
    next(x)
    x = iter(x)
    for i in x:
        print(i)
