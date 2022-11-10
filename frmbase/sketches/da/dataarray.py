from ipdb import set_trace as idebug 
from pprint import pprint
import numpy as np

from collections import namedtuple
from typing import Dict

"""
A sketch of a lighterweight replacement to DataFrames

A dataarray is a wrapper around a dictionary of columnar data with intuitive indexing
Columns are referred to by strings (or other hashable types), and rows are indexed by integer.
This allows

da[:4, 'col]  to work, with no messing around with .loc and .iloc

By having no operations that work at the series level other than getting (and maybe setting)
the class is much smaller, easier to maintain, and more flexible.

Todo
-----
o Implement setters
o Groupers -- untested
x row() to return a named tuple?

o Define acceptable datatypes
    o What operations do they support
        o slicing
        o ???
        o list indexing col[ [1,2,3,5] ]
    o A datetime column 
    o A string column
        __getitem__(slice)
        upper
        lower
        contains -> bool array
        len  -> int array
        regex
        + (concat)       
    o A list column (just a list, but allows binary indexing) 
o Metadata
o merge -- see merge.py
    o left out and right out and cross joins
    o unit tests
o immutable flag
o filter method
"""


def lmap(x, y):
    return list(map(x, y))

def npmap(x, y):
    return np.array(lmap(x, y))

class DataArray:
    ...
class DataArray:
    """
    Addressing

    da['a'] #Get the entire column called 'a'
    da[:4, 'a'] #Get the first 4 elts of column a
    da[:4]   #Get first elts of all arrays

    da[ ['a', 'b']]  #Get a DataArray containing just these two cols 
    da[:4, ['a', 'b']]  #Get first 4 rows of just these two cols 

    Setting should work for all these operations too.
    """
    def __init__(self, src:dict = None, meta:Dict[str,str]=None):
        self.meta = None or meta

        if src is None:
            self.size = None
            self.dict = dict()
        else:
            sizes = npmap(lambda x: len(src[x]), src.keys())
            assert np.all(sizes == sizes[0]) , sizes
            self.size = sizes[0]
            self.dict = src

    def __contains__(self, col):
        return col in self.dict.keys()

    def __len__(self):
        return self.size

    def __repr__(self):
        cols = self.columns()
        return "<DataArray %i rows, %i columns: %s>" %(self.size,len(cols), cols)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        if len(key) == 1:
            key = key[0]
            if isinstance(key, list):
                return self._get_column_subset(key)
            elif isinstance(key, slice):
                return self._get_slice(key)
            elif isinstance(key, np.ndarray) and key.dtype == bool:
                return self._get_slice(key)
            else:
                return self.dict[key[0]]
        elif len(key) == 2:
            col = key[1]

            if col == slice(None, None, None):
                col = list(self.columns())  #colon implies all columns
            sl = key[0]
            tmp = self.__getitem__(col)
            return tmp.__getitem__(sl)
        else:
            raise ValueError("Too many dimensions")

    def __setitem__(self, key, value):
        raise TypeError("DataArrays are immutable for now")
        # if not isinstance(key, tuple):
        #     key = (key,)
        #
        # if len(key) == 1:
        #     assert len(value) == self.size or self.size is None
        #     self.dict[key[0]] = value
        # if len(key) == 2:
        #     col = key[1]
        #     sl = key[0]
        #     self.dict[col][sl] = value

    def columns(self) -> set:
        #Maybe a list?
        return set(self.dict.keys())

    def select(self, *cols) -> DataArray:
        return self._get_column_subset(cols)

    def filter(self, predicate) -> DataArray:
        raise NotImplementedError()

    def copy(self) -> DataArray:
        data = {}
        for c in self.columns():
            data[c] = self.dict[c].copy()  #May not be general enough
        return DataArray(data)

    def _get_column_subset(self, cols):
        src = dict()
        for c in cols:
            src[c] = self.dict[c].copy()
        return DataArray(src)

    def _get_slice(self, sl):
        src = dict()
        for c in self.columns():
            src[c] = self.dict[c][sl]
        return DataArray(src)


    def row(self, i) ->namedtuple:
        """This looks very slow..."""
        cols = self.columns()
        vals = lmap(lambda x: self.dict[x][i], cols)
        ttype = namedtuple('row', self.columns())
        row = ttype(*vals)
        return row


class Grouper:
    def __init__(self, da, *cols, **opts):
        """This algo insists you pass in column names, no option of passing in an array
        Not to hard to change.
        """

        groups = {}
        for i in range(len(da)):
            if len(cols) == 1:
                hash = da[i,cols[0]]
            else:
                hash = tuple(map(lambda x: da[i, x], cols))

            try:
                groups[hash].append(i)
            except KeyError:
                groups[hash] = [i]

        self.da = da
        self.groups = groups

    def get_keys(self):
        return list(self.groups.keys())

    def get_indices(self, key):
        return self.groups[key]

    def get_group(self, key):
        wh = self.get_indices(key)
        # idebug()
        return self.da[wh, :]

    def apply(self, func, *args, **kwargs):
        dflist = []
        for k in self.get_keys():
            da = self.get_group(k)
            out = func(da, *args, **kwargs)
            dalist.append(out)
        return concat(dalist)


def concat(*das):
    num_da = len(das)
    size = np.sum(npmap(len, das))

    cols = {}
    for da in das:
        cols |= set(da.columns())

    out = dict()
    for c in cols:
        #TODO Write a more general purpose concatenator
        out[c] = np.concatenate(npmap(lambda x: x[c], das))

    return DataArray(out)


