from ipdb import set_trace as idebug 
from pprint import pprint
import numpy as np

def lmap(x, y):
    return list(map(x, y))

def npmap(x, y):
    return np.array(lmap(x, y))

class DataArray:
    def __init__(self, src:dict = None):

        if src is None:
            self.size = None
            self.dict = dict()
        else:
            sizes = npmap(lambda x: len(src[x]), src.keys())
            assert np.all(sizes == sizes[0]) , sizes
            self.size = sizes[0]
            self.dict = src

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        if len(key) == 1:
            col = key[0]
            if isinstance(col, list):
                return self.get_sub_array(col)
            else:
                return self.dict[key[0]]
        elif len(key) == 2:
            col = key[1]
            sl = key[0]
            return self.dict[col][sl]
        else:
            raise ValueError("Too many dimensions")

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            key = (key,)

        if len(key) == 1:
            assert len(value) == self.size or self.size is None
            self.dict[key[0]] = value
        if len(key) == 2:
            col = key[1]
            sl = key[0]
            self.dict[col][sl] = value

    def keys(self):
        return self.dict.keys()

    def get_sub_array(self, cols):
        src = dict()
        for c in cols:
            src[c] = self.dict[c].copy()
        return DataArray(src)
"""
col
"""