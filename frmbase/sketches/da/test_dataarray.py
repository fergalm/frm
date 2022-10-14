from ipdb import set_trace as idebug 
from pprint import pprint
import numpy as np

import dataarray as da 


def test_set_col():
    obj = da.DataArray()
    obj['x'] = np.arange(10)

    y = obj['x']
    assert np.allclose(y, np.arange(10))


def test_get_slice():
    obj = da.DataArray()
    obj['x'] = np.arange(10)

    y = obj['x'][:4]
    assert np.allclose(y, np.arange(4))    

    y = obj[:4, 'x']
    assert np.allclose(y, np.arange(4))    


def test_set_slice():
    obj = da.DataArray()
    obj['x'] = np.arange(10)
    obj[:4, 'x'] = [11, 12, 13, 14]

    y = obj['x']
    assert np.allclose(y, [11, 12, 13, 14, 4, 5, 6, 7, 8, 9])    


def test_get_many_cols():
    x = np.arange(10)
    src = dict(a=x, b=x)
    obj = da.DataArray(src)
    obj2 = obj[ ['a', 'b']]
    assert isinstance(obj2, da.DataArray)
    for a, b in zip(obj2.keys(), ['a', 'b']):
        assert a == b