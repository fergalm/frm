from ipdb import set_trace as idebug 
from pprint import pprint
import numpy as np

import dataarray as da 

def load_test_da1():
    xx = {
        'a': np.arange(10),
        'b': np.arange(10) + 10,
        'c': np.arange(10) + 20,
    }
    return da.DataArray(xx)

def test_get_col():
    xx = load_test_da1()
    col = xx['a']
    assert np.allclose(col, np.arange(10))

def test_get_slice():
    xx = load_test_da1()
    col = xx[:4, 'a']
    # idebug()
    assert np.allclose(col, np.arange(4))

    col = xx[2:6, 'a']
    assert np.allclose(col, np.arange(2,6))


def test_select_one_row():
    xx = load_test_da1()
    yy = xx[0, 'c']
    assert isinstance(yy, np.ndarray)
    assert len(yy) == 1 

    yy = xx[0, ['b', 'c']]
    assert isinstance(yy, da.DataArray)
    assert len(yy) == 1 


def test_select_multiple_cols():
    xx = load_test_da1()
    yy = xx[['a', 'b']]

    print(yy)
    assert yy.columns() == set('a b'.split())

def test_select_multiple_cols_by_slice():
    xx = load_test_da1()
    yy = xx[:4, ['a', 'b']]

    print(yy)
    assert yy.columns() == set('a b'.split())
    assert np.allclose(yy['a'], np.arange(4))

def test_select_by_boolean_array():
    xx = load_test_da1()
    idx = np.arange(10) > 5

    yy = xx[idx]
    assert len(yy) == 4

def test_select_by_list():
    xx = load_test_da1()
    sl = [0,3,6]
    yy = xx[sl, 'a']
    print(yy)
    assert np.allclose(yy, [0,3,6]), yy

def test_select_by_two_lists():
    xx = load_test_da1()
    sl = [0,3,6]
    yy = xx[sl, ['a', 'b']]

    assert len(yy) == len(sl)
    assert yy.columns() == set(['a', 'b']), yy


def test_in_operator():
    xx = load_test_da1()
    assert 'a' in xx
    assert not 'aa' in xx


def test_get_row():
    xx = load_test_da1()
    row = xx.row(0)
    assert row.a == 0
    assert row.b == 10
    assert row.c == 20
# def test_set_col():
#     obj = da.DataArray()
#     obj['x'] = np.arange(10)
#
#     y = obj['x']
#     assert np.allclose(y, np.arange(10))
#
#
# def test_get_slice():
#     obj = da.DataArray()
#     obj['x'] = np.arange(10)
#
#     y = obj['x'][:4]
#     assert np.allclose(y, np.arange(4))
#
#     y = obj[:4, 'x']
#     assert np.allclose(y, np.arange(4))
#
#
# def test_set_slice():
#     obj = da.DataArray()
#     obj['x'] = np.arange(10)
#     obj[:4, 'x'] = [11, 12, 13, 14]
#
#     y = obj['x']
#     assert np.allclose(y, [11, 12, 13, 14, 4, 5, 6, 7, 8, 9])
#
#
# def test_get_many_cols():
#     x = np.arange(10)
#     src = dict(a=x, b=x)
#     obj = da.DataArray(src)
#     obj2 = obj[ ['a', 'b']]
#     assert isinstance(obj2, da.DataArray)
#     for a, b in zip(obj2.keys(), ['a', 'b']):
#         assert a == b