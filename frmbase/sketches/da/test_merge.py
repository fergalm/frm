from ipdb import set_trace as idebug 
from pprint import pprint
import numpy as np

import dataarray as da
import merge 

def load_test_da1():
    xx = {
        'a': np.arange(10),
        'b': np.arange(10) + 10,
        'c': np.arange(10) + 20,
    }
    return da.DataArray(xx)

def load_test_da2():
    xx = {
        'a': np.arange(10),
        'b': np.arange(10) ,
        'd': np.arange(10) + 20,
    }
    return da.DataArray(xx)


#These are implementation tests. Can be removed if design changes
def test_get_key():
    xx = load_test_da1()

    cols = ['a']
    keys = merge.get_keys(xx, cols)

    assert len(keys) == len(xx)
    for elt in keys:
        assert len(elt) == len(cols)
    # return keys



def test_bisearch():
    x = np.arange(10)
    x[4:7] = 4
    
    wh = merge.bisearch(x, 4)
    assert np.all( wh == [4,5,6])


def test_gen_cols1():
    x1 = load_test_da1()
    x2 = load_test_da2()

    left, right = merge.gen_cols(x1, x2, ['a'], ['a'], "_x", "_y" )
    print(left, right)
    assert left == set(['a', 'b_x', 'c'])
    assert right == set(['b_y', 'd'])

def test_gen_cols2():
    x1 = load_test_da1()
    x2 = load_test_da2()

    left, right = merge.gen_cols(x1, x2, ['a'], ['b'], "_x", "_y" )
    print(left, right)
    assert left == set(['a_x', 'b', 'c'])
    assert right == set(['a_y', 'd'])
