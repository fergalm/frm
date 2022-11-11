from ipdb import set_trace as idebug 
from pprint import pprint
import numpy as np

import dataarray as da


def load_test_da1():
    xx = {
        'a': np.arange(10),
        'b': np.arange(10) + 10,
        'c': np.fmod(np.arange(10) , 3),
    }
    return da.DataArray(xx)


def test_group():
    xx = load_test_da1()
    gr = da.Grouper(xx, 'c')

    assert np.all(gr.get_keys() == [0,1,2])
    da0 = gr.get_group(0)
    return da0
