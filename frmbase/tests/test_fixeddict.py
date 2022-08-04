from frmbase.fixeddict import FixedDict
from ipdb import set_trace as idebug
from pprint import pprint
import pytest

def test_smoke():
    d = FixedDict(*'a b c d'.split())

    d['a'] = 4
    with pytest.raises(KeyError):
        d['aa']

def test_setting():
    data = dict(a=1, b=2)
    d = FixedDict(data)

    d['a'] = 2011
    with pytest.raises(KeyError):
        d['aa'] = 2011


def test_init_with_kwargs():
    d = FixedDict(a=1, b=2)
    assert d['a'] == 1

def test_init_with_dict():
    data = dict(a=1, b=2)
    d = FixedDict(data)

    assert d['a'] == 1
    with pytest.raises(KeyError):
        d['aa']

def test_attribute_access():
    data = dict(a=1, b=2)
    d = FixedDict(data)

    assert d.a == 1
    assert list(d.keys()) == list(data.keys())

    with pytest.raises(AttributeError):
        d.aa

def test_attribute_setting():
    data = dict(a=1, b=2)
    d = FixedDict(data)

    d['a'] = 5
    assert d.a == 5
    d.a = 6
    assert d.a == 6

    with pytest.raises(AttributeError):
        d.aa = 4  #Key doesn't exist

def test_hasatter():
    data = dict(a=1, b=2)
    d = FixedDict(data)

    assert hasattr(d, 'keys')
    assert hasattr(d, 'a')


from frmbase.parmap import parmap 

def test_parallel():
    """Fixed dict seems to fail when called in parallel
    
    This fails because a FixedDict needs to do control
    how it is unpickled better. 
    """
    data = dict(a=1, b=2)
    d = FixedDict(data)
    keys = 'a b'.split()

    res = parmap(_parallel, [d], keys, engine='serial')
    assert all(res)
    res = parmap(_parallel, [d], keys, engine='multi', timeout_sec=2)
    assert all(res)


def _parallel(d, key):
    pprint(d.__dict__)
    d[key] = 0
    return True


import pickle 
def test_pickling():
    data = dict(a=1, b=2)
    d = FixedDict(data)

    data = pickle.dumps(d)
    print(data)
    d_new = pickle.loads(data)
