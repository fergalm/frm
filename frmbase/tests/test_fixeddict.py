from frmbase.fixeddict import FixedDict
from ipdb import set_trace as idebug
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