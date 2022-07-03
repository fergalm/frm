#testing code
import numpy as np
import pytest

from frmbase.parmap import parmap
import asyncio

def sqr(x):
    print("The squre of %i is %i" %(x, x**2))
    return x*x


def power(x, n):
    return x**n


def hypot(x, y):
    return np.sqrt(x**2 + y**2)

def hypotn(x, y, n):
    return x**n + y**n


async def asqr(x):
    await asyncio.sleep(2)
    return x**2


def test_sqr():
    x = np.arange(10)

    for sp in ['serial', 'multi', 'threads']:
        res = parmap(sqr, x, engine=sp, progress='silent')
        assert np.all(res == x **2)


def test_pow():
    x = np.arange(10)

    for sp in ['serial', 'multi', 'threads']:
        res = parmap(power, x, fargs={'n':3}, engine=sp, progress='silent')
        assert np.all(res == x **3)

def test_hypot():
    x = np.arange(10)
    y = 10 + np.arange(10)

    for sp in ['serial', 'multi', 'threads']:
        res = parmap(hypot, x, y, engine=sp, progress='silent')
        assert len(res) == 100
    return res


def test_hypotn():
    x = np.arange(10)
    y = np.arange(10)

    for sp in ['serial', 'multi', 'threads']:
        res = parmap(hypotn, x, y, fargs={'n':3}, engine=sp, progress='silent')
        assert len(res) == 100
        assert res[-1] == 2*9**3


def failing_task(x):
    if x == 2:
        1/0
    return x

async def afailing_task(x):
    if x == 2:
        1/0
    return x


def test_task_that_fails():
    x = np.arange(5)

    with pytest.raises(ZeroDivisionError):
        res = parmap(failing_task, x, engine='serial')

    res = parmap(failing_task, x, engine='multi')
    assert res[2] is None
    assert res[1] == 1


def test_async():
    x = np.arange(10)
    res = parmap(asqr, x, engine='async')
    assert np.all(res == x**2)

def test_async_fail():
    x = np.arange(10)

    with pytest.raises(ZeroDivisionError):
        parmap(afailing_task, x, engine='async')


