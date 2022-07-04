import frmbase.support as support
from ipdb import set_trace as idebug
import pandas as pd 
import numpy as np
import pytest



def test_chop():
    strr = "This is a line\n"
    assert support.chop(strr) == "This is a line"


def test_chomp():
    str1 = "This is a line\n"
    assert support.chomp(str1) == "This is a line"

    str2 = "This is a line"
    assert support.chomp(str2) == "This is a line"


def test_int_to_bin_str():
    int = 12
    assert support.int_to_bin_str(int) == "0000 0000 0000 0000 0000 0000 0000 1100"


    arr = np.arange(4)
    assert support.int_to_bin_str(arr)[0] == "0000 0000 0000 0000 0000 0000 0000 0000"    



def test_npmap():
    x = np.arange(4)
    y = support.npmap(lambda x: x**2, x)
    assert      np.allclose(y, x**2)


def test_lmap():
    x = [0, 1, 2,3, 4]
    y = support.lmap(lambda x: x**2, x)
    assert  y == [0, 1, 4, 9 ,16]



def test_printe():
    val = support.printe(12.3456, .12)
    assert val == "12.35(12)", val


def test_printse():
    val = support.printse(12000.3456, 12)
    assert val == "1.2000(12)e04", val





@support.timer
def test_example1():
    """Run this from console and you should see the appropriate text being printed"""
    import time
    time.sleep(3)



def test_check_cols_in_df():
    df = pd.DataFrame(columns="a b c".split())

    assert support.check_cols_in_df(df, ['a', 'b'])
    assert not support.check_cols_in_df(df, 'abcd'.split())