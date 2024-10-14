"""
Created on Mon Nov 28 14:27:46 2016

Convenience functions to speed development

Inventory:
---------------

Decorators
-------------
@timer
    Print name of function being run, then time elapsed while it ran

@cache
    Save results of running a function with one argument to a cache file


Other functions:
Timer()
    Time blocks of code instead of a single function.


@author: fergal
"""

from ipdb import set_trace as idebug
from glob import glob
import pandas as pd
import numpy as np
import datetime
import inspect
import math
import os
import re


#Work around for windows
if 'HOME' not in os.environ:
    try:
        os.environ['HOME'] = os.environ['USERPROFILE']
    except KeyError:
        pass 

def create_chunks(array, chunksize):
    """Split an array into smaller chunks.

    Helpful when processing a large batch of very small jobs in parallel.
    Break the list into chunks and process in chunk in parallel.

    In Python >3.12 you should use itertools.batch instead
    """
    if chunksize < 1:
        raise ValueError("chunksize must be at least 1")

    bounds = np.arange(0, len(array), chunksize)
    bounds = np.append(bounds, [len(array)])
    out = []
    for i in range(len(bounds)-1):
        lwr = bounds[i]
        upr = bounds[i+1]
        out.append(array[lwr:upr])
    return out



def check_cols_in_df(df: pd.DataFrame, cols: list) -> bool:
    """Because I always get this wrong"""
    return check_columns_in_df(df, cols)

def check_columns_in_df(df: pd.DataFrame, cols: list) -> bool:
    """Check that every element of cols in a column in the dataframe

    Inputs
    --------
    df
        A dataframe
    cols
        (list or iterable) Columns you expect in dataframe


    Returns
    -------
    **True** if all `cols` are columns in dataframes
    """
    return len(set(cols) - set(df.columns)) <= 0



def count_duplicates(data):
    """Count the number of each occurence of an element in a list

    Input
    ----------
    data
        (list, array, or any 1d iterable)


    Returns
    ------------
    A dictionary
    """

    out = dict()
    for elt in data:
        try:
            out[elt] += 1
        except KeyError:
            out[elt] = 1

    return out


def chop(string):
    return string[:-1]

def chomp(string):
    if string[-1] == '\n':
        return string[:-1]
    return string


def ensure_is_in_timezone(date: pd.Series, tz: str, naive_tz='UTC') -> pd.Series:
    """Ensure the dates are in the expected timezone

    This function handles both timezone-naive timestamps and timezone
    aware timestamps in the wrong timezone.

    If the timezone is naive, assume that it represents UTC. This
    may not be what you want in all cases.

    Inputs
    ---------
    date: pd.Series
        A Pandas series contain dates as the values. This will not work if
        the dates are in the index
    tz: str
        The desired timezone, e.g "America/New_York", or "Europe/Dublin"
        See the [timezone database](https://en.wikipedia.org/wiki/Tz_database)
    naive_tz: str
        If the input dates are naive, i.e have no associated timezone,
        treat them as if they are in this timezone.

    Returns
    -----------
    A timezone aware series of datetimes in the desired timezone. Note that
    input series is not modified by this function. Also, the return series is always
    a copy of the input, even if no changes are needed.
    """
    date = date.copy()
    if date[0].tz is None:
        date = date.dt.tz_localize(naive_tz).dt.tz_convert(tz)
    elif date[0].tz != tz:
        date = date.dt.tz_convert(tz)
    return date


def int_to_bin_str(arr):
    """Convert a 32bit int (or numpy array of ints) to a bit string

    Inputs
    -------
    arr
        (int or np array of ints). Values to convert

    Returns
    --------
    string, or array of strings


    Example
    ----------
    12 ---> "0000 0000 0000 0000 0000 0000 0000 1100"
    """

    is_int = False
    if isinstance(arr, int) or isinstance(arr, np.integer):
        arr = np.array([arr])
        is_int = True

    vals = []
    for n in arr:
        strr = format(np.uint32(n), '032b')
        out = []
        for i in range(0, 32, 4):
            out.append(strr[i:i+4])
        vals.append(" ".join(out))

    if is_int:
        return vals[0]
    else:
        return np.array(vals)


def is_evenly_spaced(y):
    diff = np.diff(y)
    return np.all(diff == diff[0])

    
from typing import Union, Optional, Callable
def load_df_from_pattern(pattern, loader: Optional[Union[str, Callable]]=None, **kwargs) -> pd.DataFrame:
    """Load a set of files whose paths match pattern

    This only works for local files. Results are 
    stored in a dataframe

    Inputs
    --------
    pattern (str)
        A string that is globbed to find all appropriate files on disk
    loader (func, or string)
        Function that loads individual files. Must return a dataframe.
        Eg pd.read_csv, pd.read_parquet, etc
        If this is a string (eg csv), or None, function will try to guess the
        correction loading function. 

    All optional inputs are passed to `loader`

    Returns
    ----------
    A Pandas dataframe
    """
    #Just because function exists, doesn't mean required packages
    #are installed
    opts = {
        'csv': pd.read_csv,
        'parquet': pd.read_parquet,
        'json': pd.read_json,
        'xls': pd.read_excel,
    }

    add_src = kwargs.pop('source', False)
    flist = glob(pattern)

    if len(flist) == 0:
        raise ValueError("No files found matching %s" %(pattern))

    #Choose a loading function
    if not hasattr(loader, '__call__'):
        if isinstance(loader, str):
            ext = loader
        elif loader is None:
            ext = os.path.splitext(flist[0])[-1]
            ext = ext[1:]  #Remove . at start
        else:
            raise ValueError("loader should be a string or a function")

        try:
            loader = opts[ext]
        except KeyError:
            raise ValueError("Unrecognised file type %s" %(ext))

    def f(fn):
        try:
            df = loader(fn, **kwargs)
        except Exception as e:
            raise IOError("Error parsings %s" %fn) from e

        if add_src:
            df['_source'] = fn
        return df

    # f = lambda x: loader(x, **kwargs)
    dflist = list(map(f, flist))
    return pd.concat(dflist)


def npregex(pattern, array):
    """Search an numpy array of strings to a pattern

    Inputs
    ---------
    pattern
        string or regex object
    array
        Numpy array to search

    Returns
    -----------
    A boolean array of shape equal to pattern
    """

    f = np.vectorize( lambda x: bool(re.search(pattern, x)))
    return f(array)


def npmap(function, *array):
    """Perform a map, and package the result into an np array"""
    return np.array( lmap(function, *array))

def lmap(function, *array):
    """Perform a map, and package the result into a list, like what map did in Py2.7"""
    return list(map(function, *array))    

def orderOfMag(val):
    """Return the order of magnitude of value

    e.g orderOfMag(9999) is 3

    Input
    val (float)

    Return
    (int)
    """

    return int(math.floor(math.log10(math.fabs(float(val)))))



def printse(val, err, sigdig=2, pretty=True):
    """Return a value and its uncertainty in SI format, and
    scientific notation, eg 12.345(67)e7

    Input
    val, err (float) The value and it's associated uncertainty
    sigdig (int)     Number of significant digits to print
    pretty (bool)    If true print the number as 12.345(67).
                     If false, return 12.345 0.0067
                     pretty option not implemented yet

    Returns:
    A string

    Note
    ------
    If your number has no uncertainty, consider printh instead.

    """

    vex=orderOfMag(float(val))
    eex=orderOfMag(float(err))

    val /= 10**vex
    err /= 10**eex

    big = max(vex, eex)
    val /= 10**(big-vex)
    err /= 10**(big-eex)

    strr = printe(val, err, sigdig=sigdig, pretty=pretty)
    if pretty:
        strr = "%se%02i" %(strr, big)
    else:
        tmp = strr.split()
        strr= "%se%02i %se%02i" %(tmp[0], big, tmp[1], big)

    return strr


def printe(val, err, sigdig=2, pretty=True):
    """Return a value and its uncertainty in SI format, 12.345(67)

    Input
    val, err (float) The value and it's associated uncertainty
    sigdig (int)     Number of significant digits to print
    pretty (bool)    If true print the number as 12.345(67).
                     If false, return 12.345 0.0067

    Returns:
    A string

    Note
    ------
    If your number has no uncertainty, consider printh instead.
    """

    val = float(val)
    err = float(err)

    if err == 0:
        strr = "%.6f 0" %( val)
        return strr

    #Significant digits in the value
    sd = int(max(0, sigdig-1-orderOfMag(err)))


    if pretty:
        if err > 1 and err < 10:
            strr = " %.*f(%.1f)" %(sd, val, err)
        else:
            if sd !=0:
                err /= pow(10, orderOfMag(err)-1)
            #err /= pow(10, orderOfMag(err)-2)
            strr = "%.*f(%g)" %(sd,val,round(err))
    else:
        if err > 1 and err < 10:
            strr = " %.*f %.1f" %(sd, val, err)
        else:
            #err /= pow(10, orderOfMag(err)-1)
            #strr = " %.*f(%g)", sd, val, math.rint(err)
            strr = "%.*f %.*f" %(sd,val,sd,err)

    return strr


def printh(num:float, unit:str = None) -> str:
    """Convert a number into a human readable format by adding a suffix

    Like k for thousands, M for millions, etc.

    Input
    -------------
    num
        Number to parse
    unit
        (Optional) Units the number is in like metres or Watts. Beware
        of numbers that already have suffixes like kg or MW.

    Note
    -----
    If you number has an associated uncertainty, consider using printe,
    or printse instead.
    """
    postfix = {
         0:"",
         3:'k',
         6:'M',
         9:'G',
        12:'T',
        -3:'m',
        -6:'μ',
        -9:'n',
       -12:'p',
    }

    if unit is None:
        unit = ""

    order = np.floor(np.log10(num))
    order -= int(order % 3)  #Round to the nearest multiple of three
    modifier = postfix[order]
    val = num * 10**(-order)
    text = "%.3f %s%s" %(val, modifier, unit)
    return text



class Timer(object):
    """Time a section of code.

    Usage:
    ----------------------
    .. code-block:: python

        with Timer("Some text"):
            a = long_computation()
            b = even_longer_computation()

    Prints "Some text" to the screen, then the time taken once the computation
    is complete.
    """

    def __init__(self, msg=None):
        self.msg = msg

    def __enter__(self):
        self.t0 = datetime.datetime.now()
        print (self.msg)

    def __exit__(self, type, value, traceback):
        t1 = datetime.datetime.now()

        if self.msg is None:
            self.msg = "Elapsed Time"

        print ("%s %s" %(self.msg, t1 - self.t0))



def scale_to_range(y, mn, mx):
    """Linearly scale *y* so it's spans the range [mn,mx]"""
    mny = np.min(y)
    mxy = np.max(y)

    assert mny < mxy, "y is single valued"

    out = (y - mny) / (mxy - mny)    #Scale to [0,1]
    out = out * (mx-mn) + mn
    return out


def wmean(values, unc):
    """Computed the weighted average and uncertainty

    Inputs
    ---------
    values
        (array of floats)
    unc
        (array of floats) 1 sigma uncertainty on values

    Returns
    ---------
    (mean, unc), two floats
    """

    assert len(values) == len(unc)
    values = np.array(values)
    unc = np.array(unc)

    #Need a least one positive value and no negative values
    assert np.any(unc > 0)
    assert np.all(unc >= 0)

    weights = 1 / unc**2
    numer = np.sum(values * weights)
    denom = np.sum(weights)

    wmean = numer / denom
    wunc = 1/np.sqrt(denom)

    return wmean, wunc


#
# Decorators
#


def timer(func):
    """A decorator to time your function"""
    def wrapperForFunc(*args, **kwargs):
         with Timer(func.__name__):
             return func(*args, **kwargs)

    return wrapperForFunc


