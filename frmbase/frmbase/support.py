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



def load_df_from_pattern(pattern, loader=None, **kwargs):
    """Load a set of files whose paths match pattern

    This only works for local files. Results are 
    stored in a dataframe

    Inputs
    --------
    pattern (str)
        A string that is globbed to find all appropriate files on disk
    loader (func)
        Function that loads individual files. Must return a dataframe.
        Eg pd.read_csv, pd.read_parquet, etc

    All optional inputs are passed to `loader`

    Returns
    ----------
    A Pandas dataframe
    """
    #Just because function exists, doesn't mean required packages
    #are installed
    opts = dict(
        csv=pd.read_csv,
        parquet=pd.read_parquet,
        json=pd.read_json,
        xls=pd.read_excel,
    )

    flist = glob(pattern)

    if len(flist) == 0:
        raise ValueError("No files found matching %s" %(pattern))

    #Choose a loading function
    if loader is None:
        ext = os.path.splitext(flist[0])[-1]
        ext = ext[1:]  #Remove . at start
        try:
            loader = opts[ext]
        except KeyError:
            raise ValueError("Unrecognised file type %s" %(ext))

    f = lambda x: loader(x, **kwargs)
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


def sigma_clip(y, nSigma, maxIter=1e4, initialClip=None):
    """Iteratively find and remove outliers
    Find outliers by identifiny all points more than **nSigma** from
    the mean value. The recalculate the mean and std and repeat until
    no more outliers found.
    Inputs:
    ----------
    y
        (1d numpy array) Array to be cleaned
    nSigma
        (float) Threshold to cut at. 5 is typically a good value for
        most arrays found in practice.
    Optional Inputs:
    -------------------
    maxIter
        (int) Maximum number of iterations
    initialClip
        (1d boolean array) If an element of initialClip is set to True,
        that value is treated as a bad value in the first iteration, and
        not included in the computation of the mean and std.
    Returns:
    ------------
    1d numpy array. Where set to True, the corresponding element of y
    is an outlier.
    """
    #import matplotlib.pyplot as mp
    idx = initialClip
    if initialClip is None:
        idx = np.zeros( len(y), dtype=bool)

    assert(len(idx) == len(y))

    #x = np.arange(len(y))
    #mp.plot(x, y, 'k.')

    oldNumClipped = np.sum(idx)
    for i in range(int(maxIter)):
        mean = np.nanmean(y[~idx])
        std = np.nanstd(y[~idx])

        newIdx = np.fabs(y-mean) > nSigma*std
        newIdx = np.logical_or(idx, newIdx)
        newNumClipped = np.sum(newIdx)

        #print "Iter %i: %i (%i) clipped points " \
            #%(i, newNumClipped, oldNumClipped)

        if newNumClipped == oldNumClipped:
            return newIdx

        oldNumClipped = newNumClipped
        idx = newIdx
        i+=1
    return idx



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


# def cache(func, cache_dir=None):
#     """Decorator to run a function and cache its output.

#     Can be used, eg, for database queries or url queries where
#     the results are not expected to change

#     Notes
#     ----------
#     Can only be used to decorate functions that take a single argument.
#     Should I change that to cached based on the first argument?
#     """

#     if cache_dir is None:
#         cache_dir = os.path.join(os.environ['HOME'], ".cache", "fergal")

#     if not os.path.exists(cache_dir):
#         os.mkdir(cache_dir)

#     def wrapperForFunc(*args, **kwargs):
#         if len(args) > 1 or len(kwargs) > 0:
#             raise ValueError("cache decorator only works for functions with 1 argument")
#         cache_name = "query%s" %( str( hash(args[0]) ) )
#         cache_name = os.path.join(cache_dir, cache_name)

#         #If result is cached, use it
#         if os.path.exists(cache_name):
#             with open(cache_name) as fp:
#                 text = fp.read()
#             return text

#         #Else, query for the result, cache it, then return it.
#         text = func(args[0])
#         with open(cache_name, 'w') as fp:
#             fp.write(text)
#         return text


    return wrapperForFunc


def check_cols_in_df(df, cols):
    """Because I always get this wrong"""
    return check_columns_in_df(df, cols)

def check_columns_in_df(df, cols):
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








##############################################################
#Examples for testing

@timer
def test_example1():
    """Run this from console and you should see the appropriate text being printed"""
    import time
    time.sleep(3)

# @cache
# def _example2(query):
#     return "This is the result of the query as pure text"

def test_example2():
    query = "This is a query string. Maybe it's SQL, maybe it's a URL"
    _example2(query)

    fn = os.path.join(os.environ['HOME'], ".cache", "fergal")
    fn = os.path.join(fn,  "query%s" % (str(hash(query))))
    print (fn)
    assert os.path.exists(fn)

    with open(fn) as fp:
        text = fp.read()

    msg = "Cached text: >>%s<<" %(text)
    assert text == "This is the result of the query as pure text", msg



