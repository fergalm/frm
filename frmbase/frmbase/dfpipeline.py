from ipdb import set_trace as idebug
import pandas as pd
import numpy as np

from glob import glob
import warnings
import re
import os

"""
Dataframe pipeline is a way to construct a sequence of operations that
modify a dataframe in a clean fashion

"""

# Stratos is a more complicated pipeline package that allows
# branching pipelines (something runPipeline does not. Use it
# if it's available on this system, but don't crash if it isn't
try:
    from stratos.task import Task
except ImportError:
    #Define a stub class
    class Task:
        pass 
    
    
from typing import Optional

class AbstractStep(Task):
    pass

    def func(self, df:Optional[pd.DataFrame])-> pd.DataFrame:
        return self.apply(df)

    def __str__(self):
        classname = str(self.__class__)[:-2].split()[-1][1:]
        strr = f"<{classname} "
        for k in self.__dict__:
            strr += f"{k}={self.__dict__[k]}, "
        strr += ">"
        return strr
    

def runPipeline(tasks, df=None):
    """ """
    for i, t in enumerate(tasks):
        try:
            df = t.apply(df)
        except Exception as e:
            try:
                msg = "Error on step %i (%s). Error is: %s" % (i, t, e.args[0])
            except IndexError:
                msg = "Error on step %i (%s). No Error msg returned" % (i, t)
            e.args = (msg,)
            raise e

    return df


def pipelineToStrings(pipeline):
    """Convert a pipeline to a list of strings"""
    strs = list(map(str, pipeline))
    return strs
    #return "\n".join(strs)


def pipelineAsString(pipeline):
    """Create a string describing the steps in a pipeline

    Useful for documenting the operations applied to a dataframe

    Inputs
    ----------
    pipeline
        (list) A list of objects AbstractStep (or derived) classes

    Returns
    --------
        A string
    """
    out = []
    for p in pipeline:
        out.append(str(p))
    return "\n".join(out)


class ApplyFunc(AbstractStep):
    def __init__(self, col, func, replace=True):
        self.col = col
        self.func = func
        self.replace = replace

    def apply(self, df:pd.DataFrame):
        if self.col in df.columns and not self.replace:
            raise ValueError(
                f"Column {self.col} already exists in dataframe. Set replace=False to overwrite"
            )

        series = df.apply(self.func, axis=1)
        df[self.col] = series
        return df


class AssertColExists(AbstractStep):
    def __init__(self, *cols):
        warnings.warn("Use dfverify.VerifyColExists", DeprecationWarning, stacklevel=2)
        self.cols = cols

    def apply(self, df:pd.DataFrame):
        if set(df.columns) >= set(self.cols):
            return df

        missing = set(self.cols) - set(df.columns)
        msg = "Some required keys missing from dataframe. Keys %s\nmissing from\n%s" % (
            missing,
            df.columns,
        )
        raise KeyError(msg)


class AssertNotEmpty(AbstractStep):
    def __init__(self):
        warnings.warn("Use dfverify.VerifyNotEmpty", DeprecationWarning, stacklevel=2)

    def apply(self, df:pd.DataFrame):
        assert len(df) > 0, "No rows in dataframe"
        assert len(df.columns) > 0, "No columns in dataframe"
        return df


class Copy(AbstractStep):
    def apply(self, df:pd.DataFrame):
        return df.copy()


class Custom(AbstractStep):
    """Do something not captured by other tasks"""

    def __init__(self, action):
        self.action = action

    def apply(self, df:pd.DataFrame):
        df = df.reset_index(drop=True)
        df = eval(self.action)
        return df


class Debugger(AbstractStep):
    """Insert this step into your pipeline to manually examine the dataframe
    
    The debugger is only invoked if `action` evaluates to True, which it does
    by default.
    """
    def __init__(self, action="True"):
        self.action = action 

    def apply(self, df:pd.DataFrame):
        if eval(self.action):
            idebug()
        return df


class DropDuplicates(AbstractStep):
    def __init__(self, *cols):
        if len(cols) == 0:
            cols = None
        self.cols = cols


    def apply(self, df:pd.DataFrame):
        return df.drop_duplicates(self.cols, keep="first").copy()


class DropCol(AbstractStep):
    def __init__(self, *cols_to_remove, halt_on_missing=True):
        if isinstance(cols_to_remove[0], list):
            cols_to_remove = cols_to_remove[0]
            
        self.cols = cols_to_remove
        self.halt_on_missing = halt_on_missing

    def apply(self, df:pd.DataFrame):
        #If halt_on_missing is False, silently ignore cases where
        #the column we want to drop doesn't exist.
        #Implemented by culling requested cols to only those that exist in df
        cols = self.cols
        if self.halt_on_missing is False:
            cols = set(cols) & set(df.columns)

        return df.drop(list(cols), axis=1)


class DropNan(AbstractStep):
    def __init__(self, how='any'):
        self.how = how

    def apply(self, df:pd.DataFrame):
        return df.dropna(axis=0, how=self.how)


class Filter(AbstractStep):
    def __init__(self, predicate, copy=False):
        self.predicate = predicate
        self.copy = copy

    def apply(self, df:pd.DataFrame):
        """Eg to search df['x'] > 100
        Do Filter('x > 100').apply(df)
        """

        predicate = parsePredicate(df.columns, self.predicate)
        idx = eval(predicate)
        df = df[idx]

        if self.copy:
            df = df.copy()
        return df


class GroupApply(AbstractStep):
    """Apply df.groupby(col).apply(func)"""
    def __init__(self, groupbyCol, func, *args, **kwargs):
        self.col = groupbyCol
        self.func = func 
        self.args = args 
        self.kwargs = kwargs

    def apply(self, df:pd.DataFrame):
        gr = df.groupby(self.col)
        drop = self.kwargs.pop('drop', True)
        return gr.apply(self.func, *self.args, **self.kwargs).reset_index(drop=drop)


class GroupBy(AbstractStep):
    def __init__(self, groupbyCol, predicate):
        self.col = groupbyCol
        self.predicate = predicate

    def __repr__(self):
        return f"<frm.dfpipeline.Groupby({self.col}, {self.predicate})>"

    def apply(self, df:pd.DataFrame):
        gr = df.groupby(self.col)
        # predicate = parsePredicate(df.columns, self.predicate)
        cmd = f"gr.{self.predicate}"
        series = eval(cmd)
        return series.reset_index(drop=False)


class GroupFilter(AbstractStep):
    """Apply df.groupby(col).filter(func)"""

    def __init__(self, groupbyCol, func):
        self.col = groupbyCol
        self.func = func

    def apply(self, df:pd.DataFrame):
        gr = df.groupby(self.col)
        return gr.filter(self.func)


class Load(AbstractStep):
    """Load a set of files whose paths match pattern

    This only works for local files. Results are
    stored in a dataframe

    This should be perfectly compatible with the old Load task.
    Just in case, I keep a commented version of the old task below
    Inputs
    --------
    pattern (str)
        A string that is globbed to find all appropriate files on disk
    loader (func, or string)
        Function that loads individual files. Must return a dataframe.
        Eg pd.read_csv, pd.read_parquet, etc
        If this is a string (eg csv), or None, function will try to guess the
        corrected loading function.
    add_src
        If True, name of inputfile is added to each dataframe

    All optional inputs are passed to `loader`

    Returns
    ----------
    A Pandas dataframe
    """

    def __init__(self, pattern, loader=None, n=None, add_src=True, **kwargs):
        """
        """
        self.pattern = pattern
        self.loader = loader
        self.add_src = kwargs.pop('source', False)
        self.num = n
        self.kwargs = kwargs
        self.opts = {
            'csv': pd.read_csv,
            'parquet': pd.read_parquet,
            'json': pd.read_json,
            'xls': pd.read_excel,
        }

    def __str__(self):
        classname = str(self.__class__)[:-2].split()[-1][1:]
        strr = f"<{classname} on {self.pattern}>"
        return strr

    def apply(self, df=None)-> pd.DataFrame:
        flist = self.get_filelist(self.pattern)
        loader = self.get_loader(self.loader, flist)

        def load(fn):
            if self.num is not None:
                fn = self.head(fn, self.num)

            try:
                df = loader(fn, **self.kwargs)
            except Exception as e:
                raise IOError("Error parsing %s. %s" % (fn, e)) from e

            if self.add_src:
                df['_source'] = fn
            return df

        dflist = list(map(load, flist))
        return pd.concat(dflist)

    def head(self, fn, num):
        buffer = []
        with open(fn) as fp:
            for i in range(num+1):
                buffer.append(fp.readline())
        
        from io import StringIO 
        fout = StringIO("\n".join(buffer))
        return fout 

    def get_filelist(self, pattern):
        flist = glob(pattern)
        if len(flist) == 0:
            raise ValueError("No files found matching %s" % (pattern))
        return flist

    def get_loader(self, loader, flist):
        if not hasattr(loader, '__call__'):
            if isinstance(loader, str):
                ext = loader
            elif loader is None:
                ext = os.path.splitext(flist[0])[-1]
                ext = ext[1:]  # Remove . at start
            else:
                raise ValueError("loader should be a string or a function")

            try:
                loader = self.opts[ext]
            except KeyError:
                raise ValueError("Unrecognised file type %s" % (ext))
        return loader

    def get_handler(self, filetype):
        handers = dict(
            csv=lambda x: pd.read_csv(x, index_col=0),
            parquet=pd.read_parquet,
            json=pd.read_json,
        )
        return handers[filetype]



# class Load(AbstractStep):
#     """Deprecate, then replace wtih LoadGlob"""
#     def __init__(self, fn):
#         self.fn = fn
#
#     def apply(self, df=None):
#         filetype = self.fn.split(".")[-1]
#
#         # Todo, better error checking
#         func = self.get_handler(filetype)
#         return func(self.fn)
#
#     def get_handler(self, filetype):
#         handers = dict(
#             csv=lambda x: pd.read_csv(x, index_col=0),
#             parquet=pd.read_parquet,
#             json=pd.read_json,
#         )
#         return handers[filetype]


class Pivot(AbstractStep):
    def __init__(self, index, columns, values):
        self.index = index
        self.columns = columns 
        self.values = values 

    def apply(self, df:pd.DataFrame):
        df = df.pivot(index=self.index, columns=self.columns, values=self.values)
        df = df.reset_index()
        return df 


class RenameCol(AbstractStep):
    def __init__(self, mapper):
        self.mapper = mapper

    def apply(self, df:pd.DataFrame):
        return df.rename(self.mapper, axis=1)


class ResetIndex(AbstractStep):
    def __init__(self, drop=True):
        self.drop = drop

    def apply(self, df:pd.DataFrame):
        return df.reset_index(drop=self.drop)


class RoundDate(AbstractStep):
    """ Round a date down to a coarser value,
    e.g, "2011-09-21 19:30" --> 2011-09-21
    """
    def __init__(self, delta, col='date'):
        """
        Inputs
        ------
        delta
            (str or pd.DatetimeOffset) eg. '1H' to truncate to nearest hour,
            'D' to truncate to nearest day. Note rounding to nearest 2nd day
            by `delta='2D'` doesn't work yet.

        col
            (str) Name of column to apply truncation to.
        """
        
        self.delta = delta 
        self.col = col 

    def apply(self, df):
        #df[self.col] = df[self.col].dt.floor(self.delta)
        df[self.col] = df[self.col].dt.to_period(self.delta).dt.start_time
        return df 


class SelectCol(AbstractStep):
    def __init__(self, *cols_to_keep):
        if len(cols_to_keep) == 1 and not isinstance(cols_to_keep[0], str):
            cols_to_keep = cols_to_keep[0]  #We were given a single list
        self.cols = cols_to_keep

    def __repr__(self):
        return f"<frm.dfpipeline.SelectCol({self.cols})>"

    def apply(self, df:pd.DataFrame):
        return df[ list(self.cols)]


class SetCol(AbstractStep):
    """
    Eg

    SetColumn('foo', 'foo*2 + bar').apply(df)
    Is the same as df['foo'] = df['foo'] * 2 + df['bar']

    Anything this class can do, ApplyFunc can also do, but I think this func is faster
    """

    def __init__(self, col, predicate, replace=True):

        if isinstance(predicate, (int, float)):
            predicate = str(predicate)
        self.predicate = predicate
        self.col = col
        self.replace = replace

    def __repr__(self):
        return f"<frm.dfpipeline.SetCol({self.col}, {self.predicate})>"

    def apply(self, df:pd.DataFrame):
        if self.col in df.columns and not self.replace:
            raise ValueError(
                f"Column {self.col} already exists in dataframe. Set replace=False to overwrite"
            )

        predicate = parsePredicate(df.columns, self.predicate)
        #1/0
        df[self.col] = eval(predicate)
        return df


class SetColByFunc(AbstractStep):
    """I think I should find a way to merge this class into SetCol.
    
    Also, there's a better version of this function lieing around somewhere
    """
    def __init__(self, col, func, *args, **kwargs):
        self.func = func 
        self.col = col 
        self.args = args 
        self.kwargs = kwargs 

    def apply(self, df):
        df[self.col] =  self.func(df, *self.args, **self.kwargs)
        return df


class SetColByTest(AbstractStep):
    """Set values in a column based on a conditional

    Example::
        df = pd.DataFrame({
            'a': [1,2,3,4,5]
            'b': [1,2,3,4,5]
        }

        expected = pd.DataFrame({
            'a': [1,2,9,9,9]
            'b': [1,2,3,4,5]
        }

        df = SetColByTest('a', 'b > 2', '9', 'a').apply(df)
        assert df == expected
    """
    def __init__(self, col, test, trueValue, falseValue=None):
        self.col = col
        self.test = test
        self.trueValue = trueValue
        self.falseValue = falseValue

    def apply(self, df: pd.DataFrame):
        predicate = parsePredicate(df.columns, self.test)
        idx = eval(predicate)

        trueValue = eval(parsePredicate(df.columns, self.trueValue))
        if not hasattr(trueValue, '__len__'):
            trueValue = trueValue * np.ones(len(df))

        falseValue = eval(parsePredicate(df.columns, self.falseValue))
        if not hasattr(falseValue, '__len__'):
            falseValue = falseValue * np.ones(len(df))

        col = falseValue.copy()
        col[idx] = trueValue[idx]
        df[self.col] = col
        return df


class SetDayNum(AbstractStep):
    """
    Compute the number of days since some epoch.
    This can be useful when trying to group elements of a timeseries
    by day.

    Caution: This class does NOT respect timezones. So
    2011-09-21 00:00  2011-09-21 00:00-0500 return the same value
    of 15238

    TODO: Unit test!
    """

    def __init__(self, datecol='date', daynumcol='daynum', dtype=float):
        self.daynumcol = daynumcol 
        self.datecol = datecol
        
        assert dtype in [int, float], "Dtype must be either int or float"
        self.dtype = dtype 
    
    def apply(self, df:pd.DataFrame):
        jd0 = 2440587.5  #1970-01-01 00:00
        date = pd.to_datetime(df[self.datecol])
        jd  = pd.DatetimeIndex(date).to_julian_date()
        # jd = date.dt.to_julian_date()
        daynum = (jd - jd0).astype(self.dtype)
        df[self.daynumcol] = daynum
        return df 


class Sort(AbstractStep):
    def __init__(self, col):
        self.col = col 

    def apply(self, df:pd.DataFrame):
        return df.sort_values(self.col)

class ToDatetime(AbstractStep):
    """Convert a column to a datetime"""

    def __init__(self, col, *args, **kwargs):
        self.col = col
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return f"<frm.dfpipeline.ToDatetime({self.col})>"

    def apply(self, df:pd.DataFrame):
        df[self.col] = pd.to_datetime(df[self.col], *self.args, **self.kwargs)
        return df


def parsePredicate(cols, predicate):
    """
    Some example text that I will parse into something that can be evaluated.
    Assume the input dataframe has columns 'x' and 'y'

    'x > 100'
    'x.str[:4] == 'MAST'
    'x > 100 & y < 100'
    'x > 100 or y == 'MAST'
    """
    for c in cols:
        pattern = r"\b(%s)\b" % (c)
        predicate = re.subn(pattern, r"df['\1']", predicate)[0]

    return predicate
