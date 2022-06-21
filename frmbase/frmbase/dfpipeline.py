from ipdb import set_trace as idebug
import pandas as pd
import numpy as np
import re

"""
Dataframe pipeline is a way to construct a sequence of operations that
modify a dataframe in a clean fashion

"""


class AbstractStep:
    pass

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


def pipelineToString(pipeline):
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

    def apply(self, df):
        if self.col in df.columns and not self.replace:
            raise ValueError(
                f"Column {self.col} already exists in dataframe. Set replace=False to overwrite"
            )

        series = df.apply(self.func, axis=1)
        df[self.col] = series
        return df


class AssertColExists(AbstractStep):
    def __init__(self, *cols):
        self.cols = cols

    def apply(self, df):
        if set(df.columns) >= set(self.cols):
            return df

        missing = set(self.cols) - set(df.columns)
        msg = "Some required keys missing from dataframe. Keys %s\nmissing from\n%s" % (
            missing,
            df.columns,
        )
        raise KeyError(msg)


class AssertNotEmpty(AbstractStep):
    def apply(self, df):
        assert len(df) > 0, "No rows in dataframe"
        assert len(df.columns) > 0, "No columns in dataframe"
        return df


class Copy(AbstractStep):
    def apply(self, df):
        return df.copy()


class Custom(AbstractStep):
    """Do something not captured by other tasks"""

    def __init__(self, action):
        self.action = action

    def apply(self, df):
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

    def apply(self, df):
        if eval(self.action):
            idebug()
        return df


class DropDuplicates(AbstractStep):
    def __init__(self, *cols):
        if len(cols) == 0:
            cols = None
        self.cols = cols


    def apply(self, df):
        return df.drop_duplicates(self.cols, keep="first").copy()


class DropCol(AbstractStep):
    def __init__(self, *cols_to_remove):
        self.cols = cols_to_remove

    def apply(self, df):
        return df.drop(list(self.cols), axis=1)


class Filter(AbstractStep):
    def __init__(self, predicate, copy=False):
        self.predicate = predicate
        self.copy = copy

    def apply(self, df):
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

    def apply(self, df):
        gr = df.groupby(self.col)
        return gr.apply(self.func, *self.args, **self.kwargs).reset_index(drop=True)


class GroupBy(AbstractStep):
    def __init__(self, groupbyCol, predicate):
        self.col = groupbyCol
        self.predicate = predicate

    def __repr__(self):
        return f"<frm.dfpipeline.Groupby({self.col}, {self.predicate})>"

    def apply(self, df):
        gr = df.groupby(self.col)
        # predicate = parsePredicate(df.columns, self.predicate)
        cmd = f"gr.{self.predicate}"
        return eval(cmd)


class GroupFilter(AbstractStep):
    """Apply df.groupby(col).filter(func)"""

    def __init__(self, groupbyCol, func):
        self.col = groupbyCol
        self.func = func

    def apply(self, df):
        gr = df.groupby(self.col)
        return gr.filter(self.func)


class Load(AbstractStep):
    def __init__(self, fn):
        self.fn = fn

    def apply(self, df=None):
        filetype = self.fn.split(".")[-1]

        # Todo, better error checking
        func = self.get_handler(filetype)
        return func(self.fn)

    def get_handler(self, filetype):
        handers = dict(
            csv=lambda x: pd.read_csv(x, index_col=0),
            parquet=pd.read_parquet,
            json=pd.read_json,
        )
        return handers[filetype]


class RenameCols(AbstractStep):
    def __init__(self, mapper):
        self.mapper = mapper

    def apply(self, df):
        return df.rename(self.mapper, axis=1)


class ResetIndex(AbstractStep):
    def apply(self, df):
        return df.reset_index(drop=True)


class SelectCols(AbstractStep):
    def __init__(self, *cols_to_keep):
        self.cols = cols_to_keep

    def __repr__(self):
        return f"<frm.dfpipeline.SelectCol({self.cols})>"

    def apply(self, df):
        return df[ list(self.cols)]


class SetCol(AbstractStep):
    """
    Eg

    SetColumn('foo', 'foo*2 + bar').apply(df)
    Is the same as df['foo'] = df['foo'] * 2 + df['bar']

    Anything this class can do, ApplyFunc can also do, but I think this func is faster
    """

    def __init__(self, col, predicate, replace=True):
        self.predicate = predicate
        self.col = col
        self.replace = replace

    def __repr__(self):
        return f"<frm.dfpipeline.SetCol({self.col}, {self.predicate})>"

    def apply(self, df):
        if self.col in df.columns and not self.replace:
            raise ValueError(
                f"Column {self.col} already exists in dataframe. Set replace=False to overwrite"
            )

        predicate = parsePredicate(df.columns, self.predicate)
        df[self.col] = eval(predicate)
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
    
    def apply(self, df):
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

    def apply(self, df):
        return df.sort_values(self.col)

class ToDatetime(AbstractStep):
    """Convert a column to a datetime"""

    def __init__(self, col, *args, **kwargs):
        self.col = col
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return f"<frm.dfpipeline.ToDatetime({self.col})>"

    def apply(self, df):
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
