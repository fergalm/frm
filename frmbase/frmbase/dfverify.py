from ipdb import set_trace as idebug
import frmbase.dfpipeline as dfp
import numpy as np

"""
sketch code for some data validation pipeline steps.

These steps check that the values in a column in a dataframe are sane, i.e
* Most values are finite
* Most values are in some range 
* Some aggregate value (e.g the mean) is in some acceptable range.

The can be run by passing them to the `runPipline` function in dfpipline

By convention, all verification steps should raise ValueError if they find a problem
"""

class VerifyColExists(dfp.AbstractStep):
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
        raise ValueError(msg)


class VerifyNotEmpty(dfp.AbstractStep):
    def apply(self, df):
        if len(df) == 0:
            raise ValueError("No rows in dataframe")

        if len(df.columns) == 0:
            raise ValueError("No columns in dataframe")
        return df


class VerifyIsFinite(dfp.AbstractStep):
    def __init__(self, *cols, max_frac=None, max_num=None):
        """
        Inputs
        ------
        *cols
            Columns to check
        max_frac
            (float) Allow no more than this fraction of values
            in any one column to be Nan/Inf without raising an error
        max_num
            (int) Allow no more than this number of Nans before raising an error
        """
        self.col_list = cols
        self.max_frac = max_frac
        self.max_num = max_num

        if max_num is None and max_frac is None:
            self.max_num = 0  # Default is to accept no Nans

    def apply(self, df):
        size = len(df)
        msg = []
        for col in self.col_list:
            num = np.sum(~np.isfinite(df[col]))
            # idebug()
            if self.max_num is not None and num > self.max_num:
                msg.append(
                    f"{num} non-finite values found in col '{col}' (limit is {self.max_num})"
                )
            elif self.max_frac is not None and num > self.max_frac * size:
                msg.append(
                    f"{100*num/size:2f}% of values are non-finite in col '{col}'. Limit is {100*self.max_frac}"
                )

        if len(msg) > 0:
            msg = "\n".join(msg)
            raise ValueError(msg)


class VerifyInRange(dfp.AbstractStep):
    def __init__(self, *cols, lwr=-np.inf, upr=np.inf, max_frac=None, max_num=None):
        """
        Inputs
        ------
        *cols
            Columns to check
        lwr, upr
            (float) Maximum and minimum allowed value for each column. Note
            it is not possible to specify different limits for different columns.
        max_frac
            (float) Allow no more than this fraction of values
            in any one column to be out of range without raising an error
        max_num
            (int) Allow no more than this number of values out of range before raising an error

        Notes
        ---------
        * The default range in -infinity to infinity. You must specify at least one bound or
          the verification will always pass
        * Default behaviour is to not accept any values outside the given range. You can override
          that by setting `max_num` or `max_frac`
        """
        self.col_list = cols
        self.lwr = lwr
        self.upr = upr
        self.max_frac = max_frac
        self.max_num = max_num

        if max_num is None and max_frac is None:
            self.max_num = 0  # Default is to accept no Nans

    def apply(self, df):
        size = len(df)
        msg = []
        for col in self.col_list:
            vals = df[col]
            num = np.sum((vals < self.lwr) | (vals > self.upr))
            if self.max_num is not None and num > self.max_num:
                msg.append(
                    f"{num} values out of range ({self.lwr, self.upr}) found in col {col} (limit is {self.max_num}"
                )
            elif self.max_frac is not None and num > self.max_frac * size:
                text = f"{100 * num / size:2d}%% of values"
                text += f"are out of range ({self.lwr, self.upr}) in col {col}. "
                text += f"Limit is {100 * self.max_frac}"
                msg.append(text)

        if len(msg) > 0:
            msg = "\n".join(msg)
            raise ValueError(msg)


class VerifyFuncInRange(dfp.AbstractStep):
    def __init__(self, *cols, func=np.mean, lwr=-np.inf, upr=np.inf):
        """
        Inputs
        ------
        *cols
            Columns to check
        func
            Function to apply to each column
        lwr, upr
            (float) Maximum and minimum allowed value for result for each column. Note
            it is not possible to specify different limits for different columns.
        """
        self.col_list = cols
        self.func = func
        self.lwr = lwr
        self.upr = upr

    def apply(self, df):
        size = len(df)
        msg = []
        for col in self.col_list:
            val = self.func(df[col])
            if not (self.lwr < val < self.upr):
                msg.append(
                    f"Function {self.func} returns value {val} out of range ({self.lwr, self.upr}) for col {col}"
                )

        if len(msg) > 0:
            msg = "\n".join(msg)
            raise ValueError(msg)


class VerifyMeanInRange(VerifyFuncInRange):
    """Convenience function for VerifyFuncInRange"""

    def __init__(self, *cols, lwr=-np.inf, upr=np.inf):
        """
        Inputs
        ------
        *cols
            Columns to check
        """
        VerifyFuncInRange.__init__(self, *cols, func=np.mean, lwr=lwr, upr=upr)


class VerifyMedianInRange(VerifyFuncInRange):
    """Convenience function for VerifyFuncInRange"""

    def __init__(self, *cols, lwr=-np.inf, upr=np.inf):
        """
        Inputs
        ------
        *cols
            Columns to check
        """
        VerifyFuncInRange.__init__(self, *cols, func=np.median, lwr=lwr, upr=upr)


class VerifyRmsInRange(VerifyFuncInRange):
    """Convenience function for VerifyFuncInRange"""

    def __init__(self, *cols, lwr=-np.inf, upr=np.inf):
        """
        Inputs
        ------
        *cols
            Columns to check
        """
        VerifyFuncInRange.__init__(self, *cols, func=np.std, lwr=lwr, upr=upr)


class VerifyMadInRange(VerifyFuncInRange):
    """Convenience function for VerifyFuncInRange
    Checks the median absolute deviation is within bounds
    """

    def __init__(self, *cols, lwr=-np.inf, upr=np.inf):
        """
        Inputs
        ------
        *cols
            Columns to check
        """
        VerifyFuncInRange.__init__(self, *cols, func=self.mad, lwr=lwr, upr=upr)

    def mad(self, vals):
        med = np.median(vals)
        return np.median(np.fabs(vals - med))
