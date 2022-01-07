# -*- coding: utf-8 -*-
"""
Sketch of an idea for a timezone class


Requirements
    o Get and set time in preferred type by says
        obj = LocalTime("2019-12-10", tz="Europe/Dublin")
        unixtime = obj.asUnixtime()

    o Work with arrays of times
    o Deal seamlessly with pandas datetime types
    o Have a sensible way of returning formatted strings or datetime types
        as needed.
    o Deal with naive timezones (e.g "8am here" regardless of where in the
      world here is)

Desiridata
    o Do conversions like LocalTime(Utc)
@author: fergal
"""

from ipdb import set_trace as idebug
from pdb import set_trace as debug
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


ISO_FMT = '%Y-%M-%D'

class Time():
    def __init__(self):
        self.unixtime = None

    def setUnixtime(self, unixtime):
        self.unixtime = unixtime

    def asUnixtime(self):
        return self.unixtime

    def setUtc(self, value, fmt=ISO_FMT):
        self.unixtime = pd.to_datetime(value, format=fmt).astype(np.int64)/1e6

    def asUtc(self, fmt=ISO_FMT):
        return pd.to_datetime(self.unixtime, unit='sec').strftime(ISO_FMT)

    def setLocalTime(self, value, tz=None, fmt=ISO_FMT):
        pass

    def asLocalTime(self, fmt=ISO_FMT):
        pass

    def setDatetime(self, obj):
        pass

    def asDatetime(self, obj):
        pass
        #Must return a tz-aware dt object in UTC
        #The localtime class can extend this behaviour to return
        #A datetime in the localtime

class Unixtime(Time):
    def __init__(self, x):
        if isinstance(x, Time):
            self.setUnixtime( x.asUnixtime() )
        else:
            self.setUnixtime(x)

    def __call__(self):
        return self.asUnixtime()


class Utc(Time):
    def __init__(self, value, fmt=ISO_FMT):
        self.setUtc(value)

    def __call__(self):
        return self.asUnixtime()

class LocalTime(Time):
    def __init__(self, value, tz=None, fmt=ISO_FMT):
        self.setLocalTime(value, tz, fmt)

    def __call__(self):
        return self.asLocalTime()
