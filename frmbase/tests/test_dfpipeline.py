from ipdb import set_trace as idebug
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
import numpy as np
import pytest

import frmbase.dfpipeline as dfp

def test_settimezone():
    df = pd.DataFrame()
    df['naive'] = pd.date_range("2011-09-18", "2011-09-19", freq='1H')
    df['eastern'] = pd.date_range("2011-09-18", "2011-09-19", freq='1H')
    df['eastern'] = df.eastern.dt.tz_localize("UTC")

    print(df)
    with pytest.raises(TypeError):
        task = dfp.SetTimeZone('naive', 'Europe/Dublin').apply(df)

    df2 = dfp.SetTimeZone('naive', 'Europe/Dublin', "US/Eastern").apply(df)
    assert df2.naive.dt.tz.zone == "Europe/Dublin"

    df2 = dfp.SetTimeZone('eastern', "US/Eastern").apply(df)
    assert df2.eastern.dt.tz.zone == "US/Eastern"
    t0 = df2.eastern.iloc[0]
    assert t0.day == 17
    assert t0.hour == 20
