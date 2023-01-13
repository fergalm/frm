import pandas as pd
import numpy as np
import pytest

import frmbase.dfverify as dfv


def load_good_df():
    size = 10
    return pd.DataFrame(
        {
            "a": np.arange(size),
            "b": np.linspace(0, 1, size),
            "c": 4 + 2 * np.random.randn(size),
        }
    )


def load_bad_df():
    df = load_good_df()
    df.loc[3, "a"] = np.nan
    return df


def test_check_nan():
    df = load_good_df()
    dfv.VerifyIsFinite("a", "b", "c").apply(df)

    df = load_bad_df()
    print(df)
    with pytest.raises(ValueError):
        dfv.VerifyIsFinite("a", "b", "c").apply(df)

    dfv.VerifyIsFinite("a", "b", "c", max_num=5).apply(df)
    dfv.VerifyIsFinite("a", "b", "c", max_frac=0.5).apply(df)

    with pytest.raises(ValueError):
        dfv.VerifyIsFinite("a", "b", "c", max_num=0).apply(df)


def test_in_range():
    df = load_good_df()
    dfv.VerifyInRange("a", lwr=0, upr=11).apply(df)
    with pytest.raises(ValueError):
        dfv.VerifyInRange("a", lwr=0, upr=4).apply(df)


def test_mean_in_range():
    df = load_good_df()
    dfv.VerifyMeanInRange('c', lwr=2, upr=5).apply(df)

    with pytest.raises(ValueError):
        dfv.VerifyMeanInRange('c', lwr=-5, upr=-2).apply(df)


def test_mad_in_range():
    df = load_good_df()
    dfv.VerifyMadInRange('c', lwr=1, upr=3).apply(df)

    with pytest.raises(ValueError):
        dfv.VerifyMadInRange('c', lwr=-5, upr=-2).apply(df)
