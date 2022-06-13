

def trailing_window(df, function, *args, window_size='1D', dateCol='date', freq='1D', **kwargs):
    """Applies `function` to subsets of dataframe using a trailing window

    Example
    -------
    Suppose you have a dataframe of hourly price, with two columns, hour and
    price. To claculate the average weekly price you might do::

        func = lambda df: pd.Series(np.mean(df.price))
        trailing_window(df, func, window_size='7D', freq='1D', dateCol='hour')

    This will produce 

    You're supposed to be able to do this with pd.rolling, but rolling
    will only work on a single column
    """


    start = np.min(df[dateCol])
    end = np.max(df[dateCol])
    dates = pd.date_range(start, end, freq=freq)
    window_size = pd.to_timedelta(window_size)


    dflist = []
    for d in tqdm(dates):
        idx = (d - window_size < df[dateCol]) & (df[dateCol] <= d)
        res = function(df[idx])
        dflist.append(res)
    return pd.concat(dflist)
