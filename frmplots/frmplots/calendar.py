"""Plots for displaying information where location in the calendar is important
"""

import matplotlib.colors as mcolor
import matplotlib.pyplot as plt
import matplotlib as mpl

from pprint import pprint
import pandas as pd
import numpy as np

import frmplots.plots as fplots

def imshow_year(date, value, cmap='Blues', norm=None, fmt="%g", vmin=None, vmax=None, ncolor=7, fontsize=6, drop_duplicates=False):
    """
    Show an array of one value per year as a heatmap.

    Inputs
    -----------
    date
        Dates to plot
    value
        Values associated with each date

    Optional Inputs
    -----------------
    cmap
        (str or plt.cm.cmap object) Colormap to use.
    norm
        (mcolor.Normalize) How to map numbers to the range 0 to 1 to be used by the colormap
    fmt
        (str) Format string for annotating each cell with its value
    vmin, vmax
        (float) If `norm` is not specified, use vmin and vmax to set the min and max values
        for a linear normalisation. Default is to use the full range of `value`
    ncolor
        (int) Number of discrete colours in the colormap
    fontsize
        (int) Size of font to annotate each date
    drop_duplicates
        (bool, default False). If True, drop duplicates days. Use with caution

    Returns
    ----------
    grid (2d np array)
        grid of values
    dom, moy
        (1d np array) day of month and month of year
    cb
        The colorbar object

    Notes
    ----------
    Days with values below vmin are highlighted in red, those above vmax are highlighted in purple
    """
    norm = norm or mcolor.Normalize(vmin, vmax)

    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap].resampled(ncolor)

    #Create the dataframe
    df = pd.DataFrame()
    df['date'] = pd.to_datetime(date)
    df['month'] = df.date.dt.month
    df['dom'] = df.date.dt.day
    df['val'] = value

    #Sanity check
    if np.any(df.duplicated(['month', 'dom'])):
        if drop_duplicates:
            df = df.drop_duplicates(['month', 'dom'])
        else:
            raise ValueError("Duplicate dates found")

    #Create the grid
    grid = df.pivot(index="month", columns="dom", values="val")
    x = np.array(grid.columns)
    y = np.array(grid.index)
    X, Y = np.meshgrid(x, y)

    #plot the grid
    plt.pcolormesh(X, Y, grid, cmap=cmap, norm=norm, ec='#AAAAAA', linewidth=.5)
    fplots.label_grid(grid, x, y, fmt, cmap, norm, fontsize=fontsize)

    #Annotate
    plt.tick_params(axis='y', which='minor', length=0)  #Turn off y-subticks
    ax = plt.gca()
    ax.set_yticks([2,4,6,8,10,12])
    ax.set_yticklabels("Feb Apr Jun Aug Oct Dec".split())
    plt.xlabel("Day of Month")
    plt.ylabel("Month")
    cb = plt.colorbar()

    #Grey out non-days
    for m in [2,4,6,9,11]:
        ax.add_patch( plt.Rectangle( (31-.5, m-.5), height=1, width=1, color='lightgrey', zorder=100 ))
    ax.add_patch( plt.Rectangle( (30-.5, 2-.5), height=1, width=1, color='lightgrey', zorder=100 ))

    #Highlight values above vmax
    idx = grid > norm.vmax
    x = X[idx]
    y = Y[idx]
    for c, r in zip(x, y):
        ax.add_patch(plt.Rectangle((c-.5, r-.5), height=1, width=1, color="none", ec='m', lw=3))

    #Highlight values below vmin
    idx = grid < norm.vmin
    x = X[idx]
    y = Y[idx]
    for c, r in zip(x, y):
        ax.add_patch(plt.Rectangle((c-.5, r-.5), height=1, width=1, color="none", ec='r', lw=3))

    return grid, x, y, cb


def monthly_plot(df, *args, plotter=None, datecol='date', ycol='val', **kwargs):
    """Plot a 4x3 grid of time-series, one per calendar month

    Useful to track seasonal changes in daily time-series

    Inputs
    --------
    df
        (DataFrame)
    plotter
        (Callable) A function to plot a month's worth of data. See below
    datecol
        (str) Name of column in `df` that contains the data information
    ycol
        (str) Name of column in `df` containing values to plot

    All other arguments are passed to the `plotter` function

    Returns
    ----------
    A dictionary of axis objects

    Notes
    -----
    * If no value passed for `plotter`, then `default_monthly_plotter` is called.
    This function serves as the reference implmentation.

    * The signture to the `plotter` function is
        * df (the dataframe)
        * datecol
        * args and kwargs.
    If ycol is set in the call to the parent function, it is passed as kwargs.
    """

    plotter = plotter or default_monthly_plotter
    kwargs['ycol'] = ycol

    df[datecol] = pd.to_datetime(df[datecol])
    df = df.sort_values(datecol)
    year = df[datecol].dt.year
    month = df[datecol].dt.month

    plt.clf()
    # grid = plt.gcf().subplot_mosaic("abc\ndef\nghi\njkl", sharex=True, sharey=True)
    grid = plt.gcf().subplot_mosaic("abcd\nefgh\nijkl", sharex=True, sharey=True)
    kwargs['_plot_grid'] = grid
    grid = df.groupby([year, month]).apply(_plot_single_month, plotter, datecol, *args, **kwargs)

    plt.subplots_adjust(wspace=0)
    return grid.iloc[0]  # Is a dictionary


def _plot_single_month(df, plotter, datecol, *args, **kwargs):
    """Private function of `monthly_plot`"""
    # Pick the correct subplot
    subplots = '_abcdefghijkl'
    grid = kwargs.pop('_plot_grid')
    month = df[datecol].dt.month.iloc[0]
    plt.sca(grid[subplots[month]])

    plotter(df, datecol, *args, **kwargs)
    months = "___ Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    plt.title(months[month])
    return grid


def default_monthly_plotter(df, datecol, *args, **kwargs):
    """Example plotter for `monthly_plot()`

    Is passed a dataframe where every data point shares the same year and month.
    Any return value is ignored.

    Inputs
    -------
    df
        (DataFrame)
    datecol
        (str) Name of column to plot on the xaxis

    All other arguments are passed to the `plt.plot` function
    """

    def plot(df, datecol, *args, **kwargs):
        ycol = kwargs.pop('ycol')
        log = kwargs.pop('logy', False)

        dates = df[datecol]
        frac_hour = dates.dt.hour + dates.dt.minute / 60
        assert np.all(np.diff(frac_hour) >= 0)
        plt.plot(frac_hour, df[ycol], *args, **kwargs)

        if log:
            plt.gca().set_yscale('log')

    day = df[datecol].dt.day
    df.groupby(day).apply(plot, datecol, *args, **kwargs)
