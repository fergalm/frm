"""
Created on Mon Nov 28 14:15:53 2016
Some common plots I like to make
@author: fergal

Some tips I haven't written to functions yet
---------------------------------------------

The better way to do subplots

::
    grid = plt.gcf().subplot_mosaic("abc\def", sharex=True)
    plt.sca(grid['a'])

How to have dates on the colour bar axis

::
    from matplotlib.dates import DateFormatter,date2num
    dates = pd.date_range('2020-01-01', '2020-12-01', freq='D')
    date_num = list(map(date2num, dates))
    plt.scatter(x, y, c=date_num)
    plt.colorbar(format=DateFormatter("%Y-%m-%d"))
    
    
    
How to change the date format in the toolbar of an interactive plot. UPDATE: This is now implemented
in `fix_date_labels()` below

::
    import matplotlib.dates as mdate
    def foo(x):
        return mdate.num2date(x).isoformat()[:19]


    def main():
        ...
        ax = plt.gca()
        ax.fmt_xdata = foo
        
There is another function pointer called fmt_ydata for the y data format
An exercise is to figure out is
how to check if the x axis is plotting dates or not

"""
import gzip


import matplotlib.collections as mcollect
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as meffect
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.transforms as mtrans
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib as mpl

import pandas as pd
import numpy as np
import pickle
import copy

from typing import Iterable

from . import norm as fnorm
from . galaxyplot import galaxyPlot
from . import plotstyle

DEFAULT_CMAP = plt.cm.viridis
add_watermark = plotstyle.add_watermark
mark_as_draft = plotstyle.mark_as_draft

def add_figure_labels(xlabel, ylabel, **kwargs):
    """Add axis labels to a figure composed of subplots.

    Inputs
    ----------
    xlabel
        (str) Label for x axis
    ylabel
        (str) Label for y axis
    
    All other options are passed to plt.xlabel and plt.ylabel.
    Use plt.suptitle to put a title over the subplots
    """

    fig = plt.gcf()
    ax = fig.gca()
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, which='both')
    plt.xlabel(xlabel, **kwargs)
    plt.ylabel(ylabel, **kwargs)  
    plt.sca(ax)


def annotate_histogram(vals, bins, **kwargs):
    """Prints the number of elements in a histogram on top of each bar.

    Slightly experimental. Only works for vertical histograms. Behaviour
    for multiple bars of the same height not well tested. 

    Example
    ------------
    ::

        bins, vals, _ = plt.hist(data)
        annotate_histogram(bins, vals, color='red', fontsize=14)

    Inputs
    ---------
    vals, bins
        The two arrays returned by `np.histogram`, or the first two values
        returned by matplotlib's `hist` command
    
    Optional Arguments
    ---------------------
    fmt
        (string) Format string for the numbers. Default is '%g'
    ha
        (string) Horizontal alignment of text. Defaults to center.
    above=True
        (bool) Whether annotations should be above or below the point 
    offset
        (float) How far above (or below) top of line to place the label.
        A value of 0 means no offsetting. A value of 1 offsets by the 
        difference between the heights of the max and in bars.
    jitter
        (float) How much to offset adjacent labels in the y direction.
        Prevents labels from over writing each other. Units the same
        as `offset`

    All other arguments are passed to `plt.text`

    """
    #Define some default values
    fmt = kwargs.pop('fmt', '%g')
    ha = kwargs.pop('ha', 'center')
    offset = kwargs.pop('offset', 1)
    offset_sign = 2* kwargs.pop('above', True) - 1
    jitter = kwargs.pop('jitter', .04)
    fontsize = kwargs.pop('fontsize', 12)

    va = kwargs.pop('va', None)
    if va is None:
        if offset_sign > 0:
            va = 'bottom'
        else:
            va = 'top'

    if len(bins) == len(vals):
        locs = bins
    elif len(bins) == len(vals) + 1:
        locs = bins[:-1] + .5* np.diff(bins)

    # scale = np.max(vals) - np.min(vals)
    # offset = (offset * scale * offset_sign) + jitter * scale
    # sign = -1
    # old = 0
    # for xpos, val in zip(locs, vals):
    #     if val <= 0:
    #         continue 
    # 
    #     ypos = val + offset
    #     if np.fabs(ypos - old) < offset:
    #         ypos += sign * offset 
    #         sign *= -1
        
    # old = ypos 
    # plt.text(xpos, ypos + offset, fmt%(val), ha=ha, va=va, **kwargs)

    import matplotlib.transforms as mtransforms
    pointsPerInch = 72
    rnd = np.random.rand()
    offset_points = offset_sign * (offset + jitter*rnd) * fontsize/pointsPerInch
    trans = mtransforms.offset_copy(plt.gca().transData, fig=plt.gcf(), y=offset_points, units='inches')
    # print(offset_points)
    # import ipdb; ipdb.set_trace()

    for xpos, val in zip(locs, vals):
        rnd = np.random.rand()
        offset_points = offset_sign * (offset + jitter*rnd) * fontsize/pointsPerInch

        trans = mtransforms.offset_copy(plt.gca().transData, fig=plt.gcf(), y=offset_points, units='inches')

        plt.text(xpos, val, fmt%(val), ha=ha, va=va, transform=trans, fontsize=fontsize, **kwargs)


def barcode(x, clr='C0', lw=.5, alpha=.4, ymin=0, ymax=.1):
    """Draw a whisker at the bottom of the plot for each value of x

    See also plt.eventplot()

    This is useful if your points are clustered together too closely to be seen, but for 
    some reason the densityPlot() isn't suitable for your plot.

    This implementation uses a LineCollection class to speed rendering.
    Inputs
    ---------
    x
        (1d np array) List of x values to plot
    
    Optional Inputs
    -------------------
    clr
        (str) Colour of lines, in a format matplotlib expects 
    lw
        (float) Width of lines
    alpha
        (float) transparency
    ymin
        (float) Bottom of whisker in axis units 
    ymax
        (float) Top of whisker in axis units.

    Returns
    ---------
    **None**
    """
    #Matplotlib expects the data in this odd format. [row, line_elt, xOrY]
    data = np.empty( (len(x), 2, 2) )
    data[:,0,0] = x
    data[:,0,1] = ymin
    data[:,1,0] = x
    data[:,1,1] = ymax

    ax = plt.gca()
    trans = ax.get_xaxis_transform(which='grid')
    collection = mcollect.LineCollection(data, linewidths=lw, colors=clr, alpha=alpha, transform=trans)
    ax.add_collection(collection)


def borderplot(x, y, *args, **kwargs):
    """Plot x against y, and show histograms in x and y at the same time.

    Inputs
    ---------
    x, y
        (float) Data to plot

    Optional Inputs
    ----------------
    xbins, ybins
        (int, array, str) Binning specification for the histogram
        in the x and y directions. Is passed directly to `plt.hist`
    color
        (matplotlib colour specification
    func
        (function) What function is used to draw main panel. Default is plt.plot

    All other options are passed to `plt.plot`

    Returns
    ---------
    ax
        Axis handle for main plot
    xh
        Axis handle for x-axis histogram
    yh
        Axis handle for y-axis histogram
    """

    xbins = kwargs.pop('xbins', 10)
    ybins = kwargs.pop('ybins', 10)
    color = kwargs.pop('color', 'C0')
    func = kwargs.pop('func', plt.plot)

    #Set up subplots
    i0, j0 = 2, 7
    gs = GridSpec(9,9)
    ax = plt.subplot(gs[i0:, :j0])
    func(x, y, *args, **kwargs)

    xh = plt.subplot(gs[:i0, :j0], sharex=ax)
    plt.hist(x, bins=xbins, color=color)
    xh.xaxis.tick_top()

    yh = plt.subplot(gs[i0:, j0:], sharey=ax)
    plt.hist(y, bins=ybins, color=color, orientation='horizontal')
    yh.yaxis.tick_right()

    plt.sca(ax)
    return ax, xh, yh


def create_colorbar(cmap, norm=None, vmin=None, vmax=None, dates=False, fmt=None) -> mpl.colorbar.Colorbar:
    """Create a colorbar when matplotlib refuses to

    Creating a colorbar is usually as simple as calling `plt.colorbar()`. Occasionaly
    this fails because you are trying to do something clever, and fooled matplotlib
    so it doesn't understand what the colour range is. In such situations, this
    function comes to the rescue

    Inputs
    ---------
    * cmap
        * Colormap to use. A `matplotlib.colorbar.Colorbar()` object

    Optional Inputs
    ------------------
    * norm
        A `matplotlib.colors.Normalize()` object that maps values to the range 0..1
        If not supplied you must supply `vmin` and `vmax`
    * vmin, vmax
        (floats) The min and max values of the colorbar. If `norm` is supplied, these
        values are ignored.
    * dates
        (boolean) If **True**, the tick labels are formated as dates. `vmin` and `vmax`,
        if used, are aslo assumed to be datetime objects.
    * fmt
        A `matplotlib.ticker.Formatter` object used to format the tick labels. Default
        is the scalar formater, which works well in most cases, or a DateFormatter if
        `date=True`

    Returns
    -----------
    A `matplotlib.colors.Colorbar()` object.

    """

    if norm is None and (vmin is None or vmax is None):
        raise ValueError("Must specify either a normalisation object or a value range")

    if dates:
        vmin = mdates.date2num(vmin)
        vmax = mdates.date2num(vmax)
        fmt = fmt or mdates.DateFormatter("%Y-%m-%d")
    norm = norm or Normalize(vmin, vmax)

    cmappable = ScalarMappable(norm=norm, cmap=cmap)
    return plt.colorbar(cmappable, ax=plt.gca(), format=fmt)


def densityPlot(x,y, xBins, yBins, *args, **kwargs):
    """
    Plot x against y as points where points are sparse, but as shaded region when crowded.

    When you have a lot of points to plot in a 2d graph, it is often difficult
    to decide the best way to plot it. If you are interested in the typical
    behaviour, an image of a 2d histogram is best, showing the density of
    points in different regions. If you're interested in the outliers it
    is better to plot each point individually.

    ``densityPlot`` combines the best of both worlds, showing indivdual
    outliers overlaid on a density plot of a 2d histogram.


    Inputs:
    -------
    x, y
        (np.array) Arrays of values to plot
    xbins, yBins
        What bins to use for density 2d histogram. Can be ints or arrays
        as allowd by ``np.histogram2d``

    Optional Inputs:
    ---------------
    threshold
        Individual data points are plotted if there are fewer than this
        number in a 2d histogram bin.
    cmap
        Which colour map to use for histogram. Default: mp.cm.rainbow
    aspect
        Default value scales axes to give square bins.

    All other optional arguments are passed to ``mp.plot()``


    Returns:
    --------
    **None**

    Notes:
    -----
    * Inspired by a similar IDL function written by Micheal Strauss in IDL
    * Won't work with postscript output
    * Best results are obtained by using a colour map similar to the
      colour of the points your using, e.g mp.cm.Greens and green circles.

    TODO:
    ---------
    Instead of having a wantLog, I should be able to pass in a normalisation
    object.
    """
    threshold = kwargs.pop('threshold', 10)
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    norm = kwargs.pop('norm', fnorm.DiscreteNorm(7, vmin, vmax))
    cmap = kwargs.pop('cmap', DEFAULT_CMAP)

    ls = kwargs.pop('ls', "none")
    ls = kwargs.pop('linestyle', ls)
    marker = kwargs.pop('marker', '.')
    # aspect = kwargs.pop('aspect', 'auto')   #Equal gives square bins
    zorder = kwargs.pop('zorder', 0)

    cmap = copy.copy(cmap)  #I'm annoyed with mpl for making me do this
    cmap.set_under(alpha=0)
    plt.plot(x, y, zorder=zorder-1, marker=marker, ls=ls, *args, **kwargs)
    plt.hist2d(x, y, bins=[xBins, yBins], cmap=cmap, norm=norm)
    plt.clim(vmin=threshold)




def fix_date_labels():
    """Format date strings in x-axis label so they're easier to read.

    Also fixes the box at top right of the interactive
    window that reports the cursor location
    
    Note
    ----------
    To get minor ticks once per day add the following code
    to your function after you call fix_date_labels()::

        import matplotlib.ticker as mticker
        ax.xaxis.set_minor_locator(mticker.IndexLocator(1, 0))

    This only works if you want one tick per day, and isn't helpful
    for much longer or shorter date spans

    """
    plt.gcf().autofmt_xdate()

    #If the xaxis range is of order 1 month, set the major ticks
    #to be once weekly
    locator = mdates.AutoDateLocator(minticks=2, maxticks=7, interval_multiples=False)
    locator.intervald['DAILY'] = 7
    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)

    #Fix the text box at top right of plot to show
    #dates correctly. Note, for images, use plt.gca().format_coord
    plt.gca().fmt_xdata = _format_dates_in_toolbar

def _format_dates_in_toolbar(x):
    """Fix the toolbar tooltip that shows the x and y values of the cursor so
    that it displays full dates correctly.

    Private function of fix_date_labels
    """
    return mdates.num2date(x).isoformat()[:19]


def label_grid(grid: np.ndarray, xPos: Iterable, yPos: Iterable, fmt:str, cmap, norm=None, **kwargs):
    """
    Annotate each grid cell of an image with the underlying value.

    Helps make heatmaps easier to interpret.

    Not tested.

    Inputs
    -------
    grid
        (2d np array) Data used in the heatmap image
    xPos
        (1d np array) Positions of the x-values of the centres of the grid cells.
    yPos
        (1d np array) Positions of the y-values of the centres of the grid cells.
    fmt
        Format string for label, eg "0.2f"
    cmap
        Colormap used by imshow to map values to colours
    norm
        Normalization object used by imshow to map values to the range 0..1. Defaults
        to a linear norm.

    Returns
    ----------
    **None**

    """
    norm = norm or mcolors.Normalize()
    grid = np.array(grid)

    for i, x in enumerate(xPos):
        for j, y in enumerate(yPos):
            val = grid[j, i]
            clr = cmap(norm(val))
            label_grid_square(x, y, val, fmt, clr, **kwargs)


def label_grid_square(col: float, row:float, val:float, fmt:str, clr, **kwargs):
    """
    Add a text label at a specific position in a plot

    Intended to be used by label grid, but pulled out as a separate function
    in case it is useful standalong

    Inputs
    ----------
    col, row
        Location of grid cell
    val
        Numerical value of grid cell
    fmt
        Format string for label, eg "0.2f"
    clr
        Iterable with 3 or 4 elements reprsenting the colour of that cell

    All optional arguments are passed to plt.text

    Returns
    ----------
    **None**
    """
    textcolor='w'
    if np.prod(clr) > .1:
        textcolor='k'

    text = fmt %(val)
    plt.text(col, row, text, \
        va="center",
        ha="center",
        bbox=None,
        color=textcolor,
        **kwargs,
    )


def load_figfile(filename):
    """Load a figure from a pickle, similar to Matlab's .fig format"""
    fig = plt.gcf()
    with open(filename, 'rb') as fp:
        bytes = fp.read()
        try:
            bytes = gzip.decompress(bytes)
        except gzip.BadGzipFile:
            # Older versions of save_figfile did not compress their data
            pass

        fig = pickle.loads(bytes)
    return fig


def mark_indices(x_data, y_axis, *args, **kwargs):
    """Place points with x-position referenced to data, but y-coords
    references to axis coords. This provides a kind of heads-up-display
    for when a binary event occurs (e.g a flag being raised on bad data)
    """
    if not hasattr(y_axis, 'len'):
        y_axis = y_axis + np.zeros_like(x_data, dtype=float)
        
    ax = plt.gca()
    trans = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    return ax.plot(x_data, y_axis, *args, transform=trans, **kwargs)



def mark_weekends(timestamps, tz='UTC'):
    """Plot grey bars to indicate night time in the central timezone"""
    t1 = min(timestamps)
    t2 = max(timestamps)

    t1 = "%04i-%02i-%02i 00:00" %(t1.year, t1.month, t1.day)  #10pm
    t2 = "%04i-%02i-%02i 00:00" %(t2.year, t2.month, t2.day)  #10pm
    # print (t1, t2)

    day_start = pd.date_range(start=t1, end=t2, freq='D')
    day_start = day_start.tz_localize(tz)

    #Edge case: First day of data set is Sunday
    day = day_start[0]
    if day.dayofweek == 6:
        delta = pd.to_timedelta("1D")
        handle = plt.axvspan(day-delta, day+delta, color='b', alpha=.1)
        plt.xlim(xmin=min(timestamps))

    for day in day_start:
        if day.dayofweek == 5:
            delta1 = pd.to_timedelta("6H")
            delta2 = pd.to_timedelta("30H")
            handle = plt.axvspan(day-delta1, day+delta2, color='b', alpha=.1)

    try:
        handle.set_label("Weekend")
    except UnboundLocalError:
        #No weekends marked
        pass


def mark_alternate_time_intervals(timestamps, freq, start=None, label=None):
    t1 = start or min(timestamps).round(freq)
    t2 = max(timestamps)

    delta = pd.to_timedelta(freq)
    dates = pd.date_range(t1, t2, freq=freq)
    for i in range(0, len(dates), 2):
        handle = plt.axvspan(dates[i], dates[i] + delta, color='b', alpha=.1)

    if label:
        handle.set_label(label)


def mark_nights(timestamps, tz='UTC'):
    """Plot grey bars to indicate night time in the central timezone"""
    t1 = min(timestamps)
    t2 = max(timestamps)

    t1 = "%04i-%02i-%02i 22:00" %(t1.year, t1.month, t1.day)  #10pm
    t2 = "%04i-%02i-%02i 22:00" %(t2.year, t2.month, t2.day)  #8am
    # print (t1, t2)

    day_start = pd.date_range(start=t1, end=t2, freq='D')
    day_start = day_start.tz_localize(tz)

    delta = pd.to_timedelta("10H")
    for day in day_start:
        handle = plt.axvspan(day, day+delta, color='b', alpha=.1)

    #Edge case: First day of data set is Sunday
    # day = day_start[0]
    # if day.dayofweek == 6:
    #     delta = pd.to_timedelta("1D")
    #     handle = plt.axvspan(day-delta, day+delta, color='b', alpha=.1)
    #     plt.xlim(xmin=min(timestamps))

    # for day in day_start:
    #     if day.dayofweek == 5:
    #         delta1 = pd.to_timedelta("6H")
    #         delta2 = pd.to_timedelta("30H")
    #         handle = plt.axvspan(day-delta1, day+delta2, color='b', alpha=.1)

    try:
        handle.set_label("10pm-8am")
    except UnboundLocalError:
        #No weekends marked
        pass


def outline(clr='k', lw=1):
    """Add an outline to text or a line.
    
    See also shadow()
    Usage:
    ----------
    `plt.text(x, y, text, ... path_effects=fplots.outline() )`

    See https://matplotlib.org/stable/tutorials/advanced/patheffects_guide.html
    """

    return [
        meffect.Stroke(linewidth=lw, foreground=clr),
        meffect.Normal()
    ]


def plotMeshGrid(df, axis1, axis2, xCol, yCol, show1=True, show2=True, annotate=False, *args, **kwargs):
    """Draw a mesh grid between a set of connected points

    Given a set of points in an x-y plane that represent computed
    values for some underlying grid of objects, draw contours 
    of constant values that connect the points.

    The classic example is a grid of atmosphere models for a range
    of temperature and log(g), and you want to draw lines in 
    colour-colour space between lines of constant temperature,
    and other lines between models of constant log(g)

    Inputs
    ---------
    df
        Input dataframe 
    axis1
        (str) The column in `df` of the first axis in model space (e.g temperature)
    axis2
        (str) The column in `df` of the second axis in model space (e.g logg)
    xCol
        (str) The column in `df` with the x-values of the plotted points
    yCol 
        (str) The column in `df` with the y-values of the plotted points

    show1
        (bool) Whether to show contours for `axis1`
    show2
        (bool) Whether to show contours for `axis2`
    annotate
        (bool) Whether to annotate the points (not implemented)

    Other arguments are passed to `plt.plot`
    """
    if show1:
        gr = df.sort_values('axis2').groupby('axis1')
        print(gr.groups.keys())
        gr.apply(lambda x: plt.plot(x[xCol], x[yCol], *args, **kwargs))

    if show2:    
        gr = df.sort_values('axis1').groupby('axis2')
        print(gr.groups.keys())
        gr.apply(lambda x: plt.plot(x[xCol], x[yCol], *args, **kwargs))

    if annotate:
        raise NotImplementedError("Annotating meshgrids isn't implemented yet")


def plot_model_hist(model, bins, n_elt, *args, **kwargs):
    """Plot a stats model scaled so model.cdf(np.inf) == n_elt
    
    The scipy.stats module contains a number of common statistical functions
    (like normal, gumbel, laplacian, etc.) with a consistent interface. I often
    have the problem that I fit one of these models to some data in a maximum
    likelihood sense, then want to plot the model of a historgram of the data.
    The problem is that the model returns normalised values, i.e model.cdf(np.inf) = 1
    regardless of how many data points were including in the fit. 

    I could normalise the histogram of my data points, but that hides useful information.
    Instead, this function scales the values returned by model so they match the histogram.

    Usage:
    ---------
    ```
    import scipy.stats as spstats
    
    x = np.rand.randn(1000)
    pars = spstats.norm.fit(x)
    obj = spstats.norm(*pars)
    _, bins, _ = plt.hist(x, bins='fd')
    plot_model_hist(model, bins, len(x))
    ```

    Inputs
    -----------
    model
        (scipy.stats.rvs_continuous object) A frozen stats model (i.e one where the loc and 
        scale params have been set)
    bins
        (1d np array) location of bin edges of histogram
    n_elt
        (int) Number of elements used to create histogram of data from where the model was generated

    All other arguments are passed to matplotlib's step function

    Returns
    ----------
    **None**

    Output
    ----------
    A step function is overlaid the current plot
    """
    kwargs['where'] = kwargs.pop('where', 'post')
    vals = np.diff(model.cdf(bins))
    plt.step(bins[:-1], vals * n_elt, *args, **kwargs)


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
    return grid.iloc[0]  #Is a dictionary


def _plot_single_month(df, plotter, datecol,  *args, **kwargs):
    """Private function of `monthly_plot`"""
    #Pick the correct subplot
    subplots = '_abcdefghijkl'
    grid = kwargs.pop('_plot_grid')
    month = df[datecol].dt.month.iloc[0]
    plt.sca(grid[ subplots[month]])

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


def plot_with_discrete_cb(func, *args, **kwargs):
    """Create a plot with a discrete colorbar.

    Orbital Insight plot style encourages colourbars with 7 discrete
    colours instead of continous colors. This function wraps the call
    to a plotting function with one that creates such a discrete colourbar.

    Example plots shown in
    https://orbitalinsight.atlassian.net/wiki/spaces/DATSCI/pages/83394804/Fonts+and+Plotting+Themes_


    Inputs
    -----------
    func : Function to be wrapped. All required and optional arguments should be passed to
        ``plot_with_discrete_cb`` as they would the original function.

    Optional Inputs
    ----------------
    nstep (int): How many steps in the discrete colormap
    cmap (``matplotlib.colors.Colormap`` object): The colourmap to use
    clim (2 element list of floats):  Max and min values to map to the colour map.
        This allows you to control what range of values get pinned to the extremes
        of the colour range
    cb_ticklabel_format (Format string): Control the format of the tick labels for the colourbar.

    All other optional inputs get passed to ``func``


    Example
    ------------
    .. code-block:: python
        plot_with_discreate_cb(plt.scatter, x, y, s=z, c=z, cmap=plt.cm.Greens)

    """

    nstep = kwargs.pop('nstep', 8)
    cmap = kwargs.pop('cmap', plt.cm.Greens)
    tick_format = kwargs.pop('cb_ticklabel_format', mticker.FormatStrFormatter("%i"))
    clim = kwargs.pop('clim', None)

    if clim is None:
        raise ValueError("clim must be specified")

    norm = mcolors.BoundaryNorm(np.linspace(clim[0], clim[1], nstep), cmap.N)
    kwargs['norm'] = norm
    kwargs['cmap'] = cmap

    function_handle = func(*args, **kwargs)
    cb_handle = plt.colorbar(format=tick_format)
    return function_handle, cb_handle

from frmbase.plateau import plateau
def plot_with_gaps(x, y, *args, gap_size=None, func=plt.plot, **kwargs):
    """Plot a timeseries skipping over any gaps.
    Useful for linecharts where the x values are not regularly spaced.

    Inputs
    -------
    same as plt.plot except for the keyword argument gap_size
    If the difference between two values in x[] are larger than gapsize
    then no connector is drawn between them.

    The default gapsize is 1% the span of the data

    func
        Default is plt.plot. Any function with the signature func(x, y, *args, **kwargs)
        can be specified here
        
    Note
    ------
    If the x value is in datetime format, gap_size should be be pd.to_timedelta()
    object
    """
    if gap_size is None:
        gap_size = .01 * (np.max(x) - np.min(x))

    diff = np.diff(x)
    #casting works around issues with dates
    # peaks = plateau(diff.astype(float) > gap_size, .5)
    peaks = plateau(diff > gap_size, .5)

    if len(peaks) == 0:
        peaks = [ [0, len(x)] ]
    else:
        peaks = np.concatenate([[int(0)], peaks.flatten(), [int(len(x))]]).reshape(-1, 2).astype(int)

    for p in peaks:
        lwr, upr = p
        func(x[lwr:upr], y[lwr:upr], *args, **kwargs)


def ternary(v1, v2, v3, labels=None, **kwargs):
    """Create a ternary plot

    https://en.wikipedia.org/wiki/Ternary_plot

    A ternay plot is basically a 2d representation of a plane in
    3 dimensions. For each element of the three input vector, the
    sum of `v1[i] + v2[i] + v3[i] == 1`

    Inputs
    --------
    v1, v2, v3 (1d arrays)
        Values to plot. v1[i] + v2[i] + v3[i] == 1 for all values of i,
        and 0 <= vj[i] <= 1 for all values of i,j
    labels (list of strings)
        Text to annotate the 3 corners of the triangle.

    Optional Arguments
    ----------------------
    decorate (bool)
        If **True**, some helpful contour lines are drawn on the plot

    All other optional arguments are passed to plt.scatter

    Notes
    --------
    The solid horizontal lines drawn by decorate indicate the strength
    of vector 1. The dashed lines indicate the relative strength
    of vector 3 over vector 2. This was useful for my first use case,
    and may not make sense in other use cases.

    """
    assert len(v1) == len(v2)
    assert len(v2) == len(v3)

    for v in [v1, v2, v3]:
        assert 0 <= np.min(v) and  np.max(v) <= 1
    assert np.allclose(v1 + v2 +v3, 1)

    if labels is None:
        labels = ["Vector 1", "Vector 2", "Vector 3"]
    assert len(labels) == 3

    want_decoration = kwargs.pop('decorate', False)

    #Draw and label the triangle
    x = [-.5, 0, .5, -.5]
    y = [0, np.sqrt(3)/2., 0, 0]
    plt.plot(x, y, 'k-', lw=2, zorder=-1)

    plt.text(0, .866, "  " + labels[0], ha="left", fontsize=18)
    plt.text(.5, -.05, labels[1], ha="center", fontsize=18)
    plt.text(-.5,-.05, labels[2], ha="center", fontsize=18)

    #Mark the points
    coords = _compute_ternary_coords(v1, v2, v3)
    plt.scatter(coords[:, 0], coords[:, 1], **kwargs)

    #Scale and hide the axis.
    plt.axis('equal')
    plt.axis('off')

    if want_decoration:
        _decorate_ternary_plot(labels)


def _compute_ternary_coords(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    # #Convert pandas series to numpy arrays if necessary
    # try:
    #     a = a.values
    #     b = b.values
    #     c = c.values
    # except AttributeError:
    #     pass

    root3 = np.sqrt(3)
    y = root3 * a / 2.
    x = .5 * (b-c)
    return np.vstack([x,y]).transpose()

def _decorate_ternary_plot(labels):
    #Draw horizontal lines
    for zz in [.25, .5, .75]:
        zy = (np.sqrt(3)/2 * (zz))
        zx = .5*(1-zz)
        print(zz, zx, zy)
        plt.text(zx, zy, " %.0f%% %s" %(100*zz, labels[0]))

        plt.plot([-zx, zx], [zy, zy], '-', color='grey', lw=.5)
        plt.xlim(xmax=.6)

    #Draw diagonal lines
    # for zz in [.25, .5, .75]:
    #     zx = -.5 + zz
    #     zy = 0
    #     # plt.plot([zx, zx], [zy, zy+.01], 'k-')
    #     plt.plot([0, zx], [.866, 0], 'k-', lw=1, ls="--", zorder=-2)
    #     text = "%i%% %s" %(100*zz, labels[1])
    #     plt.text(zx, -0.04, text, fontsize=10, ha='center')

    p0 = _compute_ternary_coords([0, .75], [.25, .25], [.75, 0])
    plt.plot(p0[:,0], p0[:,1], 'b-', lw=1, ls="--", zorder=-2)
    text = "%i%% %s " %(25, labels[1])
    plt.text(p0[0,0], p0[0,1]-.2, text, rotation=60, va='bottom', ha='right')

    p0 = _compute_ternary_coords([0, .5], [.5, .5], [.5, 0])
    plt.plot(p0[:,0], p0[:,1], 'b-', lw=1, ls="--", zorder=-2)
    text = "%i%% %s " %(50, labels[1])
    plt.text(p0[0,0], p0[0,1]-.2, text, rotation=60, va='bottom', ha='right')


# def save_figfile(filename):
#     """Save a figure in a pickle, similar to Matlab's .fig format"""
#     fig = plt.gcf()
#     with open(filename, 'wb') as fp:
#         pickle.dump(fig, fp)

def save_figfile(filename):
    """Save a figure in a compressed pickle, similar to Matlab's .fig format"""
    fig = plt.gcf()
    with open(filename, 'wb') as fp:
        fp.write( gzip.compress(pickle.dumps(fig)) )


def shadow(**kwargs):
    """Add a drop-shadow to a line or some text 

    See also outline()
    
    Usage:
    ----------
    `plt.plot(x, y, ... path_effects=fplots.shadow() )`

    See https://matplotlib.org/stable/tutorials/advanced/patheffects_guide.html
    """

    return [meffect.SimpleLineShadow(**kwargs), meffect.Normal()]


def text_at_axis_coords(x, y, text, *args, **kwargs):
    """Place text at coords relative to axis. 0,0 always means bottom left of plot"""

    ax = plt.gca()
    ax.text(x, y, text, *args, transform=ax.transAxes, **kwargs)


def text_at_figure_coords(x, y, text, *args, **kwargs):
    """Place text at coords relative to the figure. 0,0 always means bottom left of figure"""

    ax = plt.gca()
    f = plt.gcf()
    ax.text(x, y, text, *args, transform=f.transFigure, **kwargs)



