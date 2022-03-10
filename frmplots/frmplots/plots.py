"""
Created on Mon Nov 28 14:15:53 2016
Some common plots I like to make
@author: fergal

Some tips I haven't written to functions yet
---------------------------------------------
How to have dates on the colour bar axis

::
    from matplotlib.dates import DateFormatter,date2num
    dates = pd.date_range('2020-01-01', '2020-12-01', freq='D')
    date_num = list(map(date2num, dates))
    plt.scatter(x, y, c=date_num)
    plt.colorbar(format=DateFormatter("%Y-%m-%d"))

"""

import matplotlib.collections as mcollect
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as meffect
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import pickle
import copy

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
    All other arguments are passed to `plt.text`

    """
    #Define some default values
    fmt = kwargs.pop('fmt', '%g')
    ha = kwargs.pop('ha', 'center')
    offset_sign = 2* kwargs.pop('above', True) - 1

    va = kwargs.pop('va', None)
    if va is None:
        if offset_sign > 0:
            va = 'bottom'
        else:
            va = 'top'

    offset = offset_sign * .04 * (np.max(vals) - np.min(vals))
    locs = bins[:-1] + .5* np.diff(bins)
    sign = -1
    old = 0
    for xpos, val in zip(locs, vals):
        if val <= 0:
            continue 

        ypos = val + offset
        if np.fabs(ypos - old) < offset:
            ypos += sign * offset 
            sign *= -1
        
        old = ypos 
        plt.text(xpos, ypos + offset, fmt%(val), ha=ha, va=va, **kwargs)


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
    data[:,0,1] = 0
    data[:,1,0] = x
    data[:,1,1] = .1 

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
    cmap = kwargs.pop('cmap', DEFAULT_CMAP)
    norm = kwargs.pop('norm', None)
    ls = kwargs.pop('ls', "none")
    ls = kwargs.pop('linestyle', ls)
    marker = kwargs.pop('marker', 'o')
    # aspect = kwargs.pop('aspect', 'auto')   #Equal gives square bins
    zorder = kwargs.pop('zorder', 0)

    cmap = copy.copy(cmap)  #I'm annoyed with mpl for making me do this
    cmap.set_under(alpha=0)
    plt.plot(x, y, zorder=zorder-1, marker=marker, ls=ls, *args, **kwargs)
    plt.hist2d(x, y, bins=[xBins, yBins], cmap=cmap, norm=norm)
    plt.clim(vmin=threshold)




def fix_date_labels():
    """Format date strings in x-axis label so they're easier to read"""
    plt.gcf().autofmt_xdate()


def load_figfile(filename):
    """Load a figure from a pickle, similar to Matlab's .fig format"""
    fig = plt.gcf()
    with open(filename, 'rb') as fp:
        fig = pickle.load(fp)
    return fig



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


def save_figfile(filename):
    """Save a figure in a pickle, similar to Matlab's .fig format"""
    fig = plt.gcf()
    with open(filename, 'wb') as fp:
        pickle.dump(fig, fp)


def shadow():
    """Add a drop-shadow to a line or some text 

    See also outline()
    
    Usage:
    ----------
    `plt.plot(x, y, ... path_effects=fplots.shadow() )`

    See https://matplotlib.org/stable/tutorials/advanced/patheffects_guide.html
    """

    return [meffect.SimpleLineShadow(), meffect.Normal()]


def text_at_axis_coords(x, y, text, *args, **kwargs):
    """Place text at coords relative to axis. 0,0 always means bottom left of plot"""

    ax = plt.gca()
    ax.text(x, y, text, *args, transform=ax.transAxes, **kwargs)


def text_at_figure_coords(x, y, text, *args, **kwargs):
    """Place text at coords relative to the figure. 0,0 always means bottom left of figure"""

    ax = plt.gca()
    f = plt.gcf()
    ax.text(x, y, text, *args, transform=f.transFigure, **kwargs)



