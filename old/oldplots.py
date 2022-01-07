
import matplotlib.collections as mcollect
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import copy

# from .anygeom import AnyGeom
from . import plotstyle
from .. import oicolours as oic


def put_colorbar_at_fig_coords(rect, **kwargs):
    """Put a colorbar on a plot without changing figure coordinates of the axis

    Default behaviour for matplotlib is to decrease the width of an
    axis to make room for a colorbar. This is sometimes undesireable (e.g
    when dealing with subplots). This function lets you place a colorbar
    at your requested location on a plot, without adjusting the size
    of any other axes.

    Inputs
    ------
    rect
        (tuple) values are (left, bottom, width, height), all in figure
        units. All values should be in the range [0,1].

    Optional Inputs
    --------------
    label
        (str) Label for colorbar
    fontsize
        (int) Fontsize to use for colorbar tick labels
    ticklen
        (int) Length of ticks for colorbar

    All other keyword arguments are passed to colorbar()

    Returns
    --------
    The colorbar axis object.
    """

    label = kwargs.pop('label', None)
    fontsize = kwargs.pop('fontsize', 12)
    ticklen = kwargs.pop('ticklen', 2)

    cax = plt.axes(rect, **kwargs)
    cb = plt.colorbar(cax=cax)

    if label is not None:
        cb.set_label(label, fontsize=fontsize)
    cb.ax.tick_params(labelsize=fontsize, length=ticklen)
    return cb


# def chloropleth(polygons, values, **kwargs):
#     """Make a map where each polygon's colour is chosen on the basis of the
#     associated value

#     Inputs
#     -------------
#     polygons
#         (list or array) An array of WKT strings or ogr shapes
#     values
#         (list or array) The value associated with each polygon. The number
#         of polygons must equal the number of values

#     Optional Inputs
#     -----------------
#     cmap
#         (matplotlib colormap) The color scheme used. Default is rainbow.
#     norm
#         (string or matplotlib normalisation object). How to map from value to
#         a colour. Default is a BoundaryNorm which produces 7 discrete colours.
#         `norm` can be a normalization object or the string 'hist'. If it's
#         'hist', we use histogram normalization
#     nstep
#         (int) How many steps in the discrete colourbar.
#     vmin, vmax
#         (floats) Min and max range of colour scale. Default is taken from range
#         of `values`
#     wantCbar
#         (bool) Draw a colour bar. Default is **True**

#     All other optional inputs are passed to `matplotlib.patches.Polygon`

#     Returns
#     -----------
#     A patch collection object, and a colorbar object


#     Output
#     -----------
#     A plot is produced


#     Note
#     -----------
#     If function raises a value error "Histequ failed", try setting nstep
#     to a value less than 8.
#     """

#     assert len(polygons) == len(values), "Input args have different lengths"
#     #Choose default values
#     vmin = kwargs.pop('vmin', np.min(values))
#     vmax = kwargs.pop('vmax', np.max(values))
#     nstep = kwargs.pop('nstep', 8)
#     cmap  = kwargs.pop('cmap', DEFAULT_CMAP)
#     wantCbar = kwargs.pop('wantCbar', True )
#     ticklabel_format = kwargs.pop('ticklabel_format', '%i')


#     #Complicated way of picking the norm. Default is linear. User can specify
#     #'hist', and we'll compute a histequ norm. Otherwise, treat the passed arg
#     #as a norm object.
#     default_norm = mcolors.BoundaryNorm(np.linspace(vmin, vmax, nstep), cmap.N)
#     norm = kwargs.pop('norm', default_norm )
#     if isinstance(norm, str):
#         if norm[:4].lower() == 'hist':
#             norm = create_histequ_norm(values, cmap, vmin, vmax, nstep)
#         else:
#             raise ValueError("Unrecognised normalization option %s" %(norm))

#     # if isinstance(polygons[0], str):
#     #     polygons = map(ogr.CreateGeometryFromWkt, polygons)
#     patch_list = []
#     for p,v in zip(polygons, values):
#         #patch = ogrGeometryToPatches(p, facecolor=cmap(norm([v])[0]), alpha=a, **kwargs)
#         # patch = ogrGeometryToPatches(p, facecolor=cmap(norm([v])[0]), **kwargs)
#         patch = AnyGeom(p).as_patch(facecolor=cmap(norm([v])[0]), **kwargs)
#         patch_list.extend(patch)
#         # print(AnyGeom(p).as_wkt())
#     pc = mcollect.PatchCollection(patch_list, match_original=True)

#     print(p)
#     #Plot the patchlist
#     ax = plt.gca()
#     ax.add_collection(pc)

#     #Draw a colorbar
#     if wantCbar:
#         cax, kw = mpl.colorbar.make_axes(ax)
#         cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
#                                        format=ticklabel_format)
#     else:
#         cb = None

#     #Set the main panel as the default axis again (otherwise)
#     #future plotting will happen on the colourbar itself
#     plt.sca(ax)

#     return pc, cb


def create_histequ_norm(values, cmap, vmin, vmax, nstep):
    """Create a histogram equalized matplotlib normalization object

    A normalization object maps values in to the range [0,1]. These
    values can be subsequently mapped to colours using a colour map.
    See https://matplotlib.org/users/colormapnorms.html and
    https://matplotlib.org/users/colormapnorms.html

    A histogram equalization normalization maps all input values into a
    discrete set of output values in a way that ensures that an equal fraction
    of the input values is mapped to each output value.

    I think this is the algorithm described on Wikiepedia
    https://en.wikipedia.org/wiki/Histogram_equalization

    Inputs
    -----------
    values
        (list or array) Values to compute the histogram with
    cmap
        (matplotlib colormap) Needed to to compute the boundary object
        for some reason
    vmin, vmax
        (floats) Only values in this range are used to compute norms.
        If set to None, min or max of `values` is used as appropriate.
    nstep
        (int) Number of steps in normalization.


    Returns
    ------
    A matplotlib BoundaryNorm object.

    Exceptions
    ------------
    If `values` are integers, and `nstep` is too large, there may be
    a bin collision (it's not possible to put equal numbers of points
    in each bin). If that happens, the function raises an exception. If
    that happens, reduce the value of `nstep`
    """

    if vmin is None:
        vmin = np.min(vmin)

    if vmax is None:
        vmax = np.max(vmax)

    v = values[ (vmin <= values) & (values <= vmax) ]
    thresholds = np.percentile(v, np.linspace(0, 100, nstep))
    if np.any(np.diff(thresholds) == 0):
        raise ValueError("Histequ failed. Try setting nstep to a lower value (default is 8)")

    norm = mcolors.BoundaryNorm(thresholds, cmap.N)
    return norm




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
    interp
        Whether to interplolate between bins to produce a smoother histogram.
        Default: no interpolation (aka nearest)
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
    # interp = kwargs.pop('interpolation', 'nearest')
    # aspect = kwargs.pop('aspect', 'auto')   #Equal gives square bins
    zorder = kwargs.pop('zorder', 0)
    # wantLog = kwargs.pop('log', False)

    # extent = [np.min(xBins), np.max(xBins), np.min(yBins), np.max(yBins)]
    cmap.set_under(alpha=0)
    plt.plot(x, y, zorder=zorder-1, *args, **kwargs)
    plt.hist2d(x, y, bins=[xBins, yBins], cmap=cmap, norm=norm)

    # extent = [np.min(xBins), np.max(xBins), np.min(yBins), np.max(yBins)]
    # hist = np.histogram2d(y, x, bins=[yBins, xBins])[0]
    # if wantLog:
    #     hist = np.log10(hist + 1e-6)

    # import ipdb; ipdb.set_trace()
    # cmap.set_under(alpha=0)
    # plt.plot(x, y, zorder=zorder-1, *args, **kwargs)
    # plt.imshow(hist, origin="bottom", extent=extent,
    #            zorder=zorder,
    #            cmap=cmap,
    #            interpolation=interp,
    #            aspect=aspect,
    #            norm=norm)

    plt.clim(threshold)



def plot_timeseries(x, y, **kwargs):
    """Plot a time-series as a stem plot using OI colours

    Time series of counts of some object from satellite imagery tends to
    look ugly when you plot it with the standard plot method. The points
    are typically sparse, and clustered, and the plot is dominated by empty
    space, or meangingless lines acrossing that space.

    Stem plots look good, provided only a single time series is being plotted
    Multiple overlaid stem plots tend to look confused.

    Inputs
    -----------
    x, y
        Input data. If x is **None**, it's set to {1..len(y)}

    Optional Inputs
    ----------------
    markerfmt
        Change default plotting point from 'o' to something else
    ms
         Size of marker symbol. Deafult is 10
    color
        Color of points
    lw
        With of stem lines
    linecolor
        Color of stem lines
    basefmt
        Draw a line at zero, with style, e,g 'r--'

    Returns
    ----------
    **None**

    Output
    -------
    Points are added to the current plot
    """

    if x is None:
        x = np.arange(len(y))

    markerfmt = kwargs.pop('markerfmt', 'o')
    basefmt = kwargs.pop('basefmt', ',')
    color = kwargs.pop('color', oic.OI_LIGHT_BLUE_HEX)
    lw = kwargs.pop('lw', 1)
    ms = kwargs.pop('ms', 12)
    linecolor = kwargs.pop('linecolor', oic.OI_GRAY_HEX)
    label = kwargs.pop('label', None)

    marker, stem, _ = plt.stem(x, y, markerfmt=markerfmt, basefmt=basefmt, use_line_collection=True)
    plt.setp(stem, 'color', linecolor)
    plt.setp(stem, 'lw', lw)
    plt.setp(stem, 'zorder', -1)

    plt.setp(marker, 'color', color)
    plt.setp(marker, 'mec', "None")
    plt.setp(marker, 'ms', ms)
#    if label is not None:
    plt.setp(marker, 'label', label)

    #Put a little white space at top and bottom of plot
    padding = .02 * np.max(y)
    plt.ylim(-padding, np.max(y)+padding)



def scene_coverage_for_axis(ax, bins, scene_list, **kwargs):
    """Compute and plot the number of scenes that cover each part of map drawn in axis
    Convenience function for ``scene_coverage``. Tell it the number of
    bins you want to use to cover the entire plot and it will calculate
    the bins for you

    Inputs
    --------
    bins
        (int or two-tuple) If integer, create that many bins in x and y.
        If two tuple use different number of bins for x and y.

    scene_list
        (iterable) List of scenes bounds. Each bound may be an OGR object
        or a WKT string

    Optional Inputs
    -----------------
    Passed to ``matplotlib.pyplot.imshow()``

    Returns:
    -------------
    Coverage map as 2d array

    Output
    -------------
    Adds an image to the current plot

    Note
    --------
    This is very quick and dirty code. In particular it
    * Assumes the earth is flat

    """
    lng0, lng1, lat0, lat1 = plt.axis()

    try:
        numLngBins, numLatBins = bins
    except TypeError:   #bins is an integer
        numLngBins = bins
        numLatBins = bins

    lngBins = np.linspace(lng0, lng1, numLngBins)
    latBins = np.linspace(lat0, lat1, numLatBins)

    return scene_coverage(lngBins, latBins, scene_list, **kwargs)






def set_cbar_labels_to_int(cb):
    """Set labels on colorbar to integers
    When representing integer values with a colour map, the colorbar
    often shows the floating point values in the tick labels, which
    can be confusing. This function rounds those floating point values
    to the nearest integer to emphasise the integer nature of the values
    being plotted.
    Inputs
    ------------
    cb
        A colorbar object, as returned by ``plt.colorbar()``
    Returns
    -----------
    **None**
    Output
    ----------
    The ticklabels on the colorbar are modified.
    """

    labels = cb.ax.yaxis.get_ticklabels()
    labels = map(lambda x: "%.0f" %( float(x.get_text())), labels)
    cb.ax.yaxis.set_ticklabels(labels)


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

    #Convert pandas series to numpy arrays if necessary
    try:
        a = a.values
        b = b.values
        c = c.values
    except AttributeError:
        pass

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
    for zz in [.25, .5, .75]:
        zx = -.5 + zz
        zy = 0
        # plt.plot([zx, zx], [zy, zy+.01], 'k-')
        plt.plot([0, zx], [.866, 0], 'k-', lw=1, ls="--", zorder=-2)
        text = "%i%% %s" %(100*zz, labels[1])
        plt.text(zx, -0.04, text, fontsize=10, ha='center')



def ogrGeometryToPatches(shape, **kwargs):
    """Convert an ogr geometry object to matplotlib Patch objects

    Creates a list of patches. If input shape an ordinary
    polygon, the list will have one element. If it is a multi-polygon
    the list can have many elements.


    Inputs:
    ----------
    shape
        Ogr geometry object. Only tested with polygons and
        multi-polygons

    Optional Inputs
    --------------
    Passed directly to `matplotlib.patch.Polygon`

    Returns
    -------------
    A list of `matplotlib.patch.Polygon`  objects
    """

    patches = []
    xy_list = ogrGeometryToArray(shape)
    for elt in xy_list:
        assert elt.ndim == 2
        patches.append(mpatch.Polygon(elt, closed=True, **kwargs))

    return patches


def plot_shape(shape, *args, **kwargs):
    """Plot the outline of a shape

    Inputs
    ------------
    shape
        Ogr geometry object. Can be any single Ogr object (e.g a point,
        a line, or a polygon). Only really tested on ploygons
        and multi-polygons

    Optional Inputs
    --------------
    Passed directly to `matplotlib.pyplot.plot`

    Returns
    -------------
    **None**

    Output
    ---------
    Draws to the current matplotlib axis
    """

#    import ipdb; ipdb.set_trace()
    #shape = ogrGeometryToArray(shape)
    shape = AnyGeom(shape).as_array()[1]
    if isinstance(shape, np.ndarray) and shape.size == 0:
        return #empty geom

    if isinstance(shape, list):
        out = []
        for region in shape:
            if isinstance(region, list):
                region = region[0]
            handle = plt.plot(region[:,0], region[:,1], *args, **kwargs)
            out.append(handle)
        return out
    else:
        shape = np.atleast_2d(shape)
        return plt.plot(shape[:,0], shape[:,1], *args, **kwargs)


def ogrGeometryToArray(geometry):
    """Convert an OGR geometry object to a 2d numpy array

    The array is of shape nx2, where n is the number of points in the shape.

    This works as expected for Polygons, but does something slightly
    unexpected for multi-polygons. For a multi-polygon it returns
    a list. Each element of the list is either a 2d numpy array or
    another list. By checking each list element recursively you can
    finally arrive at a set or numpy arrays, each one repesenting a
    single polygon.

    See plot_shape
    """

#    if geometry.GetGeometryName() == 'POINT':
#        return np.array(geometry.GetX(), geometry.GetY())

    if geometry.GetGeometryCount() > 0:
        elts = []
        for i in range(geometry.GetGeometryCount()):
            elts.append( ogrGeometryToArray(geometry.GetGeometryRef(i)) )
        return flatten_list(elts)

    return  [np.atleast_2d(geometry.GetPoints())]


def flatten_list(x):
    """Recursively flatten a list

    [1,2,3, [4,5, [6]]] --> [1,2,3,4,5,6]
    """
    out = []
    if isinstance(x, list):
        for elt in x:
            out.extend( flatten_list(elt) )
    else:
        out.append(x)

    return out



def galaxyPlot(data, colList, *args, **kwargs):
    """Plot everything against everything

    Just like extra galactic astronomers are wont to do.

    Inputs:
    -------------
    data
        (2d array) Data to be plotted

    colList
        (list) list of column numbers to plot. Must have at least
        two elements

    Optional Inputs:
    ---------------
    plotFunc
        (A plotting function) What function should actually plot the x,y
        points. Default is plt.plot()

    labels
        (list of strings) Applied as x and y axis labels. If not
        given, defaults to values in colList

    wantLog
        (list or array) If True, given column is plotted in log scale.
        len(wantLog) == len(colList)

    limits
        (list) If the ith elememt of this array is an object of length 2,
        use those two number to set the plot limits in that dimension.
        See notes, below.

    All other optional arguments are passed to plt.plot (or whatever plotFunc
    is)

    Returns:
    -------
    void

    Output:
    --------
    A plot is produced.


    Notes:
    -------
    The signature of plotFunc is
    ``plotFunc(x, y, *args, **kwargs)``
    similar to plt.plot()'s most commonly called style.

    The easiest way to create the limits argument is as follows::

      limits = list(np.zeros_like(colList))
      limits[4] = [1, 10]

    The function will see that only the 4th element of the array
    is a set of limits, and default values will be chosen for all
    other dimensions

    """

    labels = kwargs.pop('labels', colList)
    ms = kwargs.pop('markersize', 3)
    plotFunc = kwargs.pop('plotFunc', plt.plot)

    wantLog = np.zeros(len(colList))
    wantLog = kwargs.pop('wantLog', wantLog)
    assert( len(wantLog) == len(colList))

#    hold = kwargs.pop('hold', False)

    limits = np.zeros(len(colList), dtype=bool)
    limits = kwargs.pop('limits', limits)
    assert( len(limits) == len(colList))


    nCol = len(colList)
    nPlot = nCol-1

    #Create a 2d array of Axes objects
    axArray = np.empty( (nCol, nCol), dtype=object)
    for i in range(nCol):
        for j in range(0, i):
            n = (nPlot)*(i-1)+j + 1
            sharex, sharey = findSharedAxes(axArray, i, j)

            if sharex is None and sharey is None:
                ax = plt.subplot(nPlot, nPlot, n)
            if sharex is None and sharey is not None:
                ax = plt.subplot(nPlot, nPlot, n, sharey=sharey)
            if sharex is not None and sharey is None:
                ax = plt.subplot(nPlot, nPlot, n, sharex=sharex)
            if sharex is not None and sharey is not None:
                ax = plt.subplot(nPlot, nPlot, n, sharex=sharex, sharey=sharey)

            axArray[i,j] = ax

    #i loops over the y-axes (the rows)
    for i in range(nCol):

        idx = np.isfinite(data[:, colList[i]])

        #Pick y limits for subplot. If not supplied, use range of data
        if hasattr(limits[i], "__len__") and len(limits[i]) == 2:
            yLwr, yUpr = limits[i]
        else:
            yLwr = np.min(data[idx, colList[i]])
            yUpr = np.max(data[idx, colList[i]])

        #j loops over the x-axes (the columns)
        for j in range(0, i):
            ax = axArray[i,j]
            plt.sca(ax)

            plotFunc(data[:, colList[j]], data[:, colList[i]], \
                markersize=ms, *args, **kwargs)
#            plt.text(.3, .3, "%i %i" %(i,j), transform=ax.transAxes )

            #Add decorations for the subplot
            if j== i-1:
                plt.title(labels[j])
#                plt.title("i=%i j=%i n=%i" %(i,j,n))

            if j == 0:
                plt.ylabel(labels[i])

            if i != nCol-1:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                plt.xlabel(labels[j])

            if wantLog[i]:
                ax.set_yscale('log')
                ax.yaxis.set_tick_params(which="major", length=10)
                ax.yaxis.set_tick_params(which="minor", length=4)


            if wantLog[j]:
                ax.set_xscale('log')
                ax.xaxis.set_tick_params(which="major", length=10)
                ax.xaxis.set_tick_params(which="minor", length=4)

            if j > 0:
                plt.setp(ax.get_yticklabels(), visible=False)

            #Pick x limits for subplot. If not supplied, use range
            #range of data
            if hasattr(limits[j], "__len__") and len(limits[j]) == 2:
                plt.xlim(limits[j])
            else:
                xLwr = np.min(data[idx, colList[j]])
                xUpr = np.max(data[idx, colList[j]])
                plt.xlim([xLwr, xUpr])
            plt.ylim([yLwr, yUpr])

    plt.subplots_adjust(wspace=0, hspace=0)


def findSharedAxes(axArray, i, j):
    xAx = None
    yAx = None

    if i>j+1:
        xAx = axArray[i-1, j]

    if j>0:
        yAx = axArray[i, j-1]

    return xAx, yAx



def addSizeGuideLegend(sizes, labels, symbols='o', colors='k', **kwargs):
    """Add a legend to associate text with symbols of different sizes

    The best analogy for this function is a map that uses circles
    of different sizes to show city population. Then the map key
    explains what population is indicated by a circle of a given size.

    This is useful for scatter plots where the size of the symbol
    is supposed to convey information.


    Inputs
    --------
    sizes
        (list or array of numbers) List of values indicating symbol sizes
        to plot labels
    labels
        (list of array of strings) Labels to attach to symbol in the legend
        `sizes` and `labels` must be the same length.


    Optional Inputs
    ---------------
    symbols
        (str or list of strings) What symbol to plot for each entry (e.g
        circle, square, start, etc.). If a list, the length should be the
        same as `sizes`

    colors
        (str or list of strings) Colour to plot the symbols. If a string,
        can be any matplotlib colour format. If a list, the length must
        be the same as `sizes`

    All other keyword arguments are passed to matplotlibs `legend` function


    Returns
    ----------
    The handle of the legend


    Notes
    ---------
    By default, matplotlib only lets you put one legend on an axis. To
    override this behaviour, use the following code to add the **other** legend


    Examples
    -----------

    .. code-block:: python

        sizes = [10, 40, 100, 400]
        labels = ["10 cows", "40 cows", "100 cows", "400 cows"]

        addSizeGuideLegend(sizes, labels)



    .. code-block:: python

        steps = np.linspace(np.min(s), np.max(s), 5)
        labels = map(lambda x: "%.1f" %(x), steps)

        addSizeGuideLegend(sizes, labels)

    """
    assert len(sizes) == len(labels)

    if isinstance(colors, str):
        colors = [colors] * len(sizes)

    if isinstance(symbols, str):
        symbols = [symbols] * len(sizes)

    ax = plt.gca()
    handles = []
    for s, lab, c, sym in zip(sizes, labels, colors, symbols):
        h = ax.scatter([], [], color=c,
                          marker=sym,
                          s=s,
                          label=lab,
                          **kwargs)
        handles.append(h)

    legend = plt.legend(handles=handles, labelspacing=2)
    ax.add_artist(legend)
    return legend



