# -*- coding: utf-8 -*-
"""
Tools to plot geolocation data. See also geoloc.py

@author: fergal
"""

from ipdb import set_trace as idebug
from pdb import set_trace as debug

import matplotlib.collections as mcollect
import matplotlib.colors as mcolors
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import matplotlib as mpl

from frm.anygeom import AnyGeom
import frm.plots as fplots
import pandas as pd
import numpy as np
import frm.support

npmap = frm.support.npmap


def chloropleth(polygons, values, **kwargs):
    """Make a map where each polygon's colour is chosen on the basis of the
    associated value

    Inputs
    -------------
    polygons
        (list or array) An array of WKT strings or ogr shapes
    values
        (list or array) The value associated with each polygon. The number
        of polygons must equal the number of values

    Optional Inputs
    -----------------
    cmap
        (matplotlib colormap) The color scheme used. Default is rainbow.
    norm
        (string or matplotlib normalisation object). How to map from value to
        a colour. Default is a BoundaryNorm which produces 7 discrete colours.
        `norm` can be a normalization object or the string 'hist'. If it's
        'hist', we use histogram normalization
    nstep
        (int) How many steps in the discrete colourbar.
    vmin, vmax
        (floats) Min and max range of colour scale. Default is taken from range
        of `values`
    wantCbar
        (bool) Draw a colour bar. Default is **True**

    All other optional inputs are passed to `matplotlib.patches.Polygon`

    Returns
    -----------
    A patch collection object, and a colorbar object


    Output
    -----------
    A plot is produced


    Note
    -----------
    If function raises a value error "Histequ failed", try setting nstep
    to a value less than 8.
    """

    assert len(polygons) == len(values), "Input args have different lengths"
    #Choose default values
    vmin = kwargs.pop('vmin', np.min(values))
    vmax = kwargs.pop('vmax', np.max(values))
    nstep = kwargs.pop('nstep', 8)
    cmap  = kwargs.pop('cmap', DEFAULT_CMAP)
    wantCbar = kwargs.pop('wantCbar', True )
    ticklabel_format = kwargs.pop('ticklabel_format', '%i')


    #Complicated way of picking the norm. Default is linear. User can specify
    #'hist', and we'll compute a histequ norm. Otherwise, treat the passed arg
    #as a norm object.
    default_norm = mcolors.BoundaryNorm(np.linspace(vmin, vmax, nstep), cmap.N)
    norm = kwargs.pop('norm', default_norm )
    if isinstance(norm, str):
        if norm[:4].lower() == 'hist':
            norm = create_histequ_norm(values, cmap, vmin, vmax, nstep)
        else:
            raise ValueError("Unrecognised normalization option %s" %(norm))

    # if isinstance(polygons[0], str):
    #     polygons = map(ogr.CreateGeometryFromWkt, polygons)
    patch_list = []
    for p,v in zip(polygons, values):
        #patch = ogrGeometryToPatches(p, facecolor=cmap(norm([v])[0]), alpha=a, **kwargs)
        # patch = ogrGeometryToPatches(p, facecolor=cmap(norm([v])[0]), **kwargs)
        patch = AnyGeom(p).as_patch(facecolor=cmap(norm([v])[0]), **kwargs)
        patch_list.extend(patch)
        # print(AnyGeom(p).as_wkt())
    pc = mcollect.PatchCollection(patch_list, match_original=True)

    #Plot the patchlist
    ax = plt.gca()
    ax.add_collection(pc)

    #Draw a colorbar
    if wantCbar:
        cax, kw = mpl.colorbar.make_axes(ax)
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                       format=ticklabel_format)
    else:
        cb = None

    #Set the main panel as the default axis again (otherwise)
    #future plotting will happen on the colourbar itself
    plt.sca(ax)

    return pc, cb

def getEnvelopeAsGeom(geom):
    """Get the envelope of a geometry as a polygon

    GDAL's GetEnvelope() function returns a 4-tuple of corner coordinates.
    I frequently want to convert those points to a rectangle, and store
    it in a geometry.

    Inputs
    ---------
    geom
        A gdal geometry object


    Returns
    ---------
    A gdal geometry object
    """

    x0, x1, y0, y1 = geom.GetEnvelope()

    data = [ [x0, y0],
             [x0, y1],
             [x1, y1],
             [x1, y0],
             [x0, y0]
           ]

#    debug()
    env = AnyGeom(data, "polygon").as_geometry()
    return env


def draw_geohash_bounds(gh, **kwargs):
    """Draw a box around the geohash on a map

    Useful to mark extent of a geohash without obscuring
    the map underneath.

    Inputs
    ----------
    gh
        (str) Geohash of interest


    Optional Inputs
    -------------
    All optional arguments passed to matplotlib.patch.Rectange

    """
    fill = kwargs.pop('fill', False)
    bounds = oi_geohash.geohash_bounds(gh)
    lng0, lat0 = bounds['sw']
    lng1, lat1 = bounds['ne']

    h = lat0 - lat1
    w = lng1 - lng0
    rect = mpatch.Rectangle((lng0, lat1), w, h, fill=fill, **kwargs)

    plt.gca().add_patch(rect)


def plot_geohash_cloropleth(ghlist, values, **kwargs):
    """Plot a cloropleth map of values per geohash"""

    assert len(ghlist) == len(values)
    values = np.array(values)

    #Set up plotting options
    wantCbar = kwargs.pop('wantCbar', True )
    ticklabel_format = kwargs.pop('ticklabel_format', '%i')

    vmin = kwargs.pop('vmin', np.min(values))
    vmax = kwargs.pop('vmax', np.max(values))
    nstep = kwargs.pop('nstep', 8)

    cmap = kwargs.pop('cmap', fplots.oi_cmap)
    default_norm = mcolors.BoundaryNorm(np.linspace(vmin, vmax, nstep), cmap.N)
    norm = kwargs.pop('norm', default_norm)

    #Work around a bit of a bug in HistEquNorm
    norm(values)

    #Compute geohash boundaries
    bounds = npmap(oi_geohash.geohash_bounds, ghlist)
    bottom_left = npmap(lambda x: x['sw'], bounds)
    top_right = npmap(lambda x: x['ne'], bounds)

    height_width = top_right - bottom_left
    height = height_width[:,0]
    width = height_width[:,1]

    #Construct patches, add to a patch collection.
    patch_list = []
    for i in range(len(bounds)):
        clr = cmap(norm(values[i]))
        patch = mpatch.Rectangle(bottom_left[i], width[i], height[i],
                                 facecolor=clr, **kwargs)
        patch_list.append(patch)

    pc = mcollect.PatchCollection(patch_list, match_original=True)
    ax = plt.gca()
    ax.add_collection(pc)

    plt.plot(bottom_left[:2,0], bottom_left[:2,1], 'w-', zorder=-1000)

    #Draw a colorbar
    if wantCbar:
        cax, kw = mpl.colorbar.make_axes(ax)
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                       format=ticklabel_format)
    else:
        cb = None

    #Set the main panel as the default axis again (otherwise)
    #future plotting will happen on the colourbar itself
    plt.sca(ax)

    return pc, cb


def mark_holidays(delta=None):
    """
    Mark major holidays as vertical lines

    Optional Inputs
    --------------
    delta
        (str) e.g '1D' or '12H'. Plot the lines with this offset from midnight


    Notes
    ---------
    Currently only valid for 2017/2018
    """

    days = "2017-09-04 2017-11-23 2017-12-25 2018-01-01 2018-01-15 2018-02-19 2018-05-28"
    days = pd.to_datetime(days.split())

    if delta is None:
        delta = pd.to_timedelta(0)
    else:
        delta = pd.to_timedelta(delta)

    for d in days:
        plt.axvline(d + delta, color='grey')


def mark_nights(timestamps, tz='UTC'):
    """Plot grey bars to indicate night time in the central timezone"""
    t1 = min(timestamps)
    t2 = max(timestamps)

    t1 = "%04i-%02i-%02i 21:00" %(t1.year, t1.month, t1.day)  #10pm
    t2 = "%04i-%02i-%02i 21:00" %(t2.year, t2.month, t2.day)
    dusk = pd.date_range(start=t1, end=t2, freq='D')

    delta = pd.to_timedelta("12h")  #8am
    t3 = pd.to_datetime(t1) + delta
    t4 = pd.to_datetime(t2) + delta
    dawn = pd.date_range(start=t3, end=t4, freq='D')

    dusk = dusk.tz_localize(tz)
    dawn = dawn.tz_localize(tz)
    for a, b in zip(dusk, dawn):
        plt.axvspan(a, b, color='k', alpha=.2)


def mark_weekends(timestamps, tz='UTC'):
    """Plot grey bars to indicate night time in the central timezone"""
    t1 = min(timestamps)
    t2 = max(timestamps)

    t1 = "%04i-%02i-%02i 00:00" %(t1.year, t1.month, t1.day)  #10pm
    t2 = "%04i-%02i-%02i 00:00" %(t2.year, t2.month, t2.day)  #10pm
    print (t1, t2)

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
            delta = pd.to_timedelta("2D")
            handle = plt.axvspan(day, day+delta, color='b', alpha=.1)

    try:
        handle.set_label("Weekend")
    except UnboundLocalError:
        #No weekends marked
        pass


def plot_barcode(df, *args, idkey='ad_id', tkey='unixtime', **kwargs):
    """Make a barcode plot for ping data

    A barcode plot is a series of whiskers, one per ping. All
    the pings from a single device are on the same row, making
    it easy to see both when the pings are emitted, and what
    the temporal distribution for a single device is.


    Inputs
    ----------
    df :
        (DataFrame) Input dataframe. Each row of the dataframe
        represents a ping for an individual device. There are
        many such devices (and many pings per device) in the input

    Optional Inputs
    -------------------
    idkey :
        Column used to identify populations in the data. Typically
        this is either 'ad_id' or 'app_id'
    tkey
        Column containing ping timestamp. Check that the input
        data isn't in string format before you pass it.
    plot_func
        (Function) Name of function to do the actual plotting.
        Use this to plot the data in a customised fashion (e.g by
        plotting some devices in a different colour)
    plot_guides
        (bool) If **False** then don't plot horizontal lines as
        guides to the eye.
    guide_props
        (dict) Properties of horizontal guides. Passed to plt.axhline()

    Returns
    -------
    None.

    Notes
    ------
    * Not a fast piece of code. Could be improved by using
      LineCollection objects
    """

    default_guide_props = {'color':'grey',
                           'lw':.5,
                           'ls':'-',
                           'spacing': 5}

    plot_func = kwargs.pop('plot_func', _plot_temporal)
    plot_guides = kwargs.pop('plot_guides', True)
    guide_props = kwargs.pop('guide_props', default_guide_props)

    plt.clf()
    iterator = _get_next_row()
    df.groupby(idkey).apply(plot_func, iterator, tkey, *args, **kwargs)

    #Plot some horizontal lines to guide the eye
    if plot_guides:
        spacing = guide_props.pop('spacing', 4)
        for i in range(0, int(plt.ylim()[1]), spacing):
            plt.axhline(i, **guide_props)

    plt.xlabel("Date")
    plt.ylabel("Device Number")


def _plot_temporal(df, iterator, tCol, *args, **kwargs):
    """Default plotting algorithm for plot_barcode
    You can write your own one of these to customise the plot
    """
    series = df[tCol]

    if len(series) < 2:
        return

    style = 'k-'
    alpha = .4

    i = next(iterator)
    for j in series:
        plt.plot([j, j], [i, i+1], style, alpha=alpha)


def _get_next_row():
    """Used by plot_barcode"""
    i = -1

    while True:
        i += 1
        yield i

