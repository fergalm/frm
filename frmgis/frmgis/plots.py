from ipdb import set_trace as idebug
from pdb import set_trace as debug

import matplotlib.collections as mcollect
import matplotlib.colors as mcolors
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import matplotlib as mpl

from frmgis.anygeom import AnyGeom
import frmplots.plots as fplots
import frmplots.norm as fnorm
import frmbase.support

import pandas as pd
import numpy as np

npmap = frmbase.support.npmap


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
    cmap  = kwargs.pop('cmap', fplots.DEFAULT_CMAP)
    wantCbar = kwargs.pop('wantCbar', True )
    ticklabel_format = kwargs.pop('ticklabel_format', '%i')


    #Complicated way of picking the norm. Default is linear. User can specify
    #'hist', and we'll compute a histequ norm. Otherwise, treat the passed arg
    #as a norm object.
    default_norm = mcolors.BoundaryNorm(np.linspace(vmin, vmax, nstep), cmap.N)
    norm = kwargs.pop('norm', default_norm )

    patch_list = []
    for p,v in zip(polygons, values):
        ap = AnyGeom(p)
        patch = ap.as_patch(facecolor=cmap(norm([v])[0]), **kwargs)
        patch_list.extend(patch)
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


def plot_shape(shape, *args, **kwargs):
    """Plot the outline of a single shape

    This is a helpful one-off function. If you have a lot of shapes to draw
    look at `plot_shape_collection()` instead.
    
    
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


def plot_shape_collection(geoms, *ars, **kwargs):
    """Plot multiple shapes

    If you only have the one shape to plot, look at `plot_shape()` instead.
    
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

    #Setup some defaults
    if 'facecolor' not in kwargs:
        kwargs['facecolor'] = "none"

    if 'edgecolor' not in kwargs:
        kwargs['edgecolor'] = 'C0'
        
    if 'lw' not in kwargs:
        kwargs['lw'] = 2 
        
    ##Work around matplotlib bug. Patches won't display unless something 
    ##else is plotted to the screen.
    #g0 = geoms[0]
    #_, points = AnyGeom(g0).as_array()
    #plt.plot(points[:,0], points[:,1], 'w.', zorder=-100) #color='r', lw=1)
    
    from tqdm import tqdm
    bbox_list = []
    patch_list = []
    for g in tqdm(geoms):
        ap = AnyGeom(g)
        patch = ap.as_patch(**kwargs)
        bbox_list.append( patch[0].get_extents() )
        patch_list.extend(patch)
    pc = mcollect.PatchCollection(patch_list, match_original=True)

    #Plot the patchlist
    ax = plt.gca()
    ax.add_collection(pc)

    from matplotlib.transforms import Bbox
    bbox = Bbox.union(bbox_list)
    plt.plot([bbox.x0, bbox.x1], [bbox.y0, bbox.y1], 'w.', zorder=-100)
