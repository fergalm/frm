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
        patch = AnyGeom(p).as_patch(facecolor=cmap(norm([v])[0]), **kwargs)
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
