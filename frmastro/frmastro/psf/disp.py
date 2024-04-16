from __future__ import print_function
from __future__ import division

from typing import Callable, Iterable

import matplotlib.colors as mcolor
import matplotlib.pyplot as plt
import numpy as np

"""
Tools for displaying Astronomical images
"""



def plotImage(img, **kwargs):
    """Plot an image in linear scale

    Inputs
    --------
    img
        (2d np array) Image to plot

    Optional Inputs
    -----------------
    log
        (bool) Plot the image in log scalings. (Default False)
    origin
        (str) If 'bottom', put origin of image in bottom left
        hand corner (default).
        If 'top', but it in top left corner

    interpolation
        (str) Interpolation method. Default is nearest. See `plt.imshow`
        for more options

    cmap
        (plt.cm.cmap) Color map. Default is YlGnBu_r

    extent
        (4-tuple) Extent of image. See `plt.imshow` for more details

    All other optional arguments passed to `plt.imshow`


    Returns
    ----------
    **None**

    Output
    -------
    A plot is returned
    """
    if "origin" not in kwargs:
        kwargs["origin"] = "lower"

    if "interpolation" not in kwargs:
        kwargs["interpolation"] = "nearest"

    if "cmap" not in kwargs:
        kwargs["cmap"] = plt.cm.YlGnBu_r

    if "norm" not in kwargs:
        kwargs["norm"] = mcolor.Normalize()

    if "extent" not in kwargs:
        shape = img.shape
        extent = [0, shape[1], 0, shape[0]]
        kwargs["extent"] = extent

    colorbar = kwargs.pop('colorbar', True)

    mask = kwargs.pop('mask', None)
    showValues = kwargs.pop("showValues", False)
    log = kwargs.pop("log", False)

    if log:
        img = img.copy()
        mn = np.min(img)
        if mn < 0:
            offset = -1.1 * mn
            img += offset
        img = np.log10(img)

    axim = plt.imshow(img, **kwargs)
    if mask is not None:
        cm = plt.cm.Reds 
        cm.set_under('#00FF0000')
        cm.set_over('#FF0000FF')
        plt.imshow(mask, vmin=0.4, vmax=.6, cmap=cm, origin=kwargs['origin'], extent=kwargs['extent'])
        plt.sci(axim)


    if showValues:
        showPixelValues(img, kwargs["cmap"], kwargs["norm"])

    if colorbar:
        plt.colorbar()




def showPixelValues(img, cmap, norm, fmt="%i"):
    """Print flux values of pixel on the image 

    Inputs
    ------------
    img
        A 2d numpy array representing an image 
    cmap
        The colormap used in the image 
    norm
        The normalisation scheme used for the colormap in the image
    fmt
        The format string used to write the numbers 

    Returns
    ---------
    **None**

    """
    nr, nc = img.shape
    for i in range(nc):
        for j in range(nr):
            clr = cmap(norm(img[j, i]))

            textcolor = "w"
            if np.prod(clr) > 0.2:
                textcolor = "k"

            txt = fmt % (img[j, i])
            plt.text(i + 0.5, j + 0.5, txt, color=textcolor, ha="center")


def plotDifferenceImage(img, **kwargs):
    """Plot a difference image.

    The colour bar is chosen so zero flux is at the centre of the colour map


    Inputs
    ------------
    img
        A 2d numpy array representing an image 
    vmax
        Max (and -min) value to display in the colormap.

        
    All other arguments are passed to `plt.plot`

    Returns
    ----------
    **None**
        
    """
    if "origin" not in kwargs:
        kwargs["origin"] = "lower"

    if "interpolation" not in kwargs:
        kwargs["interpolation"] = "nearest"

    if "cmap" not in kwargs:
        kwargs["cmap"] = plt.cm.RdBu_r

    if "extent" not in kwargs:
        shape = img.shape
        extent = [0, shape[1], 0, shape[0]]
        kwargs["extent"] = extent

    vm = kwargs.pop('vmax', None)
    if vm is None:
        vm = max(np.fabs([np.min(img), np.max(img)]))

    plt.imshow(img, **kwargs)
    plt.colorbar()
    plt.clim(-vm, vm)


def plotDiffImage(img, **kwargs):
    """Mneumonic"""
    return plotDifferenceImage(img, **kwargs)


def plotCentroidLocation(col:float, row:float, **kwargs):
    """Add a point to the an image, with sensible defaults

    Inputs
    -----------
    col, row (floats)
        Column and row to mark 
    
        
    All other arguments are passed to `plt.plot`

    Returns
    ----------
    **None**
    """
    ms = kwargs.pop("ms", 9)
    mfc = kwargs.pop("mfc", "None")
    mec = kwargs.pop("mec", "white")
    mew = kwargs.pop("mew", 1)
    marker = kwargs.pop("marker", "o")

    #plt.plot([col], [row], ms=ms + 1, **kwargs)

    plt.plot([col], [row], marker=marker, mfc=mfc, mec=mec,
             mew=mew, ms=ms, lw=0, **kwargs)


def threeplot(img:np.ndarray, modelFunc:Callable, guess:Iterable, norm=None, vmax=None):
    """
    Plot an image, a model of that image, and the residual.

    Allow interactive analysis by producing a crosshair cursor
    and messing with the toolbar text 

    Inputs
    ------------
    img
        A 2d numpy array representing an image 
    modelFunc:
        A function that accepts as input 
            numCols (int)
                Number of columns in image to model 
            numRows (int)
                Number of rows in image to model 
            arglist
                A list of parameters describing the model
        modelFunc returns a 2d numpy array of shape (nr, nc)
    guess
        A list of parameters to pass to modelFunc
    norm
        A `matplotlib.colors.Normalize()` object used for setting
        the plotting scale of the image and model displays.

    Returns
    -----------
    A `matplotlib.widgets.MultiCursor` object. Note interactivity
    will only work while this object exists. If you don't store
    it to a variable when exiting a function, the cursors will
    disappear. I've tested multicursors in the QT backend only.
    I expect they will fail in Juypter notebooks.

    Notes
    ----------
    For interactive plots, the toolbar will display the col/row
    position of the cursor, and the pixel values for each three
    plots for that cursor position. This makes inspecting the 
    plots a whole lot easier.

    """
    from .abstractprf import Bbox
    bbox = Bbox.fromImage(img)
    model = modelFunc(bbox, guess)

    diff = img - model 

    ax1 = plt.subplot(131)
    plotImage(img, norm=norm)

    ax2 = plt.subplot(132, sharex=ax1, sharey=ax1)
    plotImage(model, norm=norm)

    ax3 = plt.subplot(133, sharex=ax1, sharey=ax1)
    plotDiffImage(diff, vmax=vmax)

    from matplotlib.widgets import MultiCursor 
    canvas = plt.gcf().canvas
    multi = MultiCursor(canvas, [ax1, ax2, ax3], horizOn=True, vertOn=True,
                        lw=.5, color='r')
    
    func = lambda c, r: _formatToolbarTextFor3Plot(c, r, img, model, diff)
    ax1.format_coord = func
    return multi

def _formatToolbarTextFor3Plot(col:int, row:int, img:np.ndarray, model:np.ndarray, diff:np.ndarray):
    """Private funtion of `threeplot`"""
    c, r = int(col), int(row)
    text = f"Pixel {col:.0f}, {row:.0f}\n Values: {img[r,c]:.2f}, {model[r,c]:.2f}, {diff[r,c]:.2f}" 
    return text 
