from ipdb import set_trace as idebug
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

import  frmplots.norm as fnorm 

"""
A cartogram is a map where the drawn sizes of geographic regions are 
based on some property of the data other than their area in square kilometres.

See https://en.wikipedia.org/wiki/Cartogram

This implementation creates maps where every geographic element
is drawn as a square of equal size. These squares are coloured by
the property you wish to plot.

Based on plotKeplerFocalPlane() from the Kepler days, but adapted
to be more general.

"""

class AbstractCartogram():
    """Extend this class to create new cartograms

    Extending the class if often as simple as setting the paramets
    of the `SHAPES` variable.

    SHAPES is a dictionary matching region names to 
    """
    SHAPES  = dict()

    def __init__(self):
        pass 

    def plot(self, names, values, **kwargs):
        cmap = kwargs.pop('cmap', plt.cm.viridis)
        vmin = kwargs.pop('vmin', np.min(values))
        vmax = kwargs.pop('vmax', np.max(values))
        norm = kwargs.pop('norm', fnorm.DiscreteNorm(7, vmin, vmax))
        fmt = kwargs.pop('fmt', "%g")

        ax = plt.gca()
        assert len(names) == len(values)
        for i in range(len(names)):
            n = names[i]
            v = values[i]

            #TODO do I need an artist collection?
            clr = cmap(norm(v))
            c, r = self.shape_lookup(n)
            patch = plt.Rectangle((c,r), .9, .8, color=clr)
            ax.add_patch(patch)

            label_square(c, r, v, fmt, clr)  #Add the value in the square
            plt.text(c+.5, r+.82, n, ha="center")  #Add placename on top

        #Remove the axis ticks
        plt.axis([0, 9, 0, 6])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        cb = make_colorbar(ax, cmap, norm)
        return cb

    def shape_lookup(self, name):
        shapes = self.SHAPES
        if isinstance(name, str):
            return shapes[name]
        elif isinstance(name, int):
            return list(shapes.values())[name]
        else:
            raise KeyError("Name must be int or string")


class KeplerFocalPlane(AbstractCartogram):
    pass 


class MarylandCounties(AbstractCartogram):
    SHAPES = {
        "Allegany": (1,5),
        "Anne Arundel" : (5,3),

        "Baltimore City": (5,4),
        "Baltimore County": (5,5),

        "Calvert": (5,2),
        "Caroline": (8,2),
        "Carroll": (4,5),
        "Cecil": (7,5),
        "Charles": (4,2),

        "Dorchester": (7,1),

        "Frederick": (3,5),
        "Garett": (0, 5),
        "Harford": (6,5),
        "Howard": (4,4),
        "Kent": (7,4),
        "Montgomery": (3,4),
        "Prince George's": (4,3),
        "Queen Anne's": (7,3),
        "Somerset": (7, 0),
        "St. Mary's": (5,1),
        "Talbot": (7, 2),

        "Washington": (2,5),
        "Wimcomico": (8, 1),
        "Worcester": (8,0),
    }

    def plot(self, names, values, **kwargs):
        AbstractCartogram.plot(self, names, values, **kwargs)
        plt.plot([3.5], [3.5], '*', ms=28)


class BaltimoreCountyCensusTracts(AbstractCartogram):
    pass 


class BaltimoreCountyCensusPlaces(AbstractCartogram):
    pass 


def label_square(col, row, val, fmt, clr):
    textcolor='w'
    if np.prod(clr) > .1:
        textcolor='k'

    text = fmt %(val)
    plt.text(col+.45, row+.4, text, \
        va="center", 
        ha="center", 
        bbox=None, 
        color=textcolor,
        fontsize=26,
    )


def make_colorbar(ax, cmap, norm):
    #Create a colorbar. Because we don't call scatter or imshow
    #this is more involved than usual
    cax, kw = mpl.colorbar.make_axes(ax)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

    #Set the main panel as the default axis again (otherwise)
    #future plotting will happen on the colourbar itself
    plt.sca(ax)
    return cb


def test_maryland():
    md = MarylandCounties()

    names = list(md.SHAPES.keys())
    values = np.arange(24)

    plt.clf()
    md.plot(names, values)