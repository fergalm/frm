# -*- coding: utf-8 -*-

"""
Some matplotlib normalization objects with discrete colourbars.

Mapping values to colours in matplotlib (e.g for scatter plots or heatmaps)
is a two step process. First the values are mapped to the range [0,1], then
the normalised values are mapped to individual colours. The first step
is performed by a normalisation object, the second by a colour map.

Matplotlib comes with two useful normalization objects, a linear and a
logarithm norm. These map the input values smoothly between zero and one.
However, for many applications, it is more visually useful to map input
values to a small number of discrete ranges (e.g. zero to twenty all gets
mapped to a single colour, twenty to fourty to another colour, etc.)

The normalization objects in this module provide two options for discrete
normalziation; Linear, and Histogram Equalisation. Linear creates a mapping
where the range of values covered by each colour is equal. Histogram
Equalisation chooses the ranges so the same number of data points fall
within in each bin.
"""

from __future__ import print_function
from __future__ import division

import matplotlib.colors as mcolors
import numpy as np

eps = np.finfo(np.float32).eps

class DiscreteNorm(mcolors.BoundaryNorm):
    """Scale a collection of numbers to lie in the range [0,1]
    so that the normalized values can only take `ncolors` discrete values

    This is useful for creating colorbars that are segmented into a
    small number of discrete colours. Small numbers of colours (<8)
    are easier to distinguish than the default hundrets of colours.

    This class maps the input to output range in a linear fashion.
    More complicated approaches are also possible

    Example
    ---------
    ::

        y = np.random.randn(1000)
        ncolors = 5  #Show 7 discrete colors in colorbar
        norm = DiscreteNorm(ncolors)
        plt.scatter(x, y, norm=norm)

    """

    def __init__(self, ncolors, vmin=None, vmax=None, clip=False):
        self.clip = clip
        self.ncolors = ncolors
        self.vmin = vmin
        self.vmax = vmax
        self.boundaries = None

    def autoscale_None(self, A):
        """autoscale only None-valued vmin or vmax."""
        A = np.asanyarray(A)
        if self.vmin is None and A.size:
            self.vmin = A.min()
        if self.vmax is None and A.size:
            self.vmax = A.max()

        if self.boundaries is None:
            self.boundaries = self.computeBoundaries(A)


    def computeBoundaries(self, x):
        return np.linspace(self.vmin, self.vmax, self.ncolors+1)

    def __call__(self, values, clip=None):
        if clip is None:
            clip = self.clip

        xx, is_scalar = self.process_value(values)

        #Perform clipping if necessary
        if clip:
            np.clip(xx, self.vmin, self.vmax, out=xx)

        self.autoscale_None(xx)

        #Set norm values
        scale = float(len(self.boundaries) -2 )
        iret = np.zeros(xx.shape, dtype=float)
        for i, b in enumerate(self.boundaries):
            iret[xx > b] =  (i / scale)
        iret *= .999999

        #Set norm values for out-of-bounds inputs
        iret[xx < self.vmin] = -1
        iret[xx > self.vmax] = 2

        #Convert to masked array?
        mask = np.ma.getmaskarray(xx)
        ret = np.ma.array(iret, mask=mask)

        if is_scalar:
            return ret[0]

        return ret


class FixedDiscreteNorm(DiscreteNorm):
    """Set the boundaries between the colours directly"""
    def __init__(self, boundaries, vmin=None, vmax=None, clip=False):
        self.clip = clip
        self.vmin = vmin
        self.vmax = vmax
        self.boundaries = np.array(boundaries)
        self.ncolors = len(boundaries)

class PercentileNorm(DiscreteNorm):
    """Not tested"""
    def __init__(self, levels, vmin=None, vmax=None, clip=False):
        self.clip = clip
        self.vmin = vmin
        self.vmax = vmax
        self.pct_levels= levels

    def computeBoundaries(self, x):
        eps = np.finfo(np.float32).eps
        x = np.array(x)  #Strip out array masking if necessary
        x = x[ (x >= self.vmin) & (x <= self.vmax)]

        boundaries = np.percentile(x, self.pct_levels)
        boundaries[-1] += eps
        boundaries = fixEqualValueBoundaries(boundaries)
        assert np.all(np.diff(boundaries) > 0), "Logic Error"
        return boundaries


class DiscreteLog10Norm(DiscreteNorm):
    def __init__(self, ncolors, vmin=eps, vmax=None, clip=False):
        DiscreteNorm.__init__(self, ncolors, vmin, vmax, clip)

    def computeBoundaries(self, x):
        x = np.array(x)
        x = x[ (x >= self.vmin) & (x <= self.vmax)]

        if np.any(x <=0):
            raise ValueError("Can't apply log normalisation to negative numbers")

        lwr = np.log10(np.min(x))
        upr = np.log10(np.max(x) + eps)
        boundaries = np.logspace(lwr, upr, self.ncolors+1)
        return boundaries


class HistEquNorm(DiscreteNorm):
    """Create a DiscreteNorm object with thresholds chosen so that each colour
    is equally represented in the figure. This can sometimes expose
    detail hidden by a linear colourmap.

    While DiscreteNorm() produces equally spaced thresholds between colors
    this object chooses the thresholds so that equal numbers of the input
    data points appear in each colour 'bin'. This is useful, for example,
    for datasets drawn from power law distributions.


    Example
    ---------
    ::

        y = np.random.randn(1000)
        ncolors = 5  #Show 7 discrete colors in colorbar
        norm = HistEquNorm(y, ncolors)
        plt.scatter(x, y, norm=norm)
        plt.colorbar(spacing='proportional'
    """

    def computeBoundaries(self, x):
        x = np.array(x)  #Strip out array masking if necessary
        # eps = np.finfo(np.float32).eps
        thresholds = np.linspace(0, 100, self.ncolors+1)

        x = x[ (x >= self.vmin) & (x <= self.vmax)]
        boundaries = np.percentile(x, thresholds)
        boundaries[-1] += eps

        boundaries = fixEqualValueBoundaries(boundaries)
        assert np.all(np.diff(boundaries) > 0), "Logic Error"
        return boundaries


def fixEqualValueBoundaries(boundaries):
    """Adjust boundary values so no pair are equal

    For strange distributions of data it can happen that some of the
    boundaries chosen by percentiles (e.g in HistEquNorm.computeBoundaries)
    have equal values. This messes up the colorbar a bit.
    This function identifies when this issue occurs and adjusts
    the boundary values in nice ways to make the colourbar look
    prettier.

    It operates in two passes, a forward and reverse pass.
    The algorithm is quite slow in python (no vectoring) but
    it is typically run on arrays with fewer than 10 elements,
    so speed isn't much of an issue.

    Inputs
    ---------
    boundaries
        (list or array) Sorted list of boundary values

    Returns
    -----------
    Array of same length as input. Values in the output are
    adjusted to avoid duplicate values.
    """

    boundaries = _fixEqualValueBoundaries(boundaries)
    b = _fixEqualValueBoundaries(boundaries, reverse=True)
    return b


def _fixEqualValueBoundaries(boundaries, reverse=False):
    """See fixEqualValueBoundaries"""
    boundaries = np.array(boundaries).astype(float)

    #Check sorted
    assert np.all(np.diff(boundaries) >= 0)

    if reverse:
        boundaries = -1 * boundaries[::-1]

    diff = np.diff(boundaries)
    i = 0
    while i < len(diff):
        if diff[i] > 0:
            i += 1
            continue

        j = i+1
        while j < len(diff) and diff[j] == 0:
            j += 1

        if j == len(diff):
            #We've hit the end of the array, so stop
            return boundaries

        delta = boundaries[j+1] - boundaries[i]
        size = j - i
        boundaries[i:j+1] += np.arange(size+1) * delta / float(size+1)

        i = j

    if reverse:
        #Un-reverse
        boundaries = -1 * boundaries[::-1]
    return boundaries

#
