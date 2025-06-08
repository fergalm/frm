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
        mcolors.Normalize.__init__(self, vmin, vmax, clip)
 
        self._clip = clip
        self.ncolors = ncolors
        self._vmin = vmin
        self._vmax = vmax
        self.bounds = None

    @property
    def boundaries(self):
        if self.bounds is None:
            if self._vmin is None or self._vmax is None:
                raise ValueError("Unusual error in norm. Please set vmin and vmax")
            self.bounds = self.computeBoundaries()
        return self.bounds


    def autoscale_None(self, A):
        """autoscale only None-valued vmin or vmax."""
        # import ipdb; ipdb.set_trace()
        A = np.asanyarray(A)
        if self._vmin is None and A.size:
            self._vmin = A.min()
        if self._vmax is None and A.size:
            self._vmax = A.max()

        if self.bounds is None:
            self.bounds = self.computeBoundaries(A)


    def computeBoundaries(self, x=None):
        return np.linspace(self._vmin, self._vmax, self.ncolors+1)

    def __call__(self, values, clip=None):
        if clip is None:
            clip = self._clip

        xx, is_scalar = self.process_value(values)

        #Perform clipping if necessary
        if clip:
            np.clip(xx, self._vmin, self._vmax, out=xx)

        self.autoscale_None(xx)

        #Set norm values
        scale = float(len(self.bounds) -2 )
        iret = np.zeros(xx.shape, dtype=float)
        for i, b in enumerate(self.bounds):
            iret[xx > b] =  (i / scale)
        iret *= .999999

        #Set norm values for out-of-bounds inputs
        iret[xx < self._vmin] = -1
        iret[xx > self._vmax] = 2

        #Convert to masked array?
        mask = np.ma.getmaskarray(xx)
        ret = np.ma.array(iret, mask=mask)

        if is_scalar:
            return ret[0]

        return ret

    def inverse(self, value):
        raise ValueError("Not invertable")
#   def __call__(self, value, clip=None):
#         """
#         Normalize *value* data in the ``[vmin, vmax]`` interval into the
#         ``[0.0, 1.0]`` interval and return it.

#         Parameters
#         ----------
#         value
#             Data to normalize.
#         clip : bool
#             If ``None``, defaults to ``self.clip`` (which defaults to
#             ``False``).

#         Notes
#         -----
#         If not already initialized, ``self.vmin`` and ``self.vmax`` are
#         initialized using ``self.autoscale_None(value)``.
#         """
#         if clip is None:
#             clip = self.clip

#         result, is_scalar = self.process_value(value)

#         self.autoscale_None(result)
#         # Convert at least to float, without losing precision.
#         (vmin,), _ = self.process_value(self.vmin)
#         (vmax,), _ = self.process_value(self.vmax)
#         if vmin == vmax:
#             result.fill(0)   # Or should it be all masked?  Or 0.5?
#         elif vmin > vmax:
#             raise ValueError("minvalue must be less than or equal to maxvalue")
#         else:
#             if clip:
#                 mask = np.ma.getmask(result)
#                 result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
#                                      mask=mask)
#             # ma division is very slow; we can take a shortcut
#             resdat = result.data
#             resdat -= vmin
#             resdat /= (vmax - vmin)
#             result = np.ma.array(resdat, mask=result.mask, copy=False)
#         if is_scalar:
#             result = result[0]
#         return result

class FixedDiscreteNorm(DiscreteNorm):
    """Set the boundaries between the colours directly"""
    def __init__(self, boundaries, vmin=None, vmax=None, clip=False):
        self.mclip = clip
        self.mvmin = vmin or np.min(boundaries)
        self.mvmax = vmax or np.max(boundaries)
        self.bounds = np.array(boundaries)
        self.ncolors = len(boundaries)

    def computeBoundaries(self, x=None):
        return self.bounds

        
class PercentileNorm(DiscreteNorm):
    """Not tested"""
    def __init__(self, levels, vmin=None, vmax=None, clip=False):
        DiscreteNorm.__init__(self, levels, vmin, vmax, clip)
        self.mclip = clip
        self.mvmin = vmin
        self.mvmax = vmax

        if isinstance(levels, int):
            levels = np.linspace(0, 100, levels+1)
        self.pct_levels= levels


    def computeBoundaries(self, x):
        eps = np.finfo(np.float32).eps
        x = np.array(x)  #Strip out array masking if necessary

        vmin = self.mvmin or np.min(x)
        vmax = self.mvmax or np.max(x)

        x = x[ (x >= vmin) & (x <= vmax)]

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
        x = x[ (x >= self._vmin) & (x <= self._vmax)]

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
        thresholds = np.linspace(0, 100, self.ncolors+1)

        x = x[ (x >= self._vmin) & (x <= self._vmax)]
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
