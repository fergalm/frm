# -*- coding: utf-8 -*-

"""
Created on Mon Nov 19 16:39:13 2018

A much faster PRF fitter, with the caveat that the psf model is hardcoded.

psffit.py can fit an arbitrary PSF model to an image.
The cost of this flexibility
is that it must perform numerical intergration to calculate
the flux in each pixel.
This is slow. On my test machine, a 10x12 image takes 20ms to compute.

Since by far the most common model to fit is that of a symmetric
Gaussian function with
a constant sky background, and this model can be computed quite
quickly, this module
enables this special case to be run much faster. On the same machine,
the same image
can be computed in 95.7us, or a x200 speed up. There's still more speed up to
be had if you make a Model() class that assigns memory for the model
once and overwrites
it each time instead of computing from scratch in each call.

The downside is that none of the code is shared with the general purpose code.
Efforts to use numba don't seem to help much for some reason



@author: fergal
"""

from .abstractprf import AbstractPrfModel
from .abstractprf import Bbox

from scipy.special import erf
from typing import Sequence
import numpy as np

# Precompute for speed
sqrt2 = np.sqrt(2)



class FastGaussianModel(AbstractPrfModel):
    """Fit a Symmetric Gaussian PSF to an image, really quickly

    Most PRF models involves some form of numerical integration over a 2d 
    function. If you make the simplfying assumption that the images are 
    Gaussian in shape you can take advantage of the fact that integrals 
    of the Gaussian are precomputed on your computer and available through
    the special "erf" function.

    On a test machine, the `computeModel` method of this function can be 
    computed in <100us, more than x200 faster than looking up the PRF 
    models used by Kepler. It's very fast.

    It's also a good default choice. Unless you care in detail about the 
    shape of your residuals, or your PRFs are highly asymmetric, the 
    advantages of a more complicated model are not obvious.
    """

    def get(self, bbox:Bbox, params:Sequence):
        """Compute model flux for an image with size (numCols, numRows)

        Inputs
        -------
        numCols, numRows
            (ints) Shape of the image to compute the model PRF for
        params
            (tuple or array) Tunable parameters of the model


        The parameters are 
        `[col, row, flux, sigma, skyLevel]`

        Returns
        ----------
        A 2d numpy array representing the model PRF image.
        """
        numRows, numCols = bbox.shape
        model = np.zeros((numRows, numCols))

        xc = np.arange(numCols)
        xr = np.arange(numRows)
        cols, rows = np.meshgrid(xc, xr)

        model = analytic_gaussian_integral(cols, rows, *params)

        return model

    def getDefaultBounds(self, bbox: Bbox):
        nr, nc = bbox.shape
        bounds = [
            (0, nc),   #col within the width of the image
            (0, nr),    #row within the height of the image
            (None, None),   #No limits on flux level
            (0.2, 5),
            (None, None),   #no limits on sky level
        ]
        return bounds


def analytic_gaussian_integral(col, row, col0, row0, flux0, sigma0, sky):

    z_col1 = 0.5 * (col - col0) / sigma0
    z_col2 = 0.5 * (col + 1 - col0) / sigma0

    z_row1 = 0.5 * (row - row0) / sigma0
    z_row2 = 0.5 * (row + 1 - row0) / sigma0

    flux = flux0
    flux *= phi(z_col2) - phi(z_col1)
    flux *= phi(z_row2) - phi(z_row1)
    flux += sky
    return flux


def phi(z):
    """Compute integral of gaussian function in the range (-Inf, z],
    `z` is defined as (x - x0) / sigma, where x0 is the central value
    of the Gaussian.

    See `scipy.special.erf` for details
    """

    return 0.5 * (1 + erf(z / sqrt2))

