#Copyright 2017-2018 Orbital Insight Inc., all rights reserved.
#Contains confidential and trade secret information.
#Government Users: Commercial Computer Software - Use governed by
#terms of Orbital Insight commercial license agreement.

"""
Created on Fri Feb 23 14:49:23 2018

@author: fergal


"""
from __future__ import print_function
from __future__ import division

#import matplotlib.pyplot as plt
#import matplotlib as mpl
#import pandas as pd
import numpy as np

from scipy.special import sph_harm
#import AbstractFitter


#This paper suggests improvements over using least squares fitting
#http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1988QJRAS..29..129B&amp;data_type=PDF_HIGH&amp;whole_paper=YES&amp;type=PRINTER&amp;filetype=.pdf

class SphHarm(object):
    def __init__(self, lng_deg, lat_deg, values, unc, maxEll):
        """Fit spherical harmonic functions to some data

        WARNING
        -------------
        This class hasn't been tested yet

        Inputs
        ------
        lng_deg
        lat_deg
            (1d np arrays) Input longitude and latitude in degrees
        values
            (1d np arrays) Values at each lng/lat pair to fit. The mean of f need **not** be zero
        maxEll
            (int) Largest spherical degree to fit. The number of fitting functions is larger than ``maxEll``
            because each m value is fitted too. For example, if ``maxEll = 2``, Y00, Y10, Y11, Y20, Y21 and Y22
            are all fit

        """

        self.lng_deg = lng_deg
        self.lat_deg = lat_deg
        self.values = values
        self.unc = unc
        self.maxEll = maxEll

        self.check_inputs()
        self.fit()


    def check_inputs(self):

        if self.maxEll < 1:
            raise ValueError("maxEll must be +ve integer")

        if len(self.lng_deg) != len(self.lat_deg):
            raise ValueError("lng, lat must be same length")

        if len(self.lng_deg) != len(self.values):
            raise ValueError("lng, values must be same length")

        if self.unc is not None:
            raise NotImplementedError("Weighted fit not handled yet")

        if self.unc is None:
            self.unc = np.ones_like(self.lng_deg, dtype=float)


    def fit(self):
        """Perform the fit

        Notes
        --------
        * Calls ``np.linalg.lstsq`` which calls Lapack's ``dgelsd``, which in turn performs SVD decomposition
        """
        lng_deg = self.lng_deg
        lat_deg = self.lat_deg
        dataSize = len(lng_deg)
        values = self.values
        maxEll = self.maxEll

        #Build up fit matrix
        #Ac = f, where c are the fit coeffs and f is the data to fit
        numVec = 1 + maxEll * (maxEll + 2)
        A = np.empty( (dataSize, numVec) )

        vecNum = 0
        for ell in range(maxEll+1):
            for em in range(-ell, ell+1):
                A[:, vecNum] = computeYlm(ell, em, lng_deg, lat_deg)
                vecNum += 1

        #Doing a weighted fit should be as simple as dividing A, and values by self.s. But this statement
        #needs to be tested.


        self.Amatrix = A
        self.param = np.linalg.lstsq(A, values)[0]


    def getParams(self):
        """Get coefficients of the fit


       Returns
        ---------
        ceoff
            (2d numpy array). ``coeff[i, j]`` is the fitting coefficient for :math:`Y^{\ell}_{m} = Y^{i}_{j}`.
            The array is arranged so that the coefficient for (ell, m) = (2, -2) is stored at ``coeff[2, -2]``

        """
        self.check_fit_complete()
        maxEll = self.maxEll

        i = 0
        coeffArray = np.zeros( (maxEll+1, 2*maxEll + 1) )
        for ell in range(maxEll+1):
            for m in range(-ell, ell+1):
                coeffArray[ell, m] = self.param[i]
                i+=1


    def getBestFit(self):
        self.check_fit_complete()
        bestFit = np.matmul(self.Amatrix, self.param)
        return bestFit


    def getResiduals(self):
        self.check_fit_complete()
        bestFit = self.getBestFit()
        return self.values - bestFit


    def check_fit_complete(self):
        try:
            self.param
            self.Amatrix
        except AttributeError, e:
            raise AttributeError("Fit hasn't been performed yet: %s" %(e))



def computeYlm( ell, em, lng_deg, lat_deg):
    """Computes real-form spherical harmonics

    Scipy computes spherical harmonics in complex form. But if we are fitting a real scalar field
    to the sphere, we need to convert these complex values to a set of real, orthogonal vectors

    Inputs
    ------
    ell, em
        (int) Angular momentum and azimuthal order of spherical harmonic to fit. |em| < ell
    lng_deg
    lat_deg
        (1d np arrays) Input longitude and latitude in degrees

    Notes
    -------
    * Based on https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form

    * We usually deal with lng/lat, but spherical harmonics work on azimuth and polar angle (or zenith distance,
      angular distance from the North pole). We convert between them internally here to avoid the
      need for the user to do this themselves.
    * Scipy uses a weird convention of \theta for azimuthal angle and \phi for polar angle. To avoid
      confusion we use az for azimuthal angle and zd (or zenith distance) for polar angle.

    """

    if abs(em) > ell:
        raise ValueError(" |em| > ell (%i > %i)" %(em, ell))

    az_rad, zd_rad = lnglat_to_azzd_rad(lng_deg, lat_deg)

    ylm = sph_harm(abs(em), ell, az_rad, zd_rad)

    if em < 0:
        return np.sqrt(2) * (-1)**em * np.imag(ylm)
    elif em == 0:
        return np.real(ylm)
    else:
        return np.sqrt(2) * (-1)**em * np.real(ylm)


#Better belongs with a base class module?
def lnglat_to_azzd_rad(lng_deg, lat_deg):
    """Convert longitudes and latitudes in degrees to radians as expected by scipy's spherical harmonic function


    Inputs
    ----------
    lng_deg
    lat_deg
        (1d np arrays) Input longitude and latitude in degrees

    Returns
    -------
    az, zd
        (1d np arrays) Azimuths and Zenith Distances in radians

    Notes
    --------
    * Longitude for points west of Greenwich can be expressed as either <0 or > 180 degrees. Function will
    accept either.
    * Zenith Distance is zero at the north pole, while latitude is 90 degrees.

    """

    #Handle both conventions for longitude. Points west of greenwich can have either negative longitudes
    #or values > 180 degrees.
    if np.min(lng_deg) < 0:
        lng_deg += 180

    #Azimuth (az) is angle along the equator
    #Zenith Distance (zd), or co-altitude, is angle from north pole
    az_rad = np.radians(lng_deg)
    zd_deg = 90 - lat_deg
    zd_rad = np.radians(zd_deg)


    return az_rad, zd_rad

