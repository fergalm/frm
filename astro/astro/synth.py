# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 07:24:17 2020

@author: fergal
"""

from ipdb import set_trace as idebug
from pdb import set_trace as debug
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import const

import scipy.integrate as spInt
#import synphot.units as su
import astropy.units as u


def synthPhot(model_wavel, model_flux, filter_wavel, filter_transmission):
    """Perform synthetic photometry on a model

    Inputs
    ---------
    model_wavel
        (1d np array) Wavelenth of model. Any units
    model_flux
        (1d np array) Flux at each wavelength, any units
    filter_wavel
        (1d np array) Wavelenghts of filter. Must be same units
        as model_wavel, but need not be evaulated at the same values
    filter_transmission
        (1d np array). Fraction of light transmitted by the filter
        at this wavelength. Values should be in the range [0,1]

    The wavelength range covered by the model must bracket the
    wavelength range covered by the filter. You may need to
    trim the ends of your filter (if they are all zeros) to
    make this pre-condition true.

    This function works best if the filter is sampled at least
    as frequently as the model. If this is not true, you should
    interpolate the filter to the model wavelength grid before
    passing it to this function.
    """
    #Check object spectrum brackets the filter
    assert np.all(np.diff(filter_wavel) > 0)
    assert np.all(np.diff(model_wavel) > 0)

    #Check the model spans the range of the filter
    assert model_wavel[0] < filter_wavel[0]
    assert filter_wavel[-1] < model_wavel[-1]

    assert np.min(filter_transmission) >= 0
    assert np.max(filter_transmission) <= 1

    model_flux = np.interp(filter_wavel, model_wavel, model_flux)
    transmitted_flux = model_flux * filter_transmission

    flux = spInt.trapz(transmitted_flux, filter_wavel)
    flux /= spInt.trapz(filter_transmission, filter_wavel)
    return flux



def convertKoesterToJy(data, radius_m, dist_parsec):
    """Convert Koester's file so the wavelengths are in metres
    and the fluxes in Janskys
    """

    wavelength_ang = data[:,0]
    flux = data[:,1]

    wavel_m = wavelength_ang * 1e-10

    # Koester's flux  (4x Eddington) = 4*PI*Intensity
    # Intensity is flux per solid angle. Flux is Intensity
    # integrated over half a solid angle = PI*Intensity
    flux *= 4

    # Convert from flux in erg/cm^2/s/cm to W/m^2/m
    flux *= 1e-1

    # Convert parsecs to metres
    dist_m = dist_parsec * const.parsec

    flux *= (radius_m / dist_m) ** 2

    flux = convertFlambdaToFnu(wavel_m, flux)
    flux /= const.jansky

    data[:,0] = wavel_m
    data[:,1] = flux
    return data


def convertNextGenToJy(st, radius_m, dist_parsec):
    """Convert NextGen models by Allard to Jansky

    These models are described in Hautschild et al 1999
    """
    w = st[:,0]
    f = st[:,1]

    w = w * u.angstrom
    f = f * u.erg  / u.second / u.cm**2 / u.cm

    fjy = su.convert_flux(w, f, u.jansky)
    radius = .2  * u.solRad
    dist = 10 * u.parsec

    fjy *= (radius/dist)**2

    out = np.empty_like(st)
    out[:,0] = w.to(u.meter)
    out[:,1] = fjy.to(u.jansky)
    return out



def convertKuruczToJy(st):
    """
    Convert Kurucz models to Jy.

    The best ref for these I've found is

    https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/k93models/AA_README

    The NextGen models have finer spectral resolution
    """
    w = st[:,0]
    f = st[:,1]

    w = w * u.angstrom
    f = f * su.FLAM

    fjy = su.convert_flux(w, f, u.jansky)
    radius = .2  * u.solRad
    dist = 10 * u.parsec

    fjy *= (radius/dist)**2

    out = np.empty_like(st)
    out[:,0] = w.to(u.meter)
    out[:,1] = fjy.to(u.jansky)
    return out


def convertFlambdaToFnu(wavel_m, flux_lam):

    factor = (wavel_m **2) / const.speedOfLight
    flux_nu = flux_lam * factor
    return flux_nu



def sumModels(w1, f1, w2, f2):
    """Some two spectra, accounting for differences in wavelenth reticules.
    
    This is quick an dirty, you may want something more sophisticated
    """
    assert len(w1) == len(f1)
    assert len(w2) == len(f2)
    f3 = np.interp(w1, w2, f2)
    out = np.vstack([w1, f1+f3]).transpose()
    return out




