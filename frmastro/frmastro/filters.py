from ipdb import set_trace as idebug

from glob import glob 
import numpy as np
import os 

def lmap(function, *array):
    """Perform a map, and package the result into a list, like what map did in Py2.7"""
    return list(map(function, *array))


def loadFilters(pattern=None):
    """
    Load filter tracings

    The filter tracings are stored on disk. This function
    loads them all to a dictionary. The keys are

    Input
    ------
    pattern
        (str) A string that glob can use to match a set of files, e.g
        filters/*.dat. If not given, the function tries to guess the correct
        value.
        
    Returns
    ---------
    A dict of 2d numpy arrays. The columns of the arrays are
        * wavelength (in metres)
        * transparency (0=> opaque, 1=> transparent)
        
    The keys are : {system}_{band}, e.g sloan_u or miri_f1000w. The 
    values are 2d numpy arrays with columns
    """
    if pattern is None:
        pattern = os.path.split(__file__)[0]
        pattern = os.path.join(pattern, 'filter_tracings/*.dat')

    filterFiles = glob(pattern)

    if not filterFiles:
        raise ValueError(f"No filter tracings found in {pattern}. Please specify glob pattern as input arg")
    filterNames = lmap(getFilterName, filterFiles)


    filters = dict()
    for fn in sorted(filterFiles):
        fName = getFilterName(fn)
        data = np.loadtxt(fn)
        filters[fName] = data
    return filters


def getFilterName(fn):
    fName = os.path.split(fn)[-1]
    fName = os.path.splitext(fName)[0]
    return fName


def getZeroPointFluxes_Jy():
    """
    Optimal and NIR taken from 
    http://astroweb.case.edu/ssm/ASTR620/mags.html
    The JHK zeros are from the original paper and may not 
    match 2mass exactly.
    
    
    SDSS is on the AB scale, so the 0 mag in each
    band should be the same value. SDSS u and z are slightly off
    from AB, by amounts listed in the comments. If you
    care about accuracy at the <0.05 level you should
    correct those zero points.
    See https://www.sdss.org/dr12/algorithms/fluxcal/
        
    """
    zero = {
        'johnson-U': 1810,
        'johnson-B': 4260,
        'johnson-V': 3640,
        'johnson-R': 3080,
        'johnson-I': 2550,
        'twomass-J': 1600,
        'twomass-H': 1080,
        'twomass-Ks': 670,
        'sloan-u': 3631,  # u_sdss = u_ab + 0.04
        'sloan-g': 3631,
        'sloan-r': 3631,
        'sloan-i': 3631,
        'sloan-z': 3631, # z_sdss = z_ab - 0.02
    }
    
    zero.update(jwstZeroPointFluxes_Jy())
    return zero


def jwstZeroPointFluxes_Jy():
    """Returns a dictionary of flux in Janskies of Vega in some JWST bands

    Taken from
    http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=JWST&gname2=NIRCam&asttype=

    """
    zero = {
                #NIRCam
                'nircam-F070W' : 2748.39,
                'nircam-F090W' : 2249.58,
                'nircam-F115W' : 1741.77,
                'nircam-F140M' : 1309.05,
                'nircam-F150W' : 1175.1,
                'nircam-F162M' : 1041.79,
                'nircam-F164N' : 1007.79,
                'nircam-F150W2' : 933.57,
                'nircam-F182M' :  858.76,
                'nircam-F187N' : 813.41,
                'nircam-F200W' : 759.59,
                'nircam-F210M' : 701.37,
                'nircam-F212N' : 690.9,
                'nircam-F250M' : 515.84,
                'nircam-F277W' : 430.19,
                'nircam-F300M' : 377.25,
                'nircam-F323N' : 329.21,
                'nircam-F322W2' : 318.55,
                'nircam-F335M' :  305.6,
                'nircam-F356W' : 272.62,
                'nircam-F360M' : 266.13,
                'nircam-F405N' : 212.51,
                'nircam-F410M' : 213.27,
                'nircam-F430M' : 195.51,
                'nircam-F444W' : 184.42,
                'nircam-F460M' : 168.3,
                'nircam-F466N' : 162.02,
                'nircam-F470N' : 163.77,
                'nircam-F480M' : 157.04,

                #MIRI
                'miri-F560W'  : 116.38,
                'miri-F770W'  :  64.83,
                'miri-F1000W' :  38.99,
                'miri-F1130W' :  30.42,
                'miri-F1280W' :  23.79,
                'miri-F1500W' :  17.28,
                'miri-F1800W' :  12.18,
                'miri-F2100W' :   9.11,
                'miri-F2550W' :   6.14,
    }
    return zero


def convertMagToJy(mag, zeroPointFlux):
    """Convert a magnitude to a Jansky

    Inputs
    ------
    mag
        (float or np.array) Value(s) to convery
    zeroPointFlux
        (float) Flux in Janskies corresponding to mag=0
        in that system
    """

    return zeroPointFlux * 10**(-mag/2.5)  


def convertJyToMag(flux, zeroPointFlux, flux_unc=None):
    mag = -2.5 * np.log10(flux/zeroPointFlux)

    if flux_unc is None:
        return mag 
    
    dmag = 2.5 / np.log(10) * (flux_unc / flux)
    return mag, dmag 

def jwstBackgroundLimitsLow_Jy() -> dict:
    """Background limits for JWST for regions of the sky with low background
    
    MIRI Limits taken from Glasse et al (2015) 127, 675, Table 3
    https://iopscience.iop.org/article/10.1086/682259
    """

    limits = {
        'miri-F560W'  : 0.16e-6,
        'miri-F770W'  : 0.25e-6,
        'miri-F1000W' : 0.54e-6,
        'miri-F1130W' : 1.35e-6,
        'miri-F1280W' : 0.84e-6,
        'miri-F1500W' : 1.39e-6,
        'miri-F1800W' : 3.46e-6,
        'miri-F2100W' : 7.09e-6,
        'miri-F2550W' : 26.2e-6,
    }
    return limits


def jwstBackgroundLimitsHigh_Jy() -> dict:
    """Background limits for JWST for regions of the sky with low background
    
    MIRI Limits taken from Glasse et al (2015) 127, 675, Table 3
    https://iopscience.iop.org/article/10.1086/682259
    """

    limits = {
        'miri-F560W'  :  0.16e-6,
        'miri-F770W'  :  0.25e-6,
        'miri-F1000W' :  0.56e-6,
        'miri-F1130W' :  1.43e-6,
        'miri-F1280W' :  1.00e-6,
        'miri-F1500W' :  1.94e-6,
        'miri-F1800W' :  4.87e-6,
        'miri-F2100W' :  9.15e-6,
        'miri-F2550W' : 30.80e-6,
    }
    return limits
