
import scipy.ndimage as spimg
import numpy as np

#Functions included as a convenience
from frmbase.plateau import plateau, convert_plateau_to_index
from frmbase.gapfill import fill_gaps

def median_detrend(y, filtersize):
    """Apply a median-detrend high pass filter to y
    
    Inputs
    ------
    y
        (numpy array) Array to be detrended
    filtersize
        (int) Length of filter in same units as y

    Returns
    ----------
    An array the same shape as y ,with long term trends removed, and short term signal preserved.
    """
    if filtersize %2 == 0:
        filtersize += 1

    filt = spimg.median_filter(y, filtersize, mode='reflect')
    return y - filt


def identify_single_point_outliers(y_orig):
    tol = 4
    y = np.diff(y_orig)
    rms = 1.4826 * mad(y)  #A robust estimate of mad, assuming Gaussian scatter
    roll = np.roll(y, 1)

    #Maybe a hypotenuse here?
    #y**2 + roll**2 > (tol*rms)**2
    idx = (np.fabs(y) > tol * rms) & (np.fabs(roll) > tol * rms) & (y * roll < 0) 
    idx = np.concatenate([idx, np.array([False])])
    return idx


def mad(y):
    """Compute median absolute deviation of a timeseries.
    
    MAD is a robust estimator of the noise. For purely Gaussian data,
    std(y) = 1.4826 * mad(y)
    
    """
    assert len(y) > 1
    return np.median(np.fabs(y))
