
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


def sigma_clip(y, nSigma, maxIter=1e4, initialClip=None):
    """Iteratively find and remove outliers
    Find outliers by identifiny all points more than **nSigma** from
    the mean value. The recalculate the mean and std and repeat until
    no more outliers found.
    Inputs:
    ----------
    y
        (1d numpy array) Array to be cleaned
    nSigma
        (float) Threshold to cut at. 5 is typically a good value for
        most arrays found in practice.
    Optional Inputs:
    -------------------
    maxIter
        (int) Maximum number of iterations
    initialClip
        (1d boolean array) If an element of initialClip is set to True,
        that value is treated as a bad value in the first iteration, and
        not included in the computation of the mean and std.
    Returns:
    ------------
    1d numpy array. Where set to True, the corresponding element of y
    is an outlier.
    """
    #import matplotlib.pyplot as mp
    idx = initialClip
    if initialClip is None:
        idx = np.zeros( len(y), dtype=bool)

    assert(len(idx) == len(y))

    #x = np.arange(len(y))
    #mp.plot(x, y, 'k.')

    oldNumClipped = np.sum(idx)
    for i in range(int(maxIter)):
        mean = np.nanmean(y[~idx])
        std = np.nanstd(y[~idx])

        newIdx = np.fabs(y-mean) > nSigma*std
        newIdx = np.logical_or(idx, newIdx)
        newNumClipped = np.sum(newIdx)

        #print "Iter %i: %i (%i) clipped points " \
            #%(i, newNumClipped, oldNumClipped)

        if newNumClipped == oldNumClipped:
            return newIdx

        oldNumClipped = newNumClipped
        idx = newIdx
        i+=1
    return idx
