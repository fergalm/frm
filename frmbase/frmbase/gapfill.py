
from  ipdb import set_trace as idebug 
import scipy.interpolate as spinterp
import numpy as np

from frmbase.fitter.lsf import Lsf
import frmbase.plateau as plateau



def fill_gaps(y, small_size=5, bad_value=np.nan):
    """Fill gaps in a 1d array
    
    Gap filling is a useful step if you intend to smooth data. By filling in gaps
    with "reasonable" values you can mitigate edge effects around the gaps.
    Be careful to remove the gap filled data after smoothing to avoid analysing fake data!

    Assumes that any dc-offset has been removed from the signal (e.g by mean-zeroing
    or median-zeroing the data)

    Based on an idea I first saw in the TPS unit of Kepler SOC Pipeline, although 
    the algorithm is of an independent design  See Section 9.2.3.2 of
    https://archive.stsci.edu/kepler/manuals/KSCI-19081-001_Data_Processing_Handbook.pdf

    This implementation is not concerned with preserving noise properties in the gap.

    Inputs
    ---------
    y
        (1d np array) Data to be gap filled. The data is assumed to be sequential
        in time, with gaps indicated by Nans. The characteristic value of the data
        (i.e the mean or median) is assumed to be close to zero. If not, edge effects
        will be evident near the start and end of the data.
    
    Optional Inputs
    --------------------
    bad_value
        (float) Use this to set the value for elements of y that are in gaps. Default
        is Nan, but you can set to any floating point value.
    small_size
        (int) Controls the threshold between small and large gaps (see below)
    
    The algorithm proceeds in 3 steps.
    1. Gaps are identifed as sequences in y where the values are equal to `bad_value`
    2. Small gaps (less than small_size) are replaced by the average of nearby points.
    3. Large gaps are infilled with a spline interpolation across the gaps. 


    Returns
    ----------
    y
        (1d np array) Original array with gaps filled 
    idx
        (1d np array bool) True if that element is interpolated
    """
    y = y.copy()

    if np.isfinite(bad_value):
        idx = y == bad_value
    else:
        idx = np.isnan(y)

    gaps = plateau.plateau(idx.astype(float), .5)
    y = fill_small_gaps(y, gaps, small_size)
    y = fill_large_gaps_cubic(y, gaps, small_size)

    idx = plateau.convert_plateau_to_index(gaps, len(y))
    return y, idx

def fill_small_gaps(y, gaps, max_size):
    if len(gaps) == 0:  #No gaps to fill
        return y 

    gap_size = gaps[:,1] - gaps[:,0]
    gaps = gaps[gap_size <= max_size]

    for g in gaps:
        left, right = g[0], g[1]
        lwr = max(left - max_size, 0)
        upr = min(right + max_size + 1, len(y))
        snippet = y[lwr:upr]
        assert len(snippet) > 0
        fill_value = np.nanmean(snippet)
        y[left:right] = fill_value 
    return y


def fill_large_gaps(y, gaps, min_size):
    """Deprecated. Use fill_large_gaps_cubic instead"""
    if len(gaps) == 0:  #No gaps to fill
        return y 

    gap_size = gaps[:,1] - gaps[:,0]
    gaps = gaps[gap_size > min_size]

    t = np.arange(len(y))

    #This works better. In fill with old data
    #TODO in fill with old data from left and right (not just left)
    #then make them match up in the middle
    for g in gaps:
        pad_size = int( (g[1] - g[0]))
        left = np.max([g[0] - pad_size, 0])
        # right = np.min([g[1] + pad, len(y)])
        # slice_l = slice(left, g[0])
        range_to_interp = np.arange(g[0], g[1])
        
        fill_data = y[left:g[0]]
        if len(fill_data) < pad_size:
            #resize automatically fills the new array with copies of old data
            fill_data = np.resize(fill_data, pad_size)

        y[range_to_interp] = fill_data

    return y


def fill_large_gaps_cubic(y, gaps, min_size):
    if len(gaps) == 0:
        return y
        
    gap_size = gaps[:,1] - gaps[:,0]
    gaps = gaps[gap_size > min_size]
    y_out = y.copy()
    for g in gaps:
        pad_size = int( (g[1] - g[0]))
        y_tmp = fill_single_large_gap_cubic(y, g[0], g[1], pad_size)
        y_out[ g[0]:g[1] ] = y_tmp #fill the gap

        #TODO Add noise?
    return y_out


def fill_single_large_gap_cubic(y, y1, y2, pad_size):
    """Fill a large gap in a timeseries using cubic interpolation.
    
    Fit a cubic polynomial to a sequence before a gap, and again
    to a sequency after the gap. The fit a cubic polynomial over the gap
    that has the same value and slope as the anchoring polynomials on either side

    Works, but not well tempered in production.

    Inputs
    --------
    y
        (1d np array) Data set in which to fill the gap.
    y1, y2 
        (ints) indices of start and end of gap 
    pad_size
        (int) Number of indices of data before and after to fit to. 


    Returns
    -----------
    A numpy array of length (y2-y1) representing the interpolated polynomial.
    You may want to add some noise to the interpolation before applying smoothing

    Explanation
    ------------
    A polynomial is of the form :math:`y = ax^3 + bx^2 + cx + d`. The slope 
    of the polynomial is is :math:`y' = 3ax^3 + 2bx + c`.

    At x=0, (y, y') has the same values as the polynomial fit to the "before" data.
    At x=n, they have the same values as the polynomial fit to the "after" data. 
    Working through the simultaneous equations gives us the values of a, b, c and d
    for the interpolating polynomial.

    """

    #Order of indices: 0..left..start (gap) end ..right...
    y0 = np.max([y1 - pad_size, 0])
    y3 = np.min([y2 + pad_size, len(y)])

    if y1 > y0:
        left_anchor = y[y0:y1]
    else:
        left_anchor = [y[y0]]

    if y3 > y2:
        right_anchor = y[y2:y3]
    else:
        right_anchor = y[y3-1]

    # assert np.all(np.isfinite(left_anchor)), "Nan found before gap!"
    # assert np.all(np.isfinite(right_anchor)), "Nan found after gap!"
    left_anchor = left_anchor[ np.isfinite(left_anchor)]
    right_anchor = right_anchor[ np.isfinite(right_anchor)]
    #TODO check for min length

    ys, ms = get_params_of_anchor_section(left_anchor, left=True, dx=y0)
    ye, me = get_params_of_anchor_section(right_anchor, left=False, dx=y2)

    c = ms
    d = ys 
    dy = ye - ys 
    dm = me - ms 
    size = y2 - y1

    Amat = np.array([size**3, size**2, 3*size**2, 2*size]).reshape((2,2))
    bVec = np.array([dy - ms*size, dm]).reshape((2,1))
    res = np.linalg.solve(Amat, bVec)
    a, b = res[0], res[1]

    x = np.arange(size)
    y = (((a * x) + b) * x + c) * x + d
    assert np.all(np.isfinite(y))
    return y 

    
def get_params_of_anchor_section(y, left=True, dx=0):
    #TODO Robust fit?
    x = np.arange(len(y))
    fobj = Lsf(x, y, 1, 4)
    pars = fobj.getParams()

    d, c, b, a = pars

    if left:
        x0 = x[-1]
    else:
        x0 = x[0]

    offset = fobj.getBestFitModel(x0)[0]
    slope = ((3*a * x0) + 2*b)*x0 + c
    return offset, slope



import matplotlib.pyplot as plt 
def test_single_large_gap():
    x = np.arange(1000)    
    y = np.sin(2*np.pi*x/250)
    
    t1, t2 = 200, 340
    y[t1:t2] = np.nan
    y[t2:] -= 2

    plt.clf()
    plt.plot(x, y, 'r.-')

    gaps = np.array( [ [t1, t2]])
    y0 = y.copy()
    y_gap = fill_large_gaps_cubic(y, gaps, 3)
    y[ gaps[0,0]:gaps[0,1]] = y_gap
    
    plt.plot(x[t1:t2], y_gap, 'b.-')

# def test_fill_large_gaps1():
#     x = np.arange(1000)    
#     y = np.sin(2*np.pi*x/250)
#     y[200:300] = np.nan

#     plt.clf()
#     plt.plot(x, y, 'r-')

#     gaps = np.array( [ [200, 300]])
#     y0 = y.copy()
#     y = fill_large_gaps(y, gaps, 3)

#     plt.plot(x, y, 'b--')


# def test_fill_large_gaps2():
#     x = np.arange(1000)    
#     y = np.sin(2*np.pi*x/250)
#     y[500:] = np.sin( - 2* np.pi*x[500:]/250 + 125 ) - 1.5


#     y[475:575] = np.nan
#     plt.clf()
#     plt.plot(x, y, 'r-')

#     gaps = np.array( [ [475, 575]])
#     y0 = y.copy()
#     y = fill_large_gaps(y, gaps, 3)

#     plt.plot(x, y, 'b--')
