
from  ipdb import set_trace as idebug 
import scipy.interpolate as spinterp
import frm.plateau as plateau
import numpy as np



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
    """
    y = y.copy()

    if np.isfinite(bad_value):
        idx = y == bad_value
    else:
        idx = np.isnan(y)

    gaps = plateau.plateau(idx.astype(float), .5)
    y = fill_small_gaps(y, gaps, small_size)
    y = fill_large_gaps(y, gaps, small_size)

    idx = plateau.convert_plateau_to_index(gaps, len(y))
    return y, idx

def fill_small_gaps(y, gaps, max_size):
    gap_size = gaps[:,1] - gaps[:,0]
    gaps = gaps[gap_size <= max_size]

    for g in gaps:
        left, right = g[0], g[1]
        snippet = y[left-max_size: right + max_size+1]
        fill_value = np.nanmean(snippet)
        y[left:right] = fill_value 
    return y


def fill_large_gaps(y, gaps, min_size):
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

#I ended up liking this less
    # for g in gaps:
    #     pad = int( (g[1] - g[0])/ 2.)
    #     left = np.max([g[0] - pad, 0])
    #     right = np.min([g[1] + pad, len(y)])
    #     slice_l = slice(left, g[0])
    #     slice_r = slice(g[1], right)

    #     xx = np.concatenate( [t[slice_l], t[slice_r]])
    #     yy = np.concatenate( [y[slice_l], y[slice_r]])
    #     #Superior to Pchip and Barycentric for the Kepler TPS case
    #     #were data after the gap has the same slope as data before the 
    #     #gap, but is offset in the opposite direction.
    #     # spline = spinterp.CubicSpline(xx, yy)
    #     spline = spinterp.PchipInterpolator(xx, yy)

    #     range_to_interp = np.arange(g[0], g[1])
    #     y[range_to_interp] = spline(range_to_interp)


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
