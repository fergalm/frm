from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import scipy.stats as spstats
import numpy as np

from frmbase.support import lmap 

#Mneumonic for typing
Arr = np.ndarray

"""
    Given a set of points, (x,y), compute to properties of
    the probability density ellipse from which they were drawn.

    Given a set of points (x,y), assume they were drawn from
    a probability density distribution well described by a
    2-dimensional Gaussian. The distribution may show co-variance,
    (i.e large values of x may make large values of y more or less
    likely, and vice-versa), so in general, the distribution will
    not be aligned with the x and y-axes.

    Additionally, some of the input points are outliers, so must
    perform some kind of sigma clipping to remove those values from
    the fit.

    The probability density ellipse is descibed by

    * A vector pointing from the origin to the centre of the ellipse.
    * A vector from the centre pointing along the semi-major axis, with
      length equal to the the :math:`\\sigma` of the Gaussian in
      that direction, :math:`\\sigma_a`.
    * A vector from the centre pointing along the semi-minor axis,
      with length equal to the :math:`\\sigma` of the Gaussian in
      that direction, :math:`\\sigma_b`.

    The algorith proceeds as follows.
    1. Input all data points (x,y)
    2. Compute the centroid, semi-major and semi-minor axes of the ellipse.
    3. Transform all the input points to a coordinate system where
       their first coordainte is distance along the semi-major axis in
       units of :math:`\\sigma_a`, and the second coordinate is the distance
       along the semi-minor axis in units of :math:`\\sigma_b`. This
       distrubtion is radiall symmetric around the origin.
    4. Compute the survival probability of each point, i.e the probability
       of finding a point at least that far from the origin in this
       transformed space.
    5. Reject points whose survival probability is less that some
       threshold. These points are the outliers
    6. Go back to step 2 and repeat until no more outliers found.
"""


def diagnostic_plot(x, y, flag=None):
    """Diagnostic plot for difference image centroids used in DAVE

    Preserved here for reference.
    """

    if flag is None:
        flag = np.zeros(len(x))
    idx = flag.astype(bool)

    # Turned off outlier detection because it doesn't work well.
    print(np.sum(idx))
    idx = find_outliers(x, y, initial_clip=idx, threshold=1e-6)
    print(np.sum(idx))
    # return 
    plotErrorEllipses(x, y, idx)

    plt.axhline(0)
    plt.axvline(0)
    plt.plot(0, 0, "*", color="c", ms=28, mec="w", mew=2)
    plt.xlabel("Column shift (pixels)")
    plt.ylabel("Row shift (pixels)")
    plt.axis("equal")
    plt.legend()

    offset, signif = compute_offset_and_signif(x[~idx], y[~idx])
    msg = "Offset %i pixels\nProb Transit on Target: %.0e" % (offset, signif)
    plt.title(msg)
    return plt.gcf()


def plotErrorEllipses(col:Arr, row:Arr, flag:Arr=None, **kwargs):
    """Plot 3 nested error ellipses that contain 68, 95, 9.7% of distrib.

    This is an example function to show to use the error ellipse
    calculations. You may want to write your own with
    more customisation.

    Inputs
    ----------
    col, row: 1d Float array
        Column and row values of each point. Array lengths 
        should be the same 
    flag: 1d bool array
        Ignore entries if `flag is True`

    Optional Inputs
    ------------------
    labels:
        Text to annotate each point with in the plot
    probs:
        Probability lines to draw. Default of [.68, .95, .997]
        will draw ellipses that will (on average) contain
        68%, 95% and 99.7% of the points (corresponding to 
        what we intuitively think of as 1,2 and 3 sigma.
        (This **not** true for 2d Gaussians, but are still
        handy reference numbers.
    outlier_colour (pink)
        What colour to draw outlier points ignored in the fit.
    ellipse_color (grey)
        What color to draw the error ellipses
    showMean:
        Show the uncertainty distribution on the mean, rather than
        the overall uncertainty distribution. Assumes Gaussian scatter.
        Default is **False**

        
    Notes
    -----------
    This function adds entries to the figure legend
    """

    defaultLabels = lmap(lambda x: f"{x}", range(len(col)))
    defaultProbs = [.68, .95, .997]
    labels = kwargs.get('labels', defaultLabels)
    probs = kwargs.get('probs', defaultProbs)
    outlierColour = kwargs.get('outlierColour', 'pink')
    ellipseColour = kwargs.get('ellipseColour', 'grey')
    userPointStyle = kwargs.get('pointStyle', {})
    showMean = kwargs.get('showMean', False)


    #Style for plotting individual points. Passed to
    #plt.plot. Can be updated by user passing in a dict called pointStyle
    pointStyle = {
        'color': 'k',
        'marker': 'o',
        'mec': 'w',
        'ls': "",
        'zorder': +5,
        'label': 'Centroids',  #For legend
    }

    pointStyle.update(userPointStyle)

    if isinstance(labels, str):
        labels = [labels] * len(col)

        
    #Plot points. Highlight flagged points in pink
    plt.plot(col, row, **pointStyle)
    pointStyle['color'] = outlierColour
    if np.any(flag):
        plt.plot(
            col[flag],
            row[flag],
            **pointStyle 
        )

    #Label points
    for i in range(len(col)):
        plt.text(col[i], row[i], labels[i], zorder=+5)

    ellipse = computeErrorEllipse(col, row, flag)

    if showMean:
        scale = np.sqrt(np.sum(~flag))
        ellipse['width'] /= scale
        ellipse['height'] /= scale

    _plot_ellipses(ellipse, probs, ellipseColour)
    # # Turned off outlier detection because it doesn't work well.
    # # idx = find_outliers(x, y, initial_clip=idx, threshold=1e-5)
    # mu_x = ellipse['centre_col']
    # mu_y = ellipse['centre_row']
    # width = ellipse['width']
    # height = ellipse['height']
    # angle_deg = ellipse['angle_deg_noe']

    # if showMean:
    #     scale = np.sqrt(np.sum(~flag))
    #     width /= scale
    #     hieght /= scale

    # ax = plt.gca()
    # for p in probs:
    #     scale = computeScaleForProb(p)
    #     ell = Ellipse(
    #         [mu_x, mu_y],
    #         width=width * scale,
    #         height=height * scale,
    #         angle=angle_deg,
    #         color=ellipseColour,
    #         alpha=0.2,
    #         label="%g%% Prob" % (100 * p),
    #     )
    #     ax.add_patch(ell)


def _plot_ellipses(ellipse, probs, ellipseColour):
    mu_x = ellipse['centre_col']
    mu_y = ellipse['centre_row']
    width = ellipse['width']
    height = ellipse['height']
    angle_deg = ellipse['angle_deg_noe']

    ax = plt.gca()
    for p in probs:
        scale = computeScaleForProb(p)
        ell = Ellipse(
            [mu_x, mu_y],
            width=width * scale,
            height=height * scale,
            angle=angle_deg,
            color=ellipseColour,
            alpha=0.2,
            label="%g%% Prob" % (100 * p),
        )
        ax.add_patch(ell)


def computeScaleForProb(prob):
    """Compute the size of ellipse that encompasses 100*prob of points
    
    `computeErrorEllipse` computes the properties of the ellipse
    that describes a 2d distribution of data. We like to draw
    that ellipse in such a way that it can be expected to include
    a certain fraction of the points from which it was computed.
    (for example 68% of the points, or 95% of the points)

    This is slightly non-intuitive for 2d Gaussians. For example
    the 1 sigma ellipse generated by computeErrorEllipse 
    contains, on average, about 40% of the points.

    This function handles the mathematics of computing
    the scaling needed to encompass the appropriate number 
    of points. 

     See `plotErrorEllipse` for an example of usage 

    Inputs
    -----------
    prob
        The fraction of points you wish to contain. E.g .68, or .95

    
    Returns
    -----------
    A float. Multiple the width and height of the ellipse from
    `computeErrorEllipse()` by this float to get the error ellipse
    you desire.
    """
    assert 0 < prob < 1 
    scale = spstats.rayleigh().isf(1 - prob)
    return scale 


def computeErrorEllipse(col: Arr,row: Arr, flag:Arr = None):
    """Compute the properties of the error ellipse around some 2d points

    Computes an error ellipse around a set of points in a 2d
    plane. The ellipse is represented by a width, a height,
    and a rotation of the semi-major axis in degrees north of 
    east. This is the format that matplotlib expects.

    The length of the semi-major axis of the resulting ellipse
    is equal to the standard deviation of the distribution
    along that axis. 
    
    2d Gaussian distributions are slightly non-intuitive if you
    are used to 1d ones. The compute the `width` and `height`
    of the elllipse that would encompass 95% of the underlying
    distribution, see `compute_scale_for_prob()`

    To compute the uncertainty on the mean position, scale 
    the width and the height by `1/sqrt(len(col))`
    
    Note that this error ellipse describes the distribution of 
    the data assuming the points are drawn from a Gaussian distribution.
    The validity of this assumption is up to the user to determine

    Inputs
    ----------
    col, row: 1d Float array
        Column and row values of each point. Array lengths 
        should be the same 
    flag: 1d bool array
        Ignore entries if `flag is True`


    Returns
    -----------
    A dictionary. The elements are 
        - centre_col
            - col value of centre of ellipse 
        - centre_row
            - row value of centre of ellipse 
        - width
            - Length of the major axis (not the semi-major)
        - height
            - Length of the minor axis
        - angle_deg_noe
            - Angle of rotation of semi-major axis north of east.
        - semimajor
            - Numpy array with two elements indicating 
              the vector describing the semi-major axis
              of the error ellipse 
        - semiminor 
            - Numpy array with two elements indicating 
              the vector describing the semi-minor axis
              of the error ellipse 
   """          
    if flag is None:
        flag = np.zeros(len(col))
    flag = flag.astype(bool)
        
    assert len(col) == len(row)
    assert len(col) == len(flag)

    mu_c = np.mean(col[~flag])
    mu_r = np.mean(row[~flag])
    sma, smi = compute_eigen_vectors(col[~flag], row[~flag])

    sigma_a = 2 * np.linalg.norm(sma)  #Full width, not semi-width
    sigma_b = 2 * np.linalg.norm(smi)
    angle_deg = compute_angle_noe(sma)

    out = {
        'centre_col': mu_c,
        'centre_row': mu_r,
        'width': sigma_a, 
        'height': sigma_b,
        'angle_deg_noe': angle_deg,
        'semimajor_vec': sma, 
        'semiminor_vec': smi,
    }
    return out

#TODO: Rename compute_angle_deg_noe
def compute_angle_noe(sma):
    angle_deg = np.degrees(np.arctan2(sma[1], sma[0]))
    return angle_deg

def compute_offset_and_signif(col, row):
    """Compute the mean offset of a set of points from the origin

    Computes the mean position of the inputs, and the statistical significance
    of the offset. The statistical signifance calculation assumes the
    points are drawn from a 2d Gaussian. See module docs above.

    Inputs
    ----------
    col, row
        (1d np arrays) Column and row values for each obseration.

    Returns
    ----------
    A tuple of (offset, signif)
    The offset is the offset of the mean value for column and row from
    the origin, and is measured in the same units as the inputs.
    The significance
    is measured as the probability of seeing an offset at least this large
    in this direction, given the variance (and co-variances) of the
    column and row values. Values close to 1 indicate the offset is consistent
    with zero. Low values (< 1e-3 or so) indicate the offset is
    statistically significant
    """

    centroid = get_centroid_point(col, row)
    sma, smi = compute_eigen_vectors(col, row)

    offset_pixels = np.linalg.norm(centroid)
    prob = compute_prob_of_points([0], [0], sma, smi, centroid)
    return offset_pixels, prob


def findOutliers(
    col:Arr, 
    row:Arr, 
    threshold=1e-6, 
    initial_clip:Arr=None, 
    max_iter=10
):
    """Find data points that are not outliers

    Compute the error ellipse for a set of points. 
    Remove the largest outlier, and recompute the error ellipse.
    If the removed point is now highly unlikely, 
    `P(point|ellipse) < threshold`, mark it as an outlier.
    Repeat until you find a point that is unsurprising
    given the error ellipse of the remaining points.

    Inputs
    ----------
    col, row : 1d np arrays of floats
        Input points
    threshold: float
        A point is marked as an outlier is the probability of seeing
        a point is less likely than threshold given the distribution
        of the remaining good points. Increase this value logaritmically
        to get more agressive outlier detection
    initial_clip: 1d boolean np.ndarray
        If any values in this array are **True** assume those points
        are bad before searching for outliers. For example if 
        `initial_clip[3] is True`, ignore the point `col[3], row[3]`
    max_iter
        Raise an exception if more than this many points are removed.

    Returns
    --------------
    A 1d np array of booleans. True values indicate those points
    are outliers. True values from `initial_clip` are propegated
    into the output (i.e if a point is considered bad before
    outlier identification, it is included as a bad point in the output)
    """

    # idx is true if a point is bad
    idx = initial_clip
    if idx is None:
        idx = np.zeros(len(row), dtype=bool)

    assert len(col) == len(row)
    assert len(idx) == len(row)

    #Compute probability of points
    sma, smi = compute_eigen_vectors(col[~idx], row[~idx])
    prob = compute_prob_of_points(col, row, sma, smi)
    print(prob)
    # import ipdb; ipdb.set_trace()

    for i in range(int(max_iter)):
        if np.all(idx):
            raise RuntimeError("All points were removed during clipping")
        
        #Identify least likely point
        prob[idx] = 1  #Mask out previous bad values
        i0 = np.argmin(prob)
        idx[i0] = True 

        #Compute new probabilities
        sma, smi = compute_eigen_vectors(col[~idx], row[~idx])
        prob = compute_prob_of_points(col, row, sma, smi)
        if prob[i0] > threshold:
            #Removing that point doesn't change distriution
            #much. Let's leave it in and quit.
            idx[i0] = False 
            return idx 
        
    #If we get to here, we've failed to converge
    raise RuntimeError("Max iterations exceeded")


def compute_eigen_vectors(x, y):
    """Compute semi-major and semi-minor axes of the probability
    density ellipse.

    Proceeds by computing the eigen-vectors of the covariance matrix
    of x and y.

    Inputs
    -----------
    x, y
        (1d numpy arrays). x and y coordinates of points to fit.


    Returns
    ---------
    sma_vec
        (2 elt numpy array) Vector describing the semi-major axis
    smi_vec
        (2 elt numpy array) Vector describing the semi-minor axis
    """

    assert len(x) == len(y)
    cov = np.cov(x, y)
    assert np.all(np.isfinite(cov))
    eigenVals, eigenVecs = np.linalg.eig(cov)

    # idebug()
    sma_vec = eigenVecs[:, 0] * np.sqrt(eigenVals[0])
    smi_vec = eigenVecs[:, 1] * np.sqrt(eigenVals[1])
    return sma_vec, smi_vec


def compute_prob_of_points(x, y, sma_vec, smi_vec, cent_vec=None):
    """Compute the probability of observing points as far away as (x,y) for
    a given ellipse.

    For the ellipse described by centroid, semi-major and semi-minor axes
    `sma_vec` and `smi_vec`, compute the
    probability of observing points at least as far away as x,y in
    the direction of that point.

    If no cent_vec supplied, it is computed as the centroid of the
    input points.

    Inputs
    ---------
    x, y
        (1d numpy arrays). x and y coordinates of points to fit.
    sma_vec
        (2 elt numpy array) Vector describing the semi-major axis
    smi_vec
        (2 elt numpy array) Vector describing the semi-minor axis
    cent_vec
        (2 elt numpy array) Vector describing centroid of ellipse.
        If **None**, is set to the centroid of the input points.

    Returns
    --------
    1d numpy array of the probabilities for each point.

    Note
    --------
    Although I independently dervied this, it is quite similar to
    the concept of the Mahalanobis distance
    """
    if cent_vec is None:
        cent_vec = get_centroid_point(x, y)

    assert len(x) == len(y)
    assert len(cent_vec) == 2
    assert len(sma_vec) == 2
    assert len(smi_vec) == 2

    xy = np.vstack([x, y]).transpose()
    rel_vec = xy - cent_vec

    # The covector of a vector **v** is defined here as a vector that
    # is parallel to **v**, but has length :math:`= 1/|v|`
    # Multiplying a vector by the covector of the semi-major axis
    # gives the projected distance of that vector along that axis.
    sma_covec = sma_vec / np.linalg.norm(sma_vec) ** 2
    smi_covec = smi_vec / np.linalg.norm(smi_vec) ** 2
    coeff1 = np.dot(rel_vec, sma_covec)  # Num sigma along major axis
    coeff2 = np.dot(rel_vec, smi_covec)  # Num sigma along minor axis

    dist_sigma = np.hypot(coeff1, coeff2)
    prob = spstats.rayleigh().sf(dist_sigma)
    return prob


def get_centroid_point(x, y):
    return np.array([np.mean(x), np.mean(y)])
