
import matplotlib.pyplot as mp
import numpy as np


#Generating functions for Legendre/Chebyshev basis vectors
#$Id: legendre.py 1327 2013-06-10 15:15:53Z fmullall $
#$URL: svn+ssh://flux.amn.nasa.gov/home/fmullall/svn/kepler/py/legendre.py $
#Fergal Mullally
#
#Given an array of x values, calculate appropriate values
#for a set of orthogonal basis vectors on those x values.
#These vectors can then be passed into Lsf and be fit to data.
#
#After generating both of these, I've decided that Chebyshevs
#are better for fitting arbitrary data. Legendres are preferred
#only for situations where there is a physical reason to believe
#they are suitable.

def generateLegendre(x, maxN, limits=None):
    """Generate legengre polynomials on the range spanned by x

    Input:
    x       (1d array) Values of x to compute polynomials for
    maxN    (int) Highest order polynomial to compute
    limits   (2 element list) Map this limits onto [-1,1] (See below)
            Default is min(x) to max(x)

    Returns:
    2d numpy array of size (len(x), maxN)

    Discussion
    Legendre functions are defined so that they are orthogonal
    on the interval [-1, 1]. This makes them suitable polynomials
    for fitting data.

    The problem is that our data is rarely defined on the interval
    [-1, 1]. So what we do is convert x to the interval [-1,1],
    using theta = 2*(x-x0)/(x1-x0) - 1, where x0 and x1 can be defined
    using limits. If they are not supplied, the max and min values
    of x are used.

    Legendre polynomials can be quickly computed using the
    generating function
    P_i = 1/i [ (2i-1)*x*P_(i-1) - (i-1)*P_(i-2) ]
    where x is in the interval [-1, 1]
    See \S 7.3 (p130) of "Data Reduction and Error Analysis for the
    Physical Sciences" by Bevington and Robinson.

    Notes:
    A minimum of 2 polynomials are returned, regardless of the
    value of maxN

    """

    if maxN < 2:
        maxN = 2

    if limits is None:
        min = np.min(x)
        max = np.max(x)
        limits = [min, max]

    out = np.empty((len(x), maxN))

    #Convert x to the interval -1, +1
    theta = (x-limits[0]) / float(limits[1] - limits[0])
    #cost = np.cos(2*np.pi*theta)
    cost = theta*2 -1
    #import pdb; pdb.set_trace()
    out[:,0] = 1
    out[:,1] = cost

    for i in range(2, maxN):
        a = (2*i -1 ) * cost * out[:,i-1]
        b = (i-1) * out[:,i-2]
        out[:,i] = a - b
        out[:,i] /= float(i)

    return out



def legendre(x, order, **kwargs):
    """Fitting function to be passed to Lsf to fit
        legendre polynomials to data

       Inputs:
       x        (1d numpy array) index into array
       order    (int) Which legendre polynomial to return
       kwargs   (dict) Optional extra arguments. For legendre
                this is one required argument, basisVectors.
                This is a 2d numpy array of shape (len(x), max(order))

        Returns:
        The values of the order-th legendre polynomial

        Notes:
        This fitting function is a little different to the
        standard fitting functions in lsf.
        To fit a set of data points (x, y, [s]) with a set of n legendre
        polynomials, first generate the polys with
        leg = generateLegendre(x, n)

        Then fit (x, y, [s]) using
        Lsf(x, y, s, n, func=legendre, basisVectors=leg)

        generateLegendres computes the value of the legendre
        polynomial appropriate for each value of x given the
        interval that spans, and this function merely returns
        that value
    """

    #Note: Is this code is generic enough to use for any
    #pregenerated 1d basis vectors?
    try:
        bv = kwargs['basisVectors']
    except KeyError:
        raise KeyError("Key basisVectors not defined")


    if order > bv.shape[1]:
        raise ValueError("Order %i requested, but only %i basis vectors defined" %(order, bv.shape[1]))

    return bv[:,order]



def generateChebyshev(x, maxN, limits=None):
    """Generate Chebysehv polynomials (of the first kind) on the
    interval spanned by x

    Input:
    x       (1d array) Values of x to compute polynomials for
    maxN    (int) Highest order polynomial to compute
    limits   (2 element list) Map this limits onto [-1,1] (See below)
            Default is min(x) to max(x)

    Returns:
    2d numpy array of size (len(x), maxN)

    Discussion
    Chebyshev functions are defined so that they are orthogonal
    on the interval [-1, 1]. This makes them suitable polynomials
    for fitting data.

    The problem is that our data is rarely defined on the interval
    [-1, 1]. So what we do is convert x to the interval [-1,1],
    using theta = 2*(x-x0)/(x1-x0) - 1, where x0 and x1 can be defined
    using limits. If they are not supplied, the max and min values
    of x are used.

    Legendre polynomials can be quickly computed using the
    generating function
    T_i = [ 2*x*T_(i-1) - *T_(i-2) ]
    where x is in the interval [-1, 1]
    See en.wikipedia.org/wiki/Chebyshev_polynomials

    The chebyshev polynomials are very similiar to the
    Legendre polynomials

    Notes:
    A minimum of 2 polynomials are returned, regardless of the
    value of maxN

    """

    if maxN < 2:
        maxN = 2

    if limits is None:
        min = np.min(x)
        max = np.max(x)
        limits = [min, max]

    out = np.empty((len(x), maxN))

    #Convert x to the interval -1, +1
    y = (x-limits[0]) / float(limits[1] - limits[0])
    #y = np.cos(2*np.pi*y)
    y = y*2 -1
    #import pdb; pdb.set_trace()
    out[:,0] = 1
    out[:,1] = y

    for i in range(2, maxN):
        a = 2*y * out[:,i-1]
        b = out[:,i-2]
        out[:,i] = a - b

    return out



def chebyshev(index, order, **kwargs):
    """Fitting function to be passed to Lsf to fit
        Chebyshev polynomials (of the first order) to data

       Inputs:
       index        (1d numpy array) index into array
       order    (int) Which legendre polynomial to return
       kwargs   (dict) Optional extra arguments. For legendre
                this is one required argument, basisVectors.
                This is a 2d numpy array of shape (len(x), max(order))

        Returns:
        The values of the order-th legendre Chebyshev

        Notes:
        This fitting function is a little different to the
        standard fitting functions in lsf.
        To fit a set of data points (x, y, [s]) with a set of n Chebyshev
        polynomials, first generate the polys with
        leg = generateLegendre(x, n)

        Then fit (x, y, [s]) using
        Lsf(x, y, s, n, func=legendre, basisVectors=leg)

        generateChebyshev computes the value of the legendre
        polynomial appropriate for each value of x given the
        interval that spans, and this function merely returns
        that value
    """

    try:
        bv = kwargs['basisVectors']
    except KeyError:
        raise KeyError("Key basisVectors not defined")


    if order > bv.shape[1]:
        raise ValueError("Order %i requested, but only %i basis vectors defined" %(order, bv.shape[1]))

    return bv[index,order]






import lsf
def main():
    x = np.arange(-20, 20, .125)

    y = x**2 - 8*x + 4
    y += 15*np.sin(2*np.pi*x/5.)

    nOrder = 32
    leg = generateLegendre(x, nOrder)
    cheby = generateChebyshev(x, nOrder)

    fitL = lsf.Lsf(x, y, None, nOrder, func=legendre, basisVectors=leg)
    print( fitL.getParams())
    fitC = lsf.Lsf(x, y, None, nOrder, func=chebyshev, basisVectors=cheby)
    print( fitC.getParams())
    fitP = lsf.Lsf(x, y, None, nOrder, func=lsf.poly)
    print (fitP.getParams())

    mp.clf()
    mp.plot(x, y, 'b.')
    mp.plot(x, fitL.getBestFitModel(), 'r-')
    mp.plot(x, fitC.getBestFitModel(), 'g--')
    #mp.plot(x, fitP.getBestFitModel(), 'm-')

    mp.show()


    #numOrder = 8
    #leg = generateLegendre(x, numOrder)
    #cheby = generateChebyshev(x, numOrder)

    #mp.clf()
    #colour = 'rgbcmk'
    #for i in range(6, numOrder):
        #c = np.fmod(i, 6)
        #mp.plot(x, leg[:,i], color=colour[c], ls='-')
        #mp.plot(x, cheby[:,i], color=colour[c], ls='--')

    #mp.show()
    #return leg
