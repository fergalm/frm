import numpy as np

from . import AbstractFitter
from .AbstractFitter import AbstractFitter
from . import lsf


"""AbstractFitter should accept a mapping function to map x-values to a new range

eg
def nullMapper(x):
    return x

def unityMapper(x, fullRange=None):
    if fullRange is None:
        fullRange = [np.min(x), np.max(x)]

    return (x - fullRange[0]) / (fullRange[1] - fullRange[0])

Every function that takes an x value as input applies the mapper to it
"""


def biweight(x, y, modY):
    """A robustfit weighting function.
    Based on http://reference.wolfram.com/applications/eda/RobustFitting.html

    Inputs:
    -----------
    x
        (1d numpy array) X coords of data to be fit
    y
        (1d numpy array) Y coords of data to be fit
    modY
        (1d numpy array) best fit model at each value of x

    Returns
    --------------
    1d array of weights for each point. Outlier points
    are weighted zero
    """
    residual = y-modY
    mad = np.median( np.fabs(residual) )

    #In the corner case of perfect correlation, return equal weights
    if mad == 0:
        return residual*0 + 1

    cutoff = 6*mad
    weight = (residual/cutoff)
    weight = (1 - weight**2)**2

    #Completely deweight points over threshold
    idx = np.fabs(residual) > cutoff
    weight[idx] = 0

    return weight


class RobustFit(AbstractFitter):
    """
    Robust fit a model to data in a least squares sense

    This class is modeled on robustfit() in Matlab, but allows any
    function to be fit, not just a straight line. It iteratively
    deweights or trims points that are wildly inconsistent with the best
    fit model, until it converges on a stable solution.

    The exact behaviour depends on the choice of weighting function used,
    and the default choice here (biweight) may be a litle different than
    whats used by Matlab
    """
    def __init__(self, x, y, order,func=lsf.poly, weightFunc=biweight, **kwargs):
        """
        Input
        ----------
        x
            (1d numpy array) X coords of data to be fit
        y
            (1d numpy array) Y coords of data to be fit
        order
            (int) How many orders of func to fit
        func
            (function object) The analytic function to fit
        weightFunc
            (function object) What function to use to weight and
            cull points based on their fit residuals.
        kwargs      Addtional arguments to pass to func


        Note:
        ---------
        * Only unweighted robust fit can be performed, because robust fitting does its own weighting
        * See the function ``getUncertainties()`` for details on how the input
          ``s`` is used.

        """
        AbstractFitter.__init__(self, x, y, None, order, func=func, **kwargs)
        self.weightFunc = weightFunc

        self.check_inputs()
        self.fit()

    def fit(self):
        """Do the fit.

        This method is unusual because it creates an Lsf object to actually
        do the fitting.
        """

        maxIter = 99
        tol = 1e-3
        numIter = 0

        #First step is to do an unweighted LSF fit to find the outliers
        fObj = lsf.Lsf(self.x, self.y, None, self.order, func=self.func, **self.kwargs)
        newChi = fObj.getChiSq()
        oldChi = newChi + 2*tol
        while  newChi<oldChi and \
               np.fabs(oldChi - newChi) > tol and \
               numIter < maxIter:

            modY = fObj.getBestFitModel()
            w = self.weightFunc(self.x, self.y, modY)

            #Convert w to 1/sqrt(w), which is what Lsf expects
            #If weght is zero, make 1/sqrt(w) large
            idx  = w == 0
            w[idx] = 1  #Avoid division by zero error
            w = 1/np.sqrt(w)
            w[idx] = 1e99

            fObj = lsf.Lsf(self.x, self.y, w, self.order, func=self.func, **self.kwargs)
            oldChi = newChi
            newChi = fObj.getChiSq()
            numIter += 1

        if numIter >= maxIter:
            raise ValueError("Max iterations exceeded")

        self.weight = w
        self.param = fObj.getParams()
        self.cov = fObj.getCovariance()


    def getBestFitModel(self, x=None):
        """Get best fit model to data

        Inputs:
        x   (1d numpy array) Ordinates on which to compute
            best fit model. If not specified use ordinates
            used in fit

        Return
        1d numpy array
        """
        self.check_fit_complete()

        if x is None:
            x = self.x

        try:
            len(x)
        except TypeError:
            x = np.asarray([x])

        par = self.param
        bestFit = np.zeros( (len(x)))
        for i in range(self.order):
            bestFit += par[i] * self.func(x, i, **self.kwargs)
        return bestFit


    def getWeights(self):
        return self.weight


    def getIndicesOfOutliers(self):
        """Returns an array of length len(x) where elements are True
        if that point was markedd as an outlier"""
        return self.weight > 9e98


    def getUncertainty(self):
        """
        """

        return NotImplementedError("Robust fit is unweighted. Use Lsf() to get uncertainties")
