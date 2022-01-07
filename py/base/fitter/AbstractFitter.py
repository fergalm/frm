"""
Created on Fri Dec  2 09:52:13 2016

@author: fergal
"""

import numpy as np


class AbstractFitter(object):
    def __init__(self, x, y, s, order, func, **kwargs):
        self.x  = x
        self.y = y
        self.s = s
        self.order = order
        self.func = func
        self.kwargs = kwargs

    def to_dict(self):
        dic = self.__dict__ 
        # f = dic['func']
        # module = str(f.__module__) 
        # name = str(f.__name__) 
        # dic['func'] = "%s.%s" %(module, name)
        return dic

    @classmethod 
    def from_dict(cls, dic):
        obj = cls.__new__(cls)
        obj.__dict__ = dic 
        return obj


    def check_inputs(self):
        """Check inputs for a 1d fitter (fitting x against y)"""
        try:
            x = self.x
            y = self.y
            s = self.s
        except AttributeError as e:
            raise AttributeError("One of order, x, y and s not defined: %s" %(e))


        if len(x) == 0:
            raise ValueError("x is zero length")

        if len(x) != len(y):
            raise IndexError("x and y must have same length")

        size = len(x)
        if s is None:
            self.s = np.ones(len(x))
        elif np.isscalar(s):
            self.s = np.ones((size)) * s
        else:
            if len(x) != len(s):
                raise IndexError("x and s must have same length")
            self.s = s

        if not np.all(np.isfinite(self.x)):
            raise ValueError("Nan or Inf found in x")

        if not np.all(np.isfinite(self.y)):
            raise ValueError("Nan or Inf found in y")

        if not np.all(np.isfinite(self.s)):
            raise ValueError("Nan or Inf found in s")



    #def fit(self, x, y, s, order, func, **kwargs):
    def fit(self):
        """Fit the function to the data and return the best fit
        parameters"""
        raise NotImplementedError("Dont' call abstract class directly")


    def getBestFitModel(self, x=None):
        """Get best fit model to data"""
        raise NotImplementedError("Dont' call abstract class directly")


    def check_fit_complete(self):
        try:
            self.param
            self.cov
        except AttributeError as e:
            raise AttributeError("Fit hasn't been performed yet: %s" %(e))


    def getCovariance(self):
        """Get the covariance matrix

           Return
           -----------
           np.matrix object
        """
        self.check_fit_complete()
        return self.cov


    def getChiSq(self):
        """Get chi square for fit

            Returns
            --------------
            double
        """
        chi = self.getResiduals() / self.s
        chisq = chi**2
        return np.sum(chisq)


    def getParams(self):
        self.check_fit_complete()
        return self.param


    def getReducedChiSq(self):
        """Get reduced chi-squared."""
        num = len(self.y) - self.order
        return self.getChiSq()/float(num)


    def getResiduals(self):
        """Get residuals, defined as y - bestFit

        Returns
        -------------
        1d numpy array
        """
        return self.y - self.getBestFitModel()


    def getVariance(self):
        """Variance is the sum of the squares of the residuals.
        If your points are gaussian distributed, the standard deviation
        of the gaussian is sqrt(var).
        """

        resid = self.getResiduals()

        val = np.sum( resid**2)
        val /= len(self.y) - self.order
        return val


    def getUncertainty(self):
        """Get the uncertainty on the best fit params

        Return
        ----------------
        1d numpy array
        """
        self.check_fit_complete()

        unc = np.sqrt(self.cov.diagonal())
        return np.array(unc)[0]


    def getUncertaintyFromVariance(self):
        """Use variance of residuals to compute uncertainty

        Sometimes you have data with no easily accessible
        uncertainties. However, if you can assume
        a) The model is appropriate, and your residuals are pure noise
        b) each point has approx the same uncertainty

        you can use this function to compute the uncertainty
        by using the scatter in the residuals to estimate the
        average per data-point uncertainty.

        Inputs
        ------------
        (none)

        Returns
        ----------------
        Vector of uncertainties


        See also
        ------------
        getScatterFromVariance(): Estimate the true scatter in
        the input data from the variance.
        """

        var = self.getVariance()
        return np.sqrt(var) * self.getUncertainty()


    def getScatterFromVariance(self):
        """Estimate the true scatter in the input data based on residuals

        Estimates the correct weights (the **s** input argument)
        to apply to the data to give a fit that results in a
        reduced chi squared of 1.

        Assumes the weights homoskedastic (i.e all points have the
        same intrinsic scatter, i.e that an unweighted fit is appropriate)

        The correctness of this function for weighted fits in unexplored

        Inputs
        ------------
        (none)

        Returns
        ----------------
        float


        See also
        ------------
        getUncertaintyFromVariance(): Estimate the true uncertainty in
        the best fit parameters from the variance.

        """
        return np.sqrt(self.getVariance())
