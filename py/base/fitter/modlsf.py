"""
Created on Fri Dec  2 10:00:59 2016

@author: fergal
"""

#import matplotlib.pyplot as plt
#import matplotlib as mpl
#import pandas as pd
import numpy as np

from AbstractFitter import AbstractFitter






class ScaledModLsf(AbstractFitter):
    """Fit a model to data

    An offset and scale term are applied. The function fit is
    ``a0 + a1*model``

    Note
    ---------
    * Computing uncertainties in the coefficients not implemented yet
      I'm using numpy's lstsq function, and I haven't figured out how to do it yet

    """
    def __init__(self,  y, s, model):
        """
        Input
        --------
        x
            (1d numpy array) ordinate (eg time)
        y
            (1d numpy array) coordinate (eg flux)
        s
            1 sigma uncertainties. Can be **None**, a scalar or
            a 1d numpy array

        model
            (1d numpy array) Model to fit to data
        Optional Inputs
        -----------------
        All optional inputs passed to ``func``
        """

#        if s is not None:
#            raise NotImplementedError("Weighted fits not implemented yet")

        order = 2
        x = np.ones_like(y)
        AbstractFitter.__init__(self,x, y, s, order, None)
        self.model = model
        self.check_inputs()

        if len(model) != len(y):
            raise ValueError("Input y and model must be same length")

        if not np.all(np.isfinite(self.model)):
            raise ValueError("Non-finite value found in self.model")

        self.fit()


    def check_inputs(self):
        AbstractFitter.check_inputs(self)

        if len(self.model) != len(self.y):
            raise ValueError("Length of model not equal to length of data")


    def fit(self):
        """Fit the model to the data and return the best fit
        """
        A = np.ones( (len(self.y), 2) )
        A[:,0] = 1 / self.s
        A[:,1] = self.model / self.s

        result = np.linalg.lstsq(A, self.y / self.s)

        self.cov = []
        self.param = result[0]


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

        a,b = self.param
        bestFit = a + b*self.model
        return bestFit

    def getCovariance(self):
        raise NotImplementedError

