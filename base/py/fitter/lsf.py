"""
Created on Fri Dec  2 10:00:59 2016

@author: fergal
"""

#import matplotlib.pyplot as plt
#import matplotlib as mpl
#import pandas as pd
import numpy as np

from frm.fitter.AbstractFitter import AbstractFitter

def poly(x, order, **kwargs):
    """Polynomial of the form \Sigma a_i x**i

       This is a working prototype of the analytic model to
       be used by lsf. This function is never called by the user
       directly.

       In general, the Lsf class fits an analytic function of
       the form f(x_i) = \sigma a_j g_j(x_i),
       where a_i is a fittable parameter, and g_j is a
       function of x_i. Calling poly with a given order
       returns the value of the derivative of f(x) with respect
       to a_order = g_order(x_i)

       g_j can depend on other parameters. These are passed
       in with kwargs.

       Inputs:
       x        (1d numpy array) ordinates
       order    (int) Which parameter to take the derivative with
                respect to
       kwargs   (dict) Optional extra arguments to pass to the
                function. poly doesn't require any.
    """


    return x**order


def sine(x, order, **kwargs):
    """Single sine curve of known period.

    Fits the equivalent of A.sin( 2 pi x /period)
    The equation is re-written as r1 sin(wx) + r2 cos(wx)
    where w = 2pi/period, and r1 and r2 are the fit coefficients

    To convert these parameters to the more useful amplitude
    and phase, see computeAmplitudeAndPhase below

    Period to fit should be passed in as kwarg with key 'period'

    """

    period = kwargs['period']
    w = 2*np.pi/period
    if order == 0:
        return np.sin(w * x)
    elif order == 1:
        return np.cos(w * x)
    else:
        raise ValueError("Order should be zero or one")


def fixedPowerLaw(x, order, **kwargs):
    """Fix a series of power laws where the exponent is known

    Example:
    A1 x**(b1) + A2 x**(b2) + ...
    where b_i is known, and A_i is to be fit.


    Inputs:
    x        (1d numpy array) ordinates
    order    (int) term of function to fit

    Required kwargs:
    exponents   (float or iterable) Exponents of power laws to fit (the
                b_i's above


    Notes:
    the optional argument exponents must be passed to Lsf when calling
    this function.

    order must equal number of exponents

    To fit a function where the exponents are not known you need a
    non-linear fit
    """


    exponents = kwargs.pop('exponents', None)
    if exponents is None:
        raise ValueError("fixedPowerLaw requires list of expoenents passed as kwarg")

    if not hasattr(exponents, "__len__"):
        exponents = [exponents]

    assert(order <= len(exponents))
    return x**exponents[order]





class Lsf(AbstractFitter):
    """A linear least squares fitter"""
    def __init__(self, x, y, s, order, func=poly, **kwargs):
        """
        Input
        --------
        x
            (1d numpy array) ordinate (eg time)
        y
            (1d numpy array) coordinate (eg flux)
        y
            1 sigma uncertainties. Can be **None**, a scalar or
            a 1d numpy array

        order
            (int) How many terms of func to fit
        func
            (function object) The analytic function to fit
            see notes on poly() below. Default is poly

        Optional Inputs
        -----------------
        All optional inputs passed to ``func``
        """

        AbstractFitter.__init__(self, x, y, s, order, func, **kwargs)
        self.check_inputs()
        self.fit()


    def check_inputs(self):
        AbstractFitter.check_inputs(self)

        order = self.order
        if order < 1:
            raise ValueError("Order must be at least 1")

        if order >= len(self.x):
            raise ValueError("Length of input must be at least one greater than order")

    def fit(self):
        """Fit the function to the data and return the best fit
        parameters

        Todo
        ---------
        This is old code, and I can write something more numerically stable
        """
        dataSize = len(self.x)

        #Check array lengths agree

        df = np.empty((dataSize, self.order))
        for i in range(self.order):
            df[:,i] = self.func(self.x, i, **self.kwargs)
            df[:,i] /= self.s

        A = np.matrix(df.transpose()) * np.matrix(df)
        covar = A.I
        wy = np.matrix(self.y / self.s)
        beta = wy * df
        params = beta * covar

        #Store results and return
        self.param = np.array(params)[0]   #Convert from matrix
        self.cov = covar

        return self.param


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


    def getBestFitUnc(self, x=None):
        """Get the 1 sigma unc estimate around the best fit

        Algorithm still under development
        """

        if x is None:
            x = self.x

        val = 0
        #@TODO speed this up
        for i in range(self.order):
            for j in range(self.order):
                val += self.cov[i, j] * \
                        self.func(x, i, **self.kwargs) * \
                        self.func(x, j, **self.kwargs)

        assert np.all(val > 0)
        val = np.sqrt(val)

        fit = self.getBestFitModel(x)
        out = np.zeros( (len(x), 2) )
        out[:,0] = fit - val
        out[:,1] = fit + val

        return out


def computeAmplitudeAndPhase(par):
    """Converts the results from fitting sine function to
    amplitude and phase"""

    amp = np.hypot(par[0], par[1])

    #r1 is cos(phase), r2 is -sin(phase) (because fitting wx-phi)
    # You can remove the - r2 if you add a minus to the lineAndSine()
    # for case 3. Don't do this, you'll only confuse yourself
    r1 = par[0]/amp
    r2 = -par[1]/amp

    #Remember the sign of the cos and sin components
    invcos = np.arccos(np.fabs(r1))
    invsin = np.arcsin(np.fabs(r2))

    #Decide what quadrant we're in
    #if(r1>=0 && r2 >=0) //1st quadrant, do nothing
    if r1 <=0 and r2>=0:
        #Second quadrant
        invcos = np.pi - invcos
        invsin = np.pi - invsin
    elif r1 <=0 and r2<=0:
        #Third quadrant
        invcos += np.pi
        invsin += np.pi
    elif r1>=0 and r2 <= 0:
        #4th quadrant
        invcos = 2*np.pi - invcos
        invsin = 2*np.pi - invsin


    #Choose the average of the two deteminations to
    #reduce effects of roundoff error
    phase = .5*(invcos+invsin)
    return amp, phase

