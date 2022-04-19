from ipdb import set_trace as idebug
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import scipy.optimize as spopt
import frmbase.support 
import gp 

"""
Demonstration code fitting a GP to simulated data

The Nelder-Mead is slow, but performs better on this simple dataset
The L-BFGS-B is faster, but misbehaves.

In particular, the BFGS seems to expect its parameters in the range
[0,1], so I go through some contortions to scale my data appropriately.

Even then it seems to find a poorer fit than the Simplex
"""

Timer = frmbase.support.Timer
npmap = frmbase.support.npmap 

def main():
    np.random.seed(123456)

    #Create some simulated data with the desired covariance
    amp = 1
    scale0 = 12
    sigma = 1e-1

    kernel = gp.ExpSqrKernel(amp, scale0)
    x = np.arange(100)
    y = kernel.rvs(x)
    white_noise = sigma * np.random.randn(*(y.shape))
    y += white_noise

    #Set up the fit. Initial guess (pars0) is deliberately a poor choice
    pars0 = [4*amp, 8+scale0]
    args = (x, y, sigma)

    method = 'Nelder-Mead'
    with Timer(method):
        res1 = spopt.minimize(fitFunc, pars0, args=args, method=method)
    print(res1)
    
    bounds = [
        (0, 10),
        (1, 50),
    ]
    with Timer("BFGS"):
        res2 =  bfgs_fit(fitFunc, pars0, bounds, args)
    print(res2)

    #Plot the results.
    #Both do well in the region with data, but the extrapolation
    #from BFGS has unecessarily large uncertainty 
    xstar = np.arange(150)

    plt.clf()
    plt.plot(x, y, 'o')

    am, sc = res1.x
    kernel = gp.ExpSqrKernel(am, sc)
    mu, unc = gp.predict_function(x, y, xstar, sigma, kernel)
    y1 = mu - unc 
    y2 = mu + unc 
    plt.fill_between(xstar, y1, y2, alpha=.2, color='C1', label="Nelder-Mead")

    am, sc = res2.x
    kernel = gp.ExpSqrKernel(am, sc)
    mu, unc = gp.predict_function(x, y, xstar, sigma, kernel)
    y1 = mu - unc 
    y2 = mu + unc 
    plt.fill_between(xstar, y1, y2, alpha=.2, color='C2', label="L-BFGS-B")

    plt.legend()


def fitFunc(pars, x, y, sigma):
    """The function to fit. 
    
    Computes the log likelihood of the data given the parameters, then
    returns the negative log likelihood (because our methods minimise
    the objective)
    
    Inputs
    -------
    pars
        (list) Parameters of the kernel 
    x, y,
        (1d np arrays) The x and y coords of the data points 
    sigma
        (float) The assumed scatter per point

    Returns
    ---------
    float
    """
    amp, scale = pars 
    kernel = gp.ExpSqrKernel(1*amp, 1*scale)
    lnL = gp.compute_lnL(x, y, sigma, kernel)
    # print(pars, lnL)
    return -lnL


def bfgs_fit(func, initGuess, bounds, args):
    """Find the values of initGuess that minimise `func`
    
    The L-BFGS-B minimisation algorithim in `scipy.optimize` doesn't
    work weel for best fit parameters of different orders of magnitude.
    This function scales the guessed parameters to the region [0,1]
    while fitting, does clever things to the function to be optimized
    (`func`) doesn't know what we're up to

    `func` should be of the form::

        def func(pars, *args):
            ...

    For example, to fit a straight line to a dataset of {(x_i, y_i)}, you
    might use::

        def straight_line(pars, x, y):
            offset, slope = pars 
            ...
    """
    assert len(initGuess) == len(bounds)

    scaling_obj = ScalingObject(bounds)
    scaled_bounds = npmap(lambda x: [0,1], bounds)
    scaled_init_guess = scaling_obj.scaled_values(initGuess)

    scaled_func = lambda x: _fitting_func(func, x, args, scaling_obj)
    res = spopt.minimize(
        scaled_func, 
        scaled_init_guess, 
        bounds=scaled_bounds, 
        method='L-BFGS-B',
    )

    res.as_fit_x = res.x
    res.x = scaling_obj.true_values(res.x)
    return res


class ScalingObject():
    """A class to keep track of the conversion from true <--> scaled coords
    
    Scales coords are within the range 0,1 so long as they are withing the 
    stated bounds.
    """
    def __init__(self, bounds):
        """
        Inputs:
        ---------
        bounds
            (list of 2 tuples) Each tuple represents the lower and upper 
            bounds for a given fitted parameter. Must be floats. May not be **None**
        """
        self.bounds = bounds 

    def scaled_values(self, true_values):
        """Return a list of parameters each in the range"""[0..1]
        size = len(self.bounds)
        scaled_values = [0] * size 
        for i in range(size):
            val = true_values[i]
            lwr, upr = self.bounds[i]
            scaled_values[i] = (val-lwr) / (upr - lwr)
        return scaled_values 


    def true_values(self, scaled_values):
        """Return the true values of the functions"""
        size = len(self.bounds)
        true_values = [0] * size 
        for i in range(size):
            val = scaled_values[i]
            lwr, upr = self.bounds[i]
            true_values[i] = lwr + val * (upr-lwr)
        return true_values 
        




def _fitting_func(func, x, args, scaler):
    """Private function of bfgs_fit"""
    true_values = scaler.true_values(x)
    # print(f"Trying {x} => {true_values}")
    return func(true_values, *args)




#################

def fit_caiso():
    df = pd.read_csv('/home/fergal/exelon/davrt2018.csv')
    df['date'] = pd.to_datetime(df.date)

    df = df[df.rdoy.isin([6,7])]
    y = df.realtime.values
    y /= np.median(y)
    y -= 1
    x = np.arange(len(y))

    sigma = .05 
    amp0 = 1
    scale0 = 50 

    method = 'Nelder-Mead'
    # method = 'L-BFGS-B'
    pars0 = [amp0, scale0]
    args = (x, y, sigma)
    with Timer(method):
        res = spopt.minimize(fitFunc, pars0, args=args, method=method)
    print(res)

    plt.clf()   
    plt.plot(x, y, '.-')

    am, sc = res.x
    xstar = x.copy()
    kernel = gp.ExpSqrKernel(10*am, 10*sc)
    mu, unc = gp.predict_function(x, y, xstar, sigma, kernel)
    y1 = mu - unc 
    y2 = mu + unc 
    plt.fill_between(xstar, y1, y2, alpha=.2, color='C0')
    plt.fill_between(xstar, y1-unc, y2+unc, alpha=.2, color='C0')
