"""
Created on Fri Dec  2 10:06:27 2016

@author: fergal
"""

from frm.fitter.AbstractFitter import AbstractFitter
import numpy as np

def nlPoly(x, order, param, **kwargs):

    nPar = len(param)
    if order == -1:
        sum = 0
        for i in range(nPar):
            sum += param[i] * x**(i)
        return sum

    if order >= nPar:
        raise ValueError("Requested order exceeds number of dimensions")

    return x**(order)


def nlGauss(x, order, param, **kwargs):
    if order > 2:
        raise ValueError("Order exceeds 2")

    A = param[0]
    x0 = param[1]
    s = param[2]

    f = (x-x0)/s
    f = np.exp(-.5*f**2)

    if order == -1:
        return A*f
    elif order == 0:
        return f
    elif order == 1:
        return A*f*(x-x0)/s**2
    else:
        return A*f*(x-x0)**2/s**3


def nlLorentz(x, order, params, **kwargs):
    A, x0, gamma = params

    dx = x - x0
    numer = .5 * gamma
    denom = dx**2 + (.5 * gamma)**2

    f = numer/ (np.pi * denom)

    if order == -1:
        return A * f
    elif order == 0:
        return f
    elif order == 1:
        return 2 * A * dx / f
    else:
        return A * (.5 * gamma * f + 1) / denom


def nlSine(x, order, param, **kwargs):
    if len(param) != 3:
        raise ValueError("Require 3 and only 3 parameters")

    if order > 3:
        raise ValueError("Order exceeds 2")

    twopi = 2*np.pi
    A, per, phi = param
    w = twopi/per
    x -= x[0]

    if order == -1:
        return A*np.sin(w*x - phi)
    elif order == 0:
        return np.sin(w*x - phi)
    elif order == 1:
        return A*np.cos(w*x-phi) * (-twopi/per**2)*x
    else:
        return -A*np.cos(w*x-phi)


def nlManySine(x, order, param, **kwargs):

    if len(param) % 3 != 0:
        raise ValueError("Number of params must be a multiple of three")

    if order >= len(param):
        raise ValueError("Illegal derivative request")

    twopi = 2*np.pi
    numSine = len(param) / 3
    if order == -1:
        val = 0
        for i in range(numSine):
            A, per, phi = param[i*3:(i+1)*3]
            w = twopi/per
            val += A*np.sin(w*x - phi)
        return val
    else:
        j = int( np.floor(order/3.))
        A, per, phi = param[j*3:(j+1)*3]
        w = twopi/per

        if order %3 == 0:   #Amplitude
            return np.sin(w*x - phi)
        elif order %3 == 1:   #Period
            return A*np.cos(w*x-phi) * (-twopi*x/per**2)
        else:   #Phase
            return -A*np.cos(w*x-phi)


    assert(0)   #Should never reach this point


def nlSineAndHarmonics(x, order, param, **kwargs):
    """Order of params:

        0: fundemental period
        1:  Amplitude1
        2: phase 1
        3: A2
        4: phi2
        ...

    """

    if (len(param) -1) % 2 != 0:
        raise ValueError("Param shoule be [p, A1, phi1, A2, phi2, ...]")

    if order >= len(param):
        raise ValueError("Illegal derivative request")


    per = param[0]
    twopi = 2*np.pi
    w = twopi/per


    if order == -1:
        val = 0
        numSine = int(.5*(len(param) -1 ))
        for i in range(numSine):
            i1 = 1+i*2
            i2 = i1+2
            A, phi = param[i1:i2]

            harm = i+1
            val += A*np.sin(harm*w*x - phi)
        return val

    elif order == 0:    #deriv wrt period
        val = 0
        numSine = int(.5*(len(param) -1 ))
        for i in range(numSine):
            i1 = 1+i*2
            i2 = i1+2
            A, phi = param[i1:i2]

            harm = i+1
            val +=  A*np.cos(harm*w*x-phi) * (-twopi*harm*x/per**2)
        return val
    else:

        j = int( np.floor( (order-1) / 2.))
        j1 = 1+j*2
        j2 = j1+2
        A, phi = param[j1:j2]
        harm = j+1

        if order %2 == 1:   #Amplitude
            return np.sin(harm*w*x - phi)
        else:   #Phase
            return -A*np.cos(harm*w*x-phi)


    assert(False) #Should never reach this point


def nlExp(x, order ,param, **kwargs):
    """Fit A*exp(tau (t-t0) )

    Inputs:
    param  [A tau t0]
    """

    if order >= 3:
        raise ValueError("Requested order exceeds number of dimensions")

    A = param[0]
    tau = param[1]
    t0 = param[2]

    dt = x-t0
    eTerm = A*np.exp(tau*dt)

    if order==-1:
        return eTerm
    elif order == 0:
        return eTerm/A
    elif order == 1:
        return dt*eTerm
    elif order == 2:
        return -tau*eTerm

    assert(False) #Should never reach this point


def logPowerLaw(x, order, param):
    """Fit A + k*log10(x), or log10(a x**k)

    Fit a log power law to the data. Powerlaws typically span many orders
    of magnitude, which can cause overflow in computations. The way around
    this is to take the logarithm of the data, and fit a power law to that
    instead.

    Yes, you can do this with a straight line fit, but this way involves
    less thinking. If you're going to be doing many fits, and speed is
    important, it's worth figuring out how.

    Inputs
    ---------
    x
        (1d array) x-axis to compute fit at
    order
        (int) Used to control fitting function.
    param
        Tuple of [A, k]. A is estimate of log10 of the value at x==1. k
        is the order of the power law.

    Returns
    -----------
    1d np float array
    """

    if order >= 2:
        raise ValueError("Requested order exceeds number of dimensions")

    A = param[0]
    k = param[1]

    if order == -1:
        return A + k * np.log10(x)
    elif order == 0:
        return 1
    else:
        return np.log10(x)


##########################################################################
##########################################################################
##########################################################################

class Nlsf(AbstractFitter):
    def __init__(self, x, y, s, param, func=nlGauss, **kwargs):
        """
        Fit a non-linear polynomial to data using the Levenberg-Marquardt method

        Input
        x
            (1d numpy array) ordinate (eg time)
        y
            (1d numpy array) coordinate (eg flux)
        s
            1 sigma uncertainties. Can be None, a scalar or
            a 1d numpy array

        order
            (int) How many terms of func to fit
        func
            (function object) The analytic function to fit
            see notes on poly() below. Default is poly
        kwargs
            Additional arguments to pass to func.
        """
        AbstractFitter.__init__(self, x, y, s, -1, func, **kwargs)
#        self.x = x
#        self.y = y
#        self.s = s

        self.param = param
#        self.func = func
#        self.kwargs = kwargs

        #Should these be settable by kwargs?
        self.tol = 1e-12
        self.initialLambda = 1e-4
        self.maxLoops = 45

        self.check_inputs()


    def check_inputs(self):
        AbstractFitter.check_inputs(self)

        if len(self.x) < len(self.param):
            raise ValueError("Can't fit %i data points with %i params" \
                %(len(self.x), len(self.param)))


    def fit(self):
        self.cov = None  #getResiduals() complains if this is not
        term = 2*self.tol
        lamba = self.initialLambda


        chi = self.getChiSq()
        assert(np.isfinite(chi) )

        loop = 0
        covar = None
        while loop < self.maxLoops and term > self.tol:
            A = self.computeA()
            beta = self.computeBeta()

            #Scale by lambda
            for i in range(len(self.param)):
                A[i,i] *= (1+lamba)

            #Invert
            try:
                covar = A.I #Inverts
            except np.linalg.LinAlgError as e:
                print( "nlsf.py:Nlsf.fit(): Error in fit: %s" %(e))
                raise e


            dPar = beta * covar

            trial = self.param + dPar
            trial = np.array(trial)[0]   #Convert from matrix

            self.param = trial
            testChi = self.getChiSq()
            if testChi <= chi:
                #We're converging, so increase speed
                lamba /= 10.

                term = (chi-testChi)/chi
                chi = testChi
            else:
                #We're diverging
                lamba *= 10

                #Reset params
                self.param -= np.array(dPar)[0]


            loop +=1

        #If we didn't converge, set loop to zero as a flag
        if loop == self.maxLoops:
            loop = 0

        self.cov = covar
        return loop


    def getBestFitModel(self, x=None):
        """Get best fit model to data

        Inputs:
        x   (1d numpy array) Ordinates on which to compute
            best fit model. If not specified use ordinates
            used in fit

        Return
        1d numpy array
        """
        if x is None:
            x = self.x

        return self.func(x, -1, self.param, **self.kwargs)


    def computeA(self):
        nPar = len(self.param)
        A = np.empty( (nPar, nPar))

        #nlsf.c has a faster version of this algorithm, but
        #this is simpler to code
        for i in range(nPar):
            for j in range(nPar):
                dfi = self.func(self.x, i, self.param, **self.kwargs)
                dfj = self.func(self.x, j, self.param, **self.kwargs)

                val = dfi * dfj / self.s**2
                A[i,j] = val.sum()

        return np.matrix(A)

    def computeBeta(self):
        nPar = len(self.param)
        beta = np.empty( (1,nPar))

        for i in range(nPar):
            f = self.func(self.x, -1, self.param, **self.kwargs)
            df = self.func(self.x, i, self.param, **self.kwargs)

            val = (self.y-f)*df
            val /= self.s**2
            beta[0, i] = val.sum()

        return np.matrix(beta)



