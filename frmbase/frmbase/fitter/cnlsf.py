
import numpy as np
import nlsf

__version__ = "$Id: cnlsf.py 2048 2015-05-15 21:09:45Z fmullall $"
__URL__ = "$URL: svn+ssh://flux.amn.nasa.gov/home/fmullall/svn/kepler/py/cnlsf.py $"



class Cnlsf(nlsf.Nlsf):
    def __init__(self, x, y, s, param, constrain, func=nlsf.nlGauss, **kwargs):
        """
        Fit a function to data using Levenberg-Marquardt optionally holding
        some of the parameters fixed.

        Input:
        ----------
        x
            (1d numpy array) ordinate (eg time)
        y
            (1d numpy array) coordinate (eg flux)
        s
            1 sigma uncertainties. Can be None, a scalar or
            a 1d numpy array

        param
            (list/array) Initial guesses are params.
            See the fitting function (passed in as func) for
            more details.

        constrain
            (list/array) Which params to fit. This array
            is the same length as param. If an element
            is set to True, the corresponding parameter is
            not fit.

        order
            (int) How many terms of func to fit

        func
            (function object) The analytic function to fit
            see notes on poly() below. Default is poly

        kwargs
            Additional arguments to pass to func.
        """

        if len(x) == 0:
            raise ValueError("x is zero length")

        if len(x) != len(y):
            raise IndexError("x and y must have same length")
        self.x = x
        self.y = y

        size = len(x)
        if s is None:
            self.s = np.ones(len(x))
        elif np.isscalar(s):
            self.s = np.ones((size)) * s
        else:
            if len(x) != len(s):
                raise IndexError("x and s must have same length")
            self.s = s

        if len(param) != len(constrain):
            raise IndexError("Param and Constrain must be the same length")
        self.param = param
        self.constrain = constrain
        self.func = func
        self.kwargs = kwargs

        #Should these be settable by kwargs?
        self.tol = 1e-12
        self.initialLambda = 1e-4
        self.maxLoops = 45


    def computeA(self):
        nPar = len(self.param)
        A = np.identity(nPar)

        #nlsf.c has a faster version of this algorithm, but
        #this is simpler to code
        for i in range(nPar):
            if self.constrain[i]:
                continue

            for j in range(nPar):
                if self.constrain[j]:
                    continue

                dfi = self.func(self.x, i, self.param, **self.kwargs)
                dfj = self.func(self.x, j, self.param, **self.kwargs)

                val = dfi * dfj / self.s**2
                A[i,j] = val.sum()

        return np.matrix(A)

    def computeBeta(self):
        nPar = len(self.param)
        beta = np.zeros( (1,nPar))

        for i in range(nPar):
            if self.constrain[i]:
                continue

            f = self.func(self.x, -1, self.param, **self.kwargs)

            df = self.func(self.x, i, self.param, **self.kwargs)

            val = (self.y-f)*df
            val /= self.s**2
            beta[0, i] = val.sum()

        return np.matrix(beta)
