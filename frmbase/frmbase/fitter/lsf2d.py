# -*- coding: utf-8 -*-

import numpy as np

__version__ = "$Id: lsf2d.py 2010 2015-04-28 16:51:36Z fmullall $"
__URL__ = "$URL: svn+ssh://flux.amn.nasa.gov/home/fmullall/svn/kepler/py/lsf2d.py $"


##
#Common functions to fit
##
def poly2d(x, y, orderX, orderY, **kwargs):
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


    a = x**orderX
    b = y**orderY
    return a*b







class Lsf2d():
    """Least squares fit to an analytic function

    Based on lsf.py, which is in turn based on lsf.c in Wqed,
    which in turn is based on Bevington and Robinson.

    """

    def __init__(self, x, y, v, s, order, func=poly2d, **kwargs):
        """
        Fit a 2 dimension function to two dimension data.

        Input can be:
        o A single 2d array
        v = np.ones( (10,10) )
        Lsf2d(None, None, v...

        o 2d arrays of xordinate of each point, y ordinate of each
        point and value at each point, as created, eg by np.meshgrid
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        Lsf2d(x,y, v...

        o 1d arrays of x and y coords of a point and its values
        x = np.arange(9)
        y = np.arange(9)
        v = np.ones(9)
        Lsf2d(x,y, v...


        Input
        x       (1d or 2d numpy array) ordinate of data in first dimension.
                Can be set to None if y is set to None
        y       (1d or 2d numpy array) ordinate of data in second dimension.
                Can be set to None if x is set to None
        v       (1d or 2d numpy array) value of data at each (x,y) pair
        s       Uncertainty on each value of v. s.shape == v.shape

        order   (int) How many terms of func to fit
        func    (function object) The analytic function to fit
                see notes on poly() below. Default is poly2d
        kwargs  Additional arguments to pass to func.

        Returns:
        A 2d numpy array of size (order,order). The top left triangle
        of the matrix is filled, the bottom right is empty.

        The ij th element of the return array is the best fit coeff
        for the x^i \\times y^j term of the input polynomial
        """

        #If user passes in None for x and y, construct useful
        #values
        if x is None and y is None:
            r, c = v.shape
            x , y = np.meshgrid( np.arange(r), np.arange(c) )

        #Sanity checks on length on input
        if len(x) == 0:
            raise ValueError("x is zero length")

        if x.ndim != y.ndim:
            raise IndexError("x and y must have same shape")

        if x.ndim == 1:
            x,y = np.meshgrid(x,y)

        if x.shape != v.shape:
            raise IndexError("x and v must have same length")

        if s is None:
            s = np.ones_like(x)
        elif np.isscalar(s):
            s = np.ones_like(x) * s
        elif s.shape != v.shape:
            raise IndexError("v and s must have same shape is s is an array")
            s = s

        #If passed in 2d arrays for x,y and v, flatten into 1d arrays
        self.shape = v.shape
        self.x = x.flatten()
        self.y = y.flatten()
        self.v = v.flatten()
        self.s = s.flatten()

        if order >= len(self.x):
            raise ValueError("Length of input must be at least one greater than order")
        size = len(self.x)


        self.order = order
        self.func = func
        self.kwargs = kwargs

        self.fit()


    def fit(self):
        """Fit the function to the data and return the best fit
        parameters"""
        dataSize = len(self.x)

        #Check array lengths agree
        n = self.order
        order2d = int(.5*n*(n+1))

        df = np.empty((dataSize, order2d))
        k=0
        for i in range(self.order):
            for j in range(self.order - i):
                #print k, i, j
                df[:,k] = self.func(self.x, self.y, i, j, \
                    **self.kwargs)
                df[:,k] /= self.s
                k+= 1
        assert(k == order2d)

        A = np.matrix(df.transpose()) * np.matrix(df)
        covar = np.linalg.pinv(A)

        wy = np.matrix(self.v / self.s)
        beta = wy * df
        params = covar * beta.transpose()

        #Store results and return
        self.param = np.array(params.transpose())[0]   #Convert from matrix
        self.cov = covar

        return self.param


    def getParams(self):
        """Return params as a 2d array"""
        out = np.zeros( (self.order, self.order))
        k=0
        for i in range(self.order):
            for j in range(self.order - i):
                out[i,j] = self.param[k]
                k += 1
        return out

    def getVariance(self):
        """Variance is the sum of the squares of the residuals.
        If your points are gaussian distributed, the standard deviation
        of the gaussian is sqrt(var).
        """

        resid = self.getResiduals()

        val = np.sum( resid**2)
        val /= len(self.x) - self.order
        return val


    def getCovariance(self):
        """Get the covariance matrix

           Return:
           np.matrix object
        """
        assert(self.cov is not None)
        return self.cov

    def getUncertainty(self):
        """Get the uncertainty on the best fit params
            Return:
            1d numpy array
        """
        assert(self.cov is not None)
        unc = np.sqrt(self.cov.diagonal())
        val = np.array(unc)[0]
        return val.reshape( (self.order, self.order))


    def getChiSq(self):
        """Get chi square for fit

            Return double
        """
        chi = self.getResiduals() / self.s
        chisq = chi**2
        return np.sum(chisq)

    def getReducedChiSq(self):
        """Get reduced chi-squared."""
        num = len(self.x) - self.order
        return self.getChiSq()/float(num)

    def getBestFitModel(self):
        """Get best fit model to data

        Optional Inputs:
        x, y   (1d numpy array) Ordinates on which to compute
            best fit model. If not specified use ordinates
            used in fit

        Return
        2d numpy array

        Not tested yet
        """

        assert(self.param is not None)

        n = self.order
        order2d = int(.5*n*(n+1))
        bestFit = np.zeros( len(self.v))

        par = self.getParams()
        for i in range(self.order):
            for j in range(self.order - i):
                val = self.func(self.x, self.y, i, j, \
                    **self.kwargs)
                #import pdb; pdb.set_trace()
                bestFit += par[i,j]* val

        return bestFit.reshape( self.shape)


    def getResiduals(self):
        """Get residuals, defined as y - bestFit

        Returns:
        1d numpy array
        """
        return self.v.reshape(self.shape) - self.getBestFitModel()







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

