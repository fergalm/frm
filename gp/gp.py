from ipdb import set_trace as idebug
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
Scratch paper code where I try to build a model
to do "parameter learning" for data with covariance
using Gaussian processes.

In this model, I assume the model I'm trying to fit,
$f(x) =0 \forall x$. I also assume the variance
(i.e the iid errors of measurement are the same
for each point).

Extending my model to heteroskedastic errors, and 
useful functional forms shouldn't be too difficult.


Notes, an ExpSqr covariance matrix is non-invertible?

"""
class AbstractKernel():
    def rvs(self, x, size=1):
        """Draw random samples"""
        cov = self.cov(x, x)
        mu = np.zeros_like(x)
        y = np.random.multivariate_normal(mu, cov, size=size).transpose()
        return y


class ExpSqrKernel(AbstractKernel):
    def __init__(self, amp, scale):
        self.amp = amp
        self.scale = scale 

    def cov(self, x1, x2):
        
        size1 = len(x1)
        size2 = len(x2)
        out = np.empty((size1,size2))
        for i in range(size1):
            out[i] = (x1[i] - x2)**2
        
        # x1 = x1.reshape(size, size)  #Column matrix
        # x2 = x1.reshape(1, -1)  #Row matrix

        out /= 2 * self.scale**2
        out =  self.amp**2 * np.exp(-out)
        return out

class IdentityKernel(AbstractKernel):
    """You probably want to use the Null kernel, not this on"""
    def cov(self, x1, x2):
        assert len(x1) == len(x2)
        size = len(x1)

        #TODO, test that this will work before attempting
        out = np.eye(size) 
        return out

class NullKernel(AbstractKernel):
    """No covariance, or indeed variance"""

    def cov(self, x1, x2):
        assert len(x1) == len(x2)
        size = len(x1)

        #TODO, test that this will work before attempting
        out = np.zeros((size, size)) 
        return out 



from tqdm import tqdm
def loop():
    while True:
        best_fit_prediction()
        plt.pause(3)


def linear_search():
    r"""Create syntheic data, measure lnL for a bunch of guesses for \ell"""
    # np.random.seed(1234)
    amp = 1
    scale0 = 8
    sigma = 1e-2

    x = np.arange(100)
    kernel = ExpSqrKernel(amp, scale0)
    y = kernel.rvs(x)

    white_noise = sigma * np.random.randn(*(y.shape))
    y += white_noise
    # print(compute_lnL(x, y, 0, kernel))

    # plt.clf()
    # plt.plot(x, y, 'o', ms=8)
    # plt.plot(x, y - white_noise, '-')

    scales = np.linspace(4, 12, 20)
    lnL = []
    for scale in scales:
        kernel = ExpSqrKernel(amp, scale)
        lnL.append(compute_lnL(x, y, sigma, kernel))
    i0 = np.argmax(lnL)

    # plt.clf()
    # plt.plot(scales, lnL, 'ko-')
    # plt.axvline(scale0, color='r')
    # plt.axvline(scales[i0], color='b', lw=1)
    return scales[i0]
    # lnL = compute_lnL(x, y, sigma, kernel)
    # plt.plot(x, y-white_noise, '-')
    # plt.plot(x, y, 'o-', label="ln$\mathcal{L}$ = %.2f" %(lnL))
    # plt.legend()
    # size=10
    # y = ExpSqrKernl(amp, scale).rvs(x, size=size)
    # for i in range(size):
    #     plt.plot(x, y[:,i], '-')

    # assert y.shape == (100,10), y.shape

def best_fit_prediction():
    np.random.seed(123456)
    amp = 1
    scale0 = 16
    sigma = 1e-1

    x = np.arange(100)
    kernel = ExpSqrKernel(amp, scale0)
    y = kernel.rvs(x)

    white_noise = sigma * np.random.randn(*(y.shape))
    y += white_noise

    #The best fit function
    idx = np.ones_like(x, dtype=bool)
    idx[40:60] = False
    x0 = x[idx]
    y0 = y[idx]
    xstar = x
    mu, unc = predict_function(x0, y0, x, sigma, kernel)

    plt.clf()
    plt.plot(x0, y0, 'C0o', ms=8)
    plt.pause(.1)
    input()
    plt.plot(x, y - white_noise, 'k--')
    plt.plot(xstar, mu, 'C0-')

    y1 = mu - unc 
    y2 = mu + unc 
    plt.fill_between(x, y1, y2, alpha=.2, color='C0')

    y1 = mu - 2*unc 
    y2 = mu + 2*unc 
    plt.fill_between(x, y1, y2, alpha=.1, color='C0')

    # plt.axvspan(40, 60, alpha=.2)

def compute_lnL(x, y, sigma, kernel):
    """Compute log liklihood for special case of the predicted value being always zeros
    """

    x = np.array(x)
    y = np.array(y)
    size = len(x)
    assert len(y) == size

    SMat = np.eye(size) * sigma 
    KMat = kernel.cov(x, x)
    CMat = KMat + SMat

    #Solve xT Cmat-1 x
    if True:
        tmp = np.linalg.solve(CMat, y)
        lnL = np.dot(y.transpose(), tmp)
    else:
        Cinv = np.linalg.inv(CMat)
        lnL = np.dot(y.transpose(), Cinv)
        lnL = np.dot(lnL, y)

        # plt.clf()
        # plt.subplot(131)
        # plt.imshow(CMat)
        # plt.subplot(132)
        # plt.imshow(Cinv)
        # plt.subplot(133)
        # plt.imshow(np.linalg.inv(Cinv))
        # plt.pause(.1) 

    # assert np.all(lnL > 0), lnL
    lnL = compute_norm_term(CMat) - 0.5 * lnL

    try:
        return lnL[0,0]
    except IndexError:
        return lnL


def predict_function(x, y, xstar, sigma, kernel):
    r"""Predict mean and variance of best fit curve
    
    Predict the values of the function at locations
    xstar, given that the value at location 
    $x_i = y_i \pm \sigma$.

    Assumes that both sigma and the kernel are known

    See p34 of the doc
    
    """

    n = len(x)
    nstar = len(xstar)

    k_xstar_xstar = kernel.cov(xstar, xstar)
    k_xstar_x = kernel.cov(xstar, x)
    k_x_x = kernel.cov(x, x)
    Smat = sigma * np.eye(len(x))
    Cmat = k_x_x + Smat 

    # print(k_xstar_xstar.shape)
    assert k_xstar_xstar.shape == (nstar, nstar)

    # print(k_xstar_x.shape)
    assert k_xstar_x.shape == (nstar, n)

    # print(k_x_x.shape)
    assert k_x_x.shape == (n, n)

    # print(Cmat.shape)
    assert Cmat.shape == (n, n)

    mu = np.linalg.solve(Cmat, y)
    mu = np.dot(k_xstar_x, mu)
    mu += 0  #The assumed underlying function
    if mu.ndim > 1:
        mu = mu[:,0]
    # mu = mu.transpose()
    assert mu.ndim == 1 or mu.shape[1] == 1

    unc = np.linalg.solve(Cmat, k_xstar_x.transpose())
    unc = np.dot(k_xstar_x, unc)
    unc = k_xstar_xstar - unc 

    # plt.clf()
    # plt.imshow(unc)
    # plt.pause(.1)

    # print(unc.shape)
    # print(np.sqrt(unc.diagonal()))
    unc = np.sqrt(unc.diagonal())

    return mu, unc

def compute_norm_term(mat):
    r"""
    Compute the normalisation term for the Gaussian probaility distribution

    Let :math:`C` be the covariance matrix, and :math:`|C|` its determinant

    .. math::
        P(x_i) = \frac{1}{(2\pi)^{n/2) |C|^{1/2}} \exp{\ldots}
        lnL = \ln{\frac{1}{(2\pi)^{n/2) |C|^{1/2}}  \exp{\ldots}}
        lnL = \ln{(2\pi)^{-n/2}} + \ln{|C|^{-1/2}} + \ldots
        lnL = -(n/2)\ln{2\pi} - (1/2)\ln{|C|} + \ldots

    :math:`\ln{|C|}` can be computed by numpy with greater numerical stability
    than the determinant itself using `slogdet`
    """

    size = len(mat)
    assert mat.shape == (size, size)
    
    term = - 0.5 * size * np.log(2*np.pi)  #np.log is natural log
    sign, slogdet = np.linalg.slogdet(mat)
    # print(sign, slogdet, np.exp(slogdet))
    # assert sign > 0
    term +=  -0.5 * slogdet 
    return term 


def test_lnL_one():
    """Compute lnL for a few trivial cases"""
    x = [0]
    y= [0]
    kernel = NullKernel() #No covariance, only variance

    sigma = 1
    lnL = compute_lnL(x, y, sigma, kernel)
    expected = -.5*np.log(2*np.pi)
    assert np.isclose(lnL, expected), (lnL, expected)

    sigma = 4
    lnL = compute_lnL(x, y, sigma, kernel)
    expected = -.5 * np.log(2*np.pi*sigma)
    assert np.isclose(lnL, expected), (lnL, expected)

        # print(lnL, np.exp(lnL))
        # print( (2*np.pi)**-.5 * (sigma**2)**-.5)


def test_lnL_many():
    x = [0, 1]
    y= [0, 0]
    kernel = NullKernel()

    sigma = 1
    lnL = compute_lnL(x, y, sigma, kernel)
    expected = -1*np.log(2*np.pi*sigma)
    assert np.isclose(lnL, expected)

    size = 10
    x = np.arange(size)
    y = np.zeros(size)
    lnL = compute_lnL(x, y, sigma, kernel)
    expected = -.5*size*np.log(2*np.pi*sigma)
    assert np.isclose(lnL, expected)

# def compute_lnL(x, y, sigma, kernel):


def test_compute_norm_term():
    mat = np.eye(5)

    val = compute_norm_term(mat)
    assert isinstance(val, float)

def test_ExpSqrKernel():
    x1 = np.arange(5)
    x2 = x1.copy()

    cov = ExpSqrKernel(1,1).cov(x1, x2)
    assert cov.shape == (5,5)