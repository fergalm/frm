
from .abstractprf import NumericallyIntegratedPrf
from scipy.special import j1 as besselJ1
import numpy as np


class AiryPrf(NumericallyIntegratedPrf):
    """
    The prototype of a numerically integrated PRF.


    The Airy Function is 
    $A(r) = \left( \\frac{2 \mathrm{J1}(a r)}{(ar)} \right)^2$
    
    , where `a` is a model parameter, and J1 is a Bessel function.

    The Airy disk is a more accurate representation of a diffraction limited
    PRF than the Gaussian. It is slower to compute, but still faster than
    more realistic models 

    Parameters
    col, row, flux, scale, sky
    """


    def computeFunction(self, cols: np.ndarray, rows: np.ndarray, params:list) -> np.ndarray:
        col0, row0, flux0, scale, sky = params

        radius = np.hypot(cols-col0, rows-row0)
        x = scale * radius 

        flux = (2 * besselJ1(x) / x)**2
        flux *= flux0
        flux += sky 

        return flux 


    def getDefaultBounds(self, img):
        #Remember, these can be over-ridden by passing in preferred bounds into fit()
        nr, nc = img.shape
        bounds = [
            (0, nc),   #col within the width of the image
            (0, nr),    #row within the height of the image
            (None, None),   #No limits on flux level
            (0, None),   #scale must be postive
            (None, None),   #no limits on sky level
        ]
        return bounds

