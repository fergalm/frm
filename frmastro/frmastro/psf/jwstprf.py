
from .abstractprf import AbstractPrfModel, rebin
from astropy.io import fits as pyfits 
import scipy.ndimage
import numpy as np

from typing import Sequence
from .bbox import Bbox 

class MiriPsf(AbstractPrfModel):
    """
    Create a PRF model from WebbPsf for MIRI

    """

    def __init__(self, path, overres=1):
        fits, hdr = pyfits.getdata(path, header=True)
        overSamp = hdr['OVERSAMP']
        img = scipy.ndimage.zoom(fits, overres, order=0)
        img /= np.sum(img)
        
        self.img = img 
        self.overres = overres * overSamp

        #Center of image is assumed to be centre of the psf 
        nr, nc = fits.shape 
        self.col0 = nc / (2* hdr['OVERSAMP'])
        self.row0 = nr / (2* hdr['OVERSAMP'])


    def get(self, bbox:Bbox, params:Sequence) -> np.ndarray:
        col0, row0, flux, sky = params

        col_roll = int((col0 - self.col0) * self.overres)
        row_roll = int((row0 - self.row0) * self.overres)
        img = np.copy(self.img)
        img = np.roll(img, col_roll, axis=1)  #Check this
        img = np.roll(img, row_roll, axis=0)
        
        img = rebin(img, self.overres)
        img *= flux
        img += sky

        #Select only the bbox
        nr, nc = bbox.shape 
        img = img[:nr, :nc]
        return img


    def getDefaultBounds(self, img):
        #Remember, these can be over-ridden by passing in preferred bounds into fit()
        nr, nc = img.shape
        bounds = [
            (0, nc),   #col within the width of the image
            (0, nr),    #row within the height of the image
            (None, None),   #No limits on flux level
            (None, None),   #no limits on sky level
        ]
        return bounds



def test():
    path = "/home/fergal/data/jwst/webbpsf-data/MIRI/psf/PSF_MIRI_in_flight_opd_filter_F1500W.fits"
    obj = JwstPsf(path)

    bbox = Bbox(0, 0, 40, 40)
    import frmbase.support as fsupport
    with fsupport.Timer():
        for i in range(1000):
            img = obj.get(bbox, [20, 30, 100, 0])

    import matplotlib.pyplot as plt
    plt.clf()
    plt.imshow(img, origin='lower', vmax=.01)
    plt.colorbar()