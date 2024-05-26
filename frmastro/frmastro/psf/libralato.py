from ipdb import set_trace as idebug
from .abstractprf import AbstractPrfModel, rebin
from astropy.io import fits as pyfits 
import scipy.ndimage
import numpy as np
import requests
import os 

from typing import Sequence
from .bbox import Bbox 


from .abstractlookup import AbstractLookupPrf, InterpRegImage, RegSampledPrf, SubSampledPrf
from typing import List, NewType


SubSampledPrfCube = NewType('SubSampledPrf', np.ndarray)


class LibralatoMiri(AbstractLookupPrf):
    """
    Use an empiricially derived PRF model based on bright star
    observations with MIRI

    Full documentation at 
    https://iopscience.iop.org/article/10.1088/1538-3873/ad2551
    (Libralato et al 2024 PASP 136 034502))

    Fits files with the data in them taken from 
    https://www.stsci.edu/~jayander/JWST1PASS/LIB/PSFs/STDPSFs/MIRI/

    
    Notes
    --------------
    PRF models are oversampled by a factor of 4, so there are 16
    possible PRFs depending on the location of the centroid of the PRF 
    within the pixel.

    Variation in the PRF across the field of view is captured by
    having 4 prfs near the corners of the image and linearly interpolating
    between them
    """

    def __init__(self, cachePath, band="F1500W"):
        """
        """
        self.band = band 
        self.url = "https://www.stsci.edu/~jayander/JWST1PASS/LIB/PSFs/STDPSFs/MIRI/"
        self.overSample = 4  #Note (1, 2)
        self.subSampledPrfCube: SubSampledPrfCube
        self.subSampledPrfCube = self.loadCachedImage(cachePath, band)

        # = self.loadCachedImage(cachePath, band)

    def loadCachedImage(self, cachePath, band) -> SubSampledPrfCube:
        if not os.path.exists(cachePath):
            os.mkdir(cachePath)

        fn = f"STDPSF_MIRI_{band}.fits"
        cacheFile = os.path.join(cachePath, fn)
        if not os.path.exists(cacheFile):
            self.download("/".join([self.url, fn]), cacheFile)
        
        #This might not be right
        fits = pyfits.getdata(cacheFile)
        return fits

        #Note: Fits files contain multiple PSF images for different
        #parts of the detector. I should really be carrying these 
        #around and interpolating between them. But for quick 
        #and dirty work, just take a single example close to the middle.
        #See Fig 1 of ref
        return fits[6][:100, :100], hdr 

    def download(self, url, cacheFile):
        r = requests.get(url, allow_redirects=True)
        r.raise_for_status()
        open(cacheFile, 'wb').write(r.content)

    def getInterpRegPrfForColRow(self, col:float, row:float)-> InterpRegImage:
        nExample, nCol, nRow = self.subSampledPrfCube.shape
        intCol = int(np.floor(col))
        intRow = int(np.floor(row))
        fracCol = col - intCol 
        fracRow = row - intRow 

        
        
        cIdx = np.arange(0, nCol - self.overSample, self.overSample, dtype=int)
        rIdx = np.arange(0, nRow - self.overSample, self.overSample, dtype=int)
        cIdx += int(np.round(fracCol * self.overSample))
        rIdx += int(np.round(fracRow * self.overSample))
        cIdx, rIdx = np.meshgrid(cIdx, rIdx)
        
        regSamplePrfCube = self.subSampledPrfCube[:, cIdx, rIdx] 
        #Interpolate to get a single prf  
        #to do this we need to know locs of interpolated images
        #this is a placeholder
        return regSamplePrfCube[6, :, :]
              

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
    obj = MiriPsf(path, 3)

    bbox = Bbox(0, 0, 40, 40)
    import frmbase.support as fsupport
    with fsupport.Timer():
        for i in range(1):
            img = obj.get(bbox, [20, 30, 1, 0])

    import matplotlib.pyplot as plt
    import frmplots.norm as fnorm 
    norm = fnorm.HistEquNorm(100, vmin=0, vmax=img.max())
    plt.clf()
    plt.imshow(img, origin='lower', norm=norm)
    plt.colorbar()
    print(np.sum(img))