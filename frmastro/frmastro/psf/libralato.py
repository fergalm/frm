from ipdb import set_trace as idebug
from astropy.io import fits as pyfits 
import scipy.ndimage as spImg
import numpy as np
import requests
import os 

from typing import Sequence
from .bbox import Bbox 


from .abstractlookup import AbstractLookupPrf, InterpRegImage, CroppedImage
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


    Libralato follow the pixel indexing convention used by HST and
    inherited by Kepler, K2, and Tess. Each file contains a series
    of oversampled PRF images, representing the PRF as observed
    in different corners of the image. 
    
    Each oversampled image
    represents the multiple realisations of the PRF for slightly
    different sub-pixel positions. The differences represent
    differences in sub-pixel centroid (if the centroid is
    in the centre of a pixel, that pixel will have the highest flux,
    but if the centroid is on the corner of a pixel, the 4
    nearest pixels will all have the same flux), and the 
    intra-pixel sensitivity variations.

    For MIRI, 16 different sub-pixel positions are represented
    (0.0, 0.25, 0.5, 0.75) in col and in row. 
    
    Notes
    --------------
    PRF models are oversampled by a factor of 4, so there are 16
    possible PRFs depending on the location of the centroid of the PRF 
    within the pixel.

    Variation in the PRF across the field of view is captured by
    having 4 prfs near the corners of the image and linearly interpolating
    between them

    """

    def __init__(self, cachePath, band="F1500W", refnum=5):
        """
        """
        self.refnum = refnum
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


    def download(self, url, cacheFile):
        r = requests.get(url, allow_redirects=True)
        r.raise_for_status()
        open(cacheFile, 'wb').write(r.content)


    def get(self, bbox:Bbox, params:Sequence) -> CroppedImage:
        img = AbstractLookupPrf.get(self, bbox, params)
        img = spImg.gaussian_filter(img, params[-1], mode='constant')
        return img

    def getModelPrfForColRow(self, col:float, row:float)-> InterpRegImage:
        _, nCol, nRow = self.subSampledPrfCube.shape
        intCol = int(np.floor(col))
        intRow = int(np.floor(row))
        fracCol = col - intCol 
        fracRow = row - intRow 

        #Get the indices of a single image
        ov = self.overSample
        cIdx = np.arange(ov, nCol, ov, dtype=int)
        rIdx = np.arange(ov, nRow, ov, dtype=int)
        cIdx -= 2
        rIdx -= 2
        cIdx, rIdx = np.meshgrid(cIdx, rIdx)
        
        #Extract out a single PRF image for a single position
        img = self.subSampledPrfCube[self.refnum, rIdx, cIdx] 

        dcol = (col) % 1
        drow = (row) % 1

        img = spImg.shift(img, (drow, dcol), order=3)
        return img
              

    # def getInterpRegPrfForColRow(self, col:float, row:float)-> InterpRegImage:
    #     _, nCol, nRow = self.subSampledPrfCube.shape
    #     intCol = int(np.floor(col))
    #     intRow = int(np.floor(row))
    #     fracCol = col - intCol 
    #     fracRow = row - intRow 

    #     ov = self.overSample
    #     cIdx = np.arange(ov, nCol, ov, dtype=int)
    #     rIdx = np.arange(ov, nRow, ov, dtype=int)
    #     cIdx -= int(np.floor(fracCol * ov))
    #     rIdx -= int(np.floor(fracRow * ov))
    #     print(int(np.floor(fracCol * ov)))
    #     print(int(np.floor(fracRow * ov)))
    #     cIdx, rIdx = np.meshgrid(cIdx, rIdx)
        
    #     regSamplePrfCube = self.subSampledPrfCube[:, rIdx, cIdx] 
    #     #Interpolate to get a single prf  
    #     #to do this we need to know locs of interpolated images
    #     #this is a placeholder
    #     img = regSamplePrfCube[6, :, :]

    #     return img
              

    def getDefaultBounds(self, img):
        #Remember, these can be over-ridden by passing in preferred bounds into fit()
        nr, nc = img.shape
        bounds = [
            (0, nc),   #col within the width of the image
            (0, nr),    #row within the height of the image
            (None, None),   #No limits on flux level
            (None, None),   #no limits on sky level
            (0, 2),   #rms of gaussin blue, in pixels
        ]
        return bounds




def test():
    obj = LibralatoMiri(".")

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
