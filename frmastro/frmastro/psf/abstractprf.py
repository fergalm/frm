from abc import abstractmethod, ABC
import numpy as np 

from typing import Sequence
from .bbox import Bbox


class AbstractPrfModel(ABC):
    """
    Defines the interface for all PRF models 
    
    Every PRF model must implement two methods 
    
    - `get` is given a bounding box and some model properties
      and computes an image of the model PRF for those properties 
    - `getDefaultBounds` is a helper function that defines a
      range of allowed values for the model. These are helpful
      for fitting routines to constrain the fit to plausible values
      
    Daughter classes actually implement different models. See 
    `fastgauss` for a good prototype`
    """

    def __call__(self, bbox, params):
        return self.get(bbox, params)
    
    @abstractmethod
    def get(self, bbox:Bbox, params:Sequence) -> np.ndarray:
        """Compute a realisation of the model with the given parameters 

        Inputs
        ----------
        numCols, numRows
            (ints) Shape of the image to compute the model PRF for
        params
            (tuple or array) Tunable parameters of the model

        Returns
        ----------
        A 2d numpy array representing the model PRF image.
        """
        pass 

    @abstractmethod
    def getDefaultBounds(self, bbox:Bbox):
        """Get a list of the min and max allowed values for the parameters.

        Returns
        ---------
        A list of 2-tuples. The lenght of the list should be the same
        as the expected length of the `params` argument to `get`.
        Each tuple represents the min and max allowed values. If there
        is no bound, set the value to None.

        Example
        ------------
        For a Gaussian model with `params = [col, row, height, width]` the
        bounds might look like 

        ```
        bound = [
            (0, nc),        #Must be within the width of the image
            (0, nr),        #Must be within the height of the image
            (None, None),   #No constraints on height
            (0, None),      #Sigma must be positive
        ``
        """



class NumericallyIntegratedPrf(AbstractPrfModel):
    """
    Abstract class for computing model prfs using numerical intergration.
    
    Daughter classes must implement `computeFunction` which computs the
    the model value at an individual subpixel point. Selecting the 
    points to compute, and averaging over a pixel is handled by this class

    """
    def __init__(self, overres:int=4):
        """

        Inputs
        ----------
        overres
            Number of evaluations of the function across the width of a pixel
        """

        self.overres = overres

    def get(self, bbox:Bbox, params:Sequence):
        """

        Note
        ---------
        If overres is 2, the function is evaluated 4 times per pixel, with column locations
        of 1/4, 3/4, and similar row positions. If overres is 4 the function is evaluated 
        at 1/8 3/8, 5,8, 7/8. Etc.
        """
        nr, nc = bbox.shape 

        overres = self.overres
        offset = 1 / (2*overres) 
        colLoc = np.linspace(offset, nc-offset, nc*overres)
        rowLoc = np.linspace(offset, nr-offset, nr*overres)

        cols, rows = np.meshgrid(colLoc, rowLoc)
        flux = self.computeFunction(cols, rows, params)
        flux = rebin(flux, overres)  #This is the integration bit
        return flux 

    @abstractmethod
    def computeFunction(self, cols: np.ndarray, rows: np.ndarray, params:list) -> np.ndarray:
        """Compute the function at a given columne and row

        See AiryDisk for an example implementation
        """
        pass


def rebin(img, binSize):
    """Rebin an image.

    Example
    ----------
    If `img` is 128x256 pixels, and binSize is 2, the output image
    is 64x128, and each pixel value is the average of 4 input pixels.
    """

    nr, nc = img.shape 
    assert np.fmod(nr, binSize) == 0
    assert np.fmod(nc, binSize) == 0

    rr = int(nr/binSize)
    cc = int(nc/binSize)
    tmp = img.reshape(rr, binSize, cc, binSize)
    tmp = np.sum(tmp, axis=3)
    tmp = np.sum(tmp, axis=1)
    tmp = tmp / binSize**2
    return tmp

