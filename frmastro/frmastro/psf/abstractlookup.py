

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 20:17:34 2020

@author: fergal

Concepts
----------

bbox
    The bounding box of the postage stamp

Model PRF
    An image of what the code expects the PRF to look like for a given
    plate scale


regularPrf
    Model PRF precomputed at some reference position on the CCD (e.g at
    the corners)

SubsampledPrf
    An image from which a reference model Prf can be derived


"""

from ipdb import set_trace as idebug
import numpy as np

from .abstractprf import AbstractPrfModel
from .abstractprf import Bbox
from abc import ABC, abstractmethod

from typing import NewType, List, Tuple, Sequence


#The hardest part of these lookup prfs is keeping track of what kind of objects
#is represented by each numpy array. These Types help me keep track

#The sub-sampled image typically has (11x50) x (11x50) pixels. 50x50 for each pixel
#in the regularly sampled PRF. You need to do clever indexing into a sub-sampled PRF
#to create a regularly sampled PRF 
SubSampledPrf = NewType('SubSampledPrf', np.ndarray)

#The regularly-sampled typicaly has 11x11 pixels (sometimes 15x15 depending on the modout)
#and is defined at only 5 points on the CCD. Interpolate between those points to get
#the interpolated image. The RegSampledPrf changes with the sub-pixel position of the centroid
RegSampledPrf = NewType('RegSampledPrf', np.ndarray)

#The interpolated image typically has 11x11 pixels, and is formed by interpolating the RegSampledPrf
#objects
InterpImage = NewType('InterpImage', np.ndarray)

#The cropped image extracts only the pixels in the bbox from the InterpImage
CroppedImage = NewType('CroppedImage', np.ndarray)


class AbstractLookupPrf(AbstractPrfModel):
    """Store and lookup a previously computed PRF function

    This is the base class for PRF models using precomputed images created
    by, e.g, the Kepler, K2 or TESS missions.

    The Kepler and TESS style is to store the PRFs for 5 locations on a CCD,
    bottom left, top left, bottom right, top right and centre.

    For each location, 2500 images of the PRF are stored, one for each location on a
    grid with 0.02 pixel spacing. To evaluate the PRF for a given col row you need
    to evaluate the sub-pixel PRF at each of the corner locations, then interpolate
    between them to get the final result.

    This final PRF will have a shape (num_rows, num_cols) decided by the mission, 
    so the `get()` method here shoehorns that default image size into the requested
    bounding box.
    """

    def __init__(self, path):
        self.path = path
        self.cache = dict()

    @abstractmethod 
    def getInterpPrfForColRow(col, row) -> InterpImage:
        pass 

    def get(self, bbox:Bbox, params:Sequence) -> CroppedImage:
        """The model PRF at the requested location.

        params = [col, row]
        """

        assert len(params) == 2 
        col, row = params 

        nRowOut, nColOut = bbox.shape
        imgOut = np.zeros( (nRowOut, nColOut) )

        #Location of origin of bbox relative to col,row.
        #This is usually zero, but need not be.
        colOffsetOut = (bbox[0] - np.floor(col)).astype(np.int)
        rowOffsetOut = (bbox[2] - np.floor(row)).astype(np.int)

        interpPrf = self.getPrfAtColRow(col, row, *args)
        nRowPrf, nColPrf = interpPrf.shape
        colOffsetPrf = -np.floor(nColPrf/2.).astype(np.int)
        rowOffsetPrf = -np.floor(nRowPrf/2.).astype(np.int)

        di = colOffsetPrf - colOffsetOut
        i0 = max(0, -di)
        i1 = min(nColOut-di , nColPrf)
        if i1 <= i0:
            raise ValueError("Central pixel column not in bounding box")
        i = np.arange(i0, i1)
        assert(np.min(i) >= 0)

        dj = rowOffsetPrf - rowOffsetOut
        j0 = max(0, -dj)
        j1 = min(nRowOut-dj, nRowPrf)
        if j1 <= j0:
            raise ValueError("Central pixel row not in bounding box")
        j = np.arange(j0, j1)
        assert(np.min(j) >= 0)

        #@TODO: figure out how to do this in one step
        for r in j:
            imgOut[r+dj, i+di] = interpPrf[r, i]

        return imgOut


    def getDefaultBounds(self, bbox) -> List[Tuple]:
        nr, nc = bbox.shape
        bounds = [
            (0, nc),
            (0, nr),
        ]
        return bounds


# def loadFitsFiles(self, flist):
#     out = []

#     for f in flist:
#         out.append(self.loadSingleImage(f))
#     return out


# def loadSingleImage(self, imageSubPath):
#     """
#     Look for the image first in memory, then on disk, then on the web.

#     The image should be expected to exist at both
#     self.path + imageSubPath and self.url + imageSubPath

#     Returns
#     ------
#     A numpy 2d array
#     """

#     key = imageSubPath
#     if key not in self.imgCache:
#         cache_path = os.path.join(self.path, imageSubPath)

#         if not os.path.exists(cache_path):
#             url = os.path.join(self.url, imageSubPath)
#             self.download(url, cache_path)
#         self.imgCache[key] = pyfits.getdata(cache_path)

#     img = self.imgCache[key]
#     assert img.ndim == 2
#     return img


# def download(self, remoteUrl, local):
#     """Download the file at `remoteUrl` to the path `local` on disk"""

#     localDir = os.path.split(local)[0]
#     if not os.path.exists(localDir):
#         os.makedirs(localDir)

#     r = requests.get(remoteUrl, allow_redirects=True)
#     if r.status_code == requests.codes.ok:
#         open(local, 'wb').write(r.content)
#     else:
#         raise IOError("Failed to download %s" %(remoteUrl))
    




def interpolatePrf(refPrfArray, col, row, evalCols, evalRows):
    """Interpolate between 4 images to find the best PRF at col, row

    This function can be optionally called by getPrfAtColRow()

    Inputs
    ------
    refPrfArray
        Array of reference Prfs.
    evalCols
        reference columns for at which the model prfs in refPrfArray
        are computed for.
    evalRows
        reference rows for at which the model prfs in refPrfArray
        are computed for.

    Returns
    -------
    A specific Prf at the location col, row, computed by linear interpolation
    of the reference prfs in two dimensions.

    Note
    --------
    The variable names in this function assume that the refPrfArray
    is ordered as (left-bottom, right-bottom, left-top, right-top),
    but as long as the same order is used for refPrfArray, evalCols,
    and evalRows, the function will work.
    """

    assert len(refPrfArray)== len(evalCols)
    assert len(refPrfArray) == len(evalRows)

    #These asserts are true for Kepler and TESS. May not always
    #be true? They check that reference PRFs are arranged in a square
    #with sides parallel to the sides of the CCD.
    #If these assumptions are not true, the rest of the code does not
    #apply, and needs to be modified
    assert evalCols[0] == evalCols[2]
    assert evalCols[1] == evalCols[3]
    assert evalRows[0] == evalRows[1]
    assert evalRows[2] == evalRows[3]

    p11, p21, p12, p22 = refPrfArray
    c0, c1 = evalCols[:2]
    r0, r1 = evalRows[1:3]

    assert c0 != c1
    assert r0 != r1

    dCol = (col-c0) / (c1-c0)
    dRow = (row-r0) / (r1 - r0)

    #Intpolate across the rows
    tmp1 = p11 + (p21 - p11) * dCol
    tmp2 = p12 + (p22 - p12) * dCol

    #Interpolate across the columns
    out = tmp1 + (tmp2-tmp1) * dRow
    return out


