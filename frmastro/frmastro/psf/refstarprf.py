# -*- coding: utf-8 -*-

"""
Fit a model based on a reference star

Assumes the intrapixel sensitivity is constant 

"""

from .abstractlookup import AbstractLookupPrf

from .abstractprf import AbstractPrfModel
from .abstractprf import Bbox

import scipy.ndimage as spImg 
from typing import Sequence
import numpy as np


# 827,506

class OrigRefStarModel(AbstractPrfModel):
    def __init__(self, refImg):
        self.refImg = refImg 
        self.bbox = Bbox.fromImage(refImg)
        self.refRow, self.refCol = spImg.center_of_mass(refImg)

    def get(self, bbox, params):
        bbox = bbox or self.bbox 
        assert bbox.shape == self.bbox.shape
        col, row, flux, sky = params 

        dcol = col - self.refCol 
        drow = row - self.refRow 
        print(col, row, dcol, drow)

        model = spImg.shift(self.refImg, (drow, dcol), order=3)
        model *= flux 
        model += sky 
        return model 

    def getDefaultBounds(self, bbox: Bbox):
        nr, nc = bbox.shape
        bounds = [
            (0, nc),   #col within the width of the image
            (0, nr),    #row within the height of the image
            (None, None),   #No limits on flux level
            (None, None),   #no limits on sky level
        ]
        return bounds


class NewRefStarModel(AbstractLookupPrf):
    def __init__(self, refImg):
        self.refImg = refImg 
        self.bbox = Bbox.fromImage(refImg)
        self.refRow, self.refCol = spImg.center_of_mass(refImg)

    def getModelPrfForColRow(self, col:float, row:float):
        """Compute model flux for an image with size (numCols, numRows)

        Inputs
        -------
        numCols, numRows
            (ints) Shape of the image to compute the model PRF for
        params
            (tuple or array) Tunable parameters of the model


        The parameters are 
        `[col, row, flux, skyLevel]`

        Returns
        ----------
        A 2d numpy array representing the model PRF image.
        """
        # bbox = bbox or self.bbox 
        # assert bbox.shape == self.bbox.shape
        # col, row, flux, sky = params 

        # dcol = col - self.refCol 
        # drow = row - self.refRow 

        #I don't know why the 0.5 is needed. It may be specific to my 
        #test image
        dcol = (col - self.refCol + .5) % 1
        drow = (row - self.refRow + .5) % 1

        model = spImg.shift(self.refImg, (drow, dcol), order=3)
        return model 

    def getDefaultBounds(self, bbox: Bbox):
        nr, nc = bbox.shape
        bounds = [
            (0, nc),   #col within the width of the image
            (0, nr),    #row within the height of the image
            (None, None),   #No limits on flux level
            (None, None),   #no limits on sky level
        ]
        return bounds

