# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 20:35:49 2020

TESS implemented their PRF lookup files in fits and matlab. This abstract class
contains common code for reading both. 

@author: fergal
"""

from pdb import set_trace as debug
import numpy as np

from .abstractlookup import AbstractLookupPrf, InterpImage, RegSampledPrf, SubSampledPrf, CroppedImage
from .abstractprf import Bbox
from typing import Sequence

class AbstractTess(AbstractLookupPrf):
    def __init__(self, path, sector, camera, ccd):
        AbstractLookupPrf.__init__(self, path)
        self.gridSize = 9
        self.path = path

        if sector < 1:
           raise ValueError("Sector must be greater than 0.")
        if (camera < 1) or (ccd < 1):
            raise ValueError("Camera or CCD is less than 1.")
        if (camera > 4) or (ccd > 4):
            raise ValueError("Camera or CCD is larger than 4.")

        self.sector = sector
        self.camera = camera
        self.ccd = ccd

    def __str__(self):
        msg = "<Tess PRF model for (sector, camera, ccd) = (%i,%i,%i). Path is %s>"
        msg = msg %(self.sector, self.camera, self.ccd, self.path)
        return msg


    def get(self, bbox:Bbox, params:Sequence) -> CroppedImage:
        """Get PRF for a bounding box.

        See `getPrfAtColRow()` and documentation in the same method in the parent class
        """
        self.validateInputs(params)
        return AbstractLookupPrf.get(self, bbox, params)


    def validateInputs(self, params:list) -> None:
        col, row = params
        if col < 45 or col > 2091:
            raise ValueError("Requested column (%i) not on phyiscal CCD [45,2091]" %(col))

        if row < 1 or row > 2047:
            raise ValueError("Requested row (%i) not on phyiscal CCD [0,2047]" %(row))


    def sectorLookup(self, sector:int) -> int:
        """Map sector of observation to PRF sector file number.

        Sectors 1-3 (inclusive) use the PRF generated for sector 1
        Sectors 4 and above use the PRF generated for sector 4

        Note, if you update this function, you may also need to update
        the getFitsDateStr() method in the TessFitsPrf daughter class
        (yeah, there's probably a neater way to do this)
        """
        if sector > 3:
            return 4
        return 1



    #Probably belongs in baseclass, if I can adapt Kepler to use it
    #Isn't currently used by TessMatlabPrf, but should be
    def getRegularPrfFromSubsampledPrf(self, subSampledModel:SubSampledPrf , col, row) -> RegSampledPrf:
        """Convert a subsampled PRF to a regularly sampled one

        Subsampled => information on many partial-pixel offsets is encoded
        in the image. Regular Sampling means only one partial pixel offset
        is encoded in the model, and the model can be compared directly to
        a real image.

        The 13x13 pixel PRFs on at each grid location are sampled at a 9x9 intra-pixel grid, to
        describe how the PRF changes as the star moves by a fraction of a pixel in row or column.
        To extract out a single PRF, you need to address the 117x117 array in a slightly funny way
        (117 = 13x9),

        .. code-block:: python

            img = array[ [colOffset, colOffset+9, colOffset+18, ...],
                         [rowOffset, rowOffset+9, ...] ]

        """

        gridSize = self.gridSize
        colOffset, rowOffset = self.getOffsetsFromPixelFractions(col, row)

        #Number of pixels in regularly sampled PRF. Typically 13x13
        nCol, nRow = subSampledModel.shape
        assert nCol % gridSize == 0
        assert nRow % gridSize == 0
        nColOut = nCol / float(gridSize)
        nRowOut = nRow / float(gridSize)

        iCol = colOffset + (np.arange(nColOut) * gridSize).astype(np.int)
        iRow = rowOffset + (np.arange(nRowOut) * gridSize).astype(np.int)

        #Don't understand why this must be a twoliner
        tmp = subSampledModel[iRow, :]
        return tmp[:,iCol]

    def getOffsetsFromPixelFractions(self, col, row):
        return self.old_getOffsetsFromPixelFractions(col, row)

    def new_getOffsetsFromPixelFractions(self, col, row):
        """Compute offset into sub-sampled model to retrieve the
        regular model for a given fractional pixel position

        For some reason, the subsampled PRFs in TESS seem to be stored
        in reverse order, hence the 1-colFrac
        """
        gridSize = self.gridSize

        colFrac = np.remainder(float(col), 1)
        rowFrac = np.remainder(float(row), 1)

        colOffset = np.round((gridSize * (1 - colFrac))) % gridSize
        rowOffset = np.round((gridSize * (1 - rowFrac))) % gridSize

        assert colOffset >=0 and colOffset < gridSize
        assert rowOffset >=0 and rowOffset < gridSize

        return int(colOffset), int(rowOffset)

    def old_getOffsetsFromPixelFractions(self, col, row):
        """Map the fractional part of the col,row position to an offset into the
        full prf image.

            This code commented out because I believe its performance
            is not quite right. But it is the code used by Kepler, so
            I don't know how confidence I can be, and I'm keeping it around
        This function was developed through trial and error, rather than by
        reference to any design document.
        """
        gridSize = self.gridSize

        colFrac = np.remainder(float(col), 1)
        rowFrac = np.remainder(float(row), 1)

        colOffset = gridSize - np.round(gridSize * colFrac) - 1
        rowOffset = gridSize - np.round(gridSize * rowFrac) - 1

        #TODO: Bounds checks like this don't belong in an inner loop. Remove them
        #once you're sure they never get triggered.
        if colOffset >= gridSize:
            raise ValueError("Requested column offset (%i) too large" %(colOffset))
        if rowOffset >= gridSize:
            raise ValueError("Requested row offset (%i) too large" %(colOffset))

        assert colOffset < gridSize
        assert rowOffset < gridSize

        return int(colOffset), int(rowOffset)



