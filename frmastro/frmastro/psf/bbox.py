import numpy as np 

class Bbox:
    """A bounding box 
    
    Used for selecting portions of images
    """
    
    def __init__(self, col0, row0, col1, row1):
        self.col0 = col0 
        self.row0 = row0
        self.col1 = col1 
        self.row1 = row1
        self.height = row1 - row0 
        self.width = col1 - col0

    @classmethod 
    def fromSize(cls, col0, row0, width, height):
        col1 = col0 + width 
        row1 = row0 + height 
        return cls(col0, row0, col1, row1)

    @classmethod
    def fromImage(cls, img: np.ndarray):
        nr, nc = img.shape 
        return cls(0, 0, nc, nr)

    @property
    def shape(self):
        return (self.height, self.width)  #Numpy style

    def asExtent(self):
        """Return corners as matplotlib's imshow expects the extent keyword"""
        return [self.col0, self.col1, self.row0, self.row1]
    
    def getSubImage(self, img):
        """Return a sub-image represented by the bounding box"""
        nr, nc = img.shape 
        c0 = int(self.col0)
        c1 = int(min(self.col1, nc))
        r0 = int(self.row0)
        r1 = int(min(self.row1, nr))

        return img[r0:r1, c0:c1]
