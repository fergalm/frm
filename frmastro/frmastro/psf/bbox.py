import numpy as np 

class Bbox:
    """A bounding box 
    
    Used for selecting portions of images
    
    Notes
    ------------
    There are multiple ways to define a bounding box,
    including 
    ```
    col0, row0, col1, row1
    col0, col1, row0, row1
    col0, row0, width, height
    ```

    One way will seem "natural" when you are writing your code, but
    entirely non-obvious to everyone else. Don't create a bbox
    by calling __init__ direction, but use one of the classmethods
    instead, which help you keep the conventions straight
    
    """
    
    def __init__(self, col0, row0, col1, row1):
        self.col0 = col0 
        self.row0 = row0
        self.col1 = col1 
        self.row1 = row1
        self.height = row1 - row0 
        self.width = col1 - col0

    def __repr__(self):
        strr = f"<Bbox cols: ({self.col0}, {self.col1}) "
        strr += f"rows: ({self.row0}, {self.row1})>"
        return strr

    @classmethod
    def fromCCRR(cls, col0, col1, row0, row1):
        return cls(col0, row0, col1, row1)
    
    @classmethod
    def fromCRCR(cls, col0, row0, col1, row1):
        return cls(col0, row0, col1, row1)
    
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

    @property
    def bottomLeft(self):
        return (self.col0, self.row0)
    
    @property 
    def topRight(self):
        return (self.col1, self.row1)
    
    @property
    def asSlices(self):
        """Return the bounding box as a tuple of slices

        Example
        -------------
        ```
            slr, slc = bbox.asSlices 
            image[slr, slc] = 0
        ```
        """
        cols = slice(self.col0, self.col0 + self.width)
        rows = slice(self.row0, self.row0 + self.height)
        return rows, cols

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

    def contains(self, bbox):
        """Not tested"""
        if self.col0 < bbox.col0 and self.row0 < bbox.row0:
            if self.width > bbox.width and self.height > bbox.height:
                return True 
        return False 

    def overlaps(self, bbox):
        raise NotImplemented
    
    def isInside(self, bbox):
        """Not tested"""
        if self.col0 > bbox.col0 and self.row0 > bbox.row0:
            if self.width < bbox.width and self.height < bbox.height:
                return True 
        return False 
