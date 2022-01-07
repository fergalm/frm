# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 19:34:33 2020

@author: fergal

https://en.wikipedia.org/wiki/Hilbert_curve


TODO
-----
cell2hilbert doesn't seem to do anything when I bitshift the values.
I don't understand what's going on.

Need a linear search function,
    Maybe also a growing linear search. Given bracket [a,b],
    search, put the pivot at a+1, a+2, a+4?

Glossary
------------

lng,lat
    Positions on the sphere

level
    How fine a level of detail to draw the hilbert line. Referenced as 'n'
    in the code

Hilbert Square
    A square with sides of length (2**level)

Hilbert Distance
    The distance along the space filling curve travelled to reach a given
    cell. Commonly called 'hilbert' in the code

Cell
    A single grid element in the Hilber square. Referenced by two integers,
    i and j, each of which is in the range [0, 2**level)

Parent Cell
    The cell at level at a lower level that contains a given cell is its
    parent.

Daughter Cell
    A cell at a higher level fully contained by the cell in question

Envelope
    A rectangle in lng/lat space that completely contains an AOI and has
    sides that follow lines of constant longitude and latitude



Coordinate System
------------------
The cell system has its origin at bottom left, with i increasing in the
horizontal direction and j in the vertical direction. For a n=2 segmentation
the hilbert curve traces the square like this

          11   10    ^
                     |
           8    9    |
                     |  j
 1    2    7    6    |
                     |
 0    3    4    5    |

 ------------------>
        i


Performance
--------------
Test: 1.9 million locations in 1.02 sec

############


Bitshifting
----------------
x << 4 == x * 2**4 ((shifts bits 4 spots to the left)
x >> 4 == x * 2**(-4)  shift bits 4 spots to the right

x >> n is the equivalent of converting a hilbert number
in a level 32 Hilbert square to the number in a 32-n hilbert square.

This can be useful in debugging where you can translate a big number
into a more manegable one.


        The parameter space is divided into x (2**level) squares.

cell a single grid point
To do a geometry intersection

Input envelope:
    compute d for 4 corners with n=25
    reduce n a step, compute blocks

        Either initial corners are in adjacent blocks,
        or reducing n a step produces blocks that are
        closer to each other.

        So keep reducing blocks until they 'touch'

    Search those 4 blocks

But then you still have to do a linear search in numpy
to filter down to just those blocks. I don't think you
gain anything

No, not true, you can do a slice slice(d1, d2),
which is more effecient because you don't
test each element, just select the good ones

"""

from pdb import set_trace as debug
import numpy as np

BIT_WIDTH = 32
DTYPE = np.uint32

def lnglat2hilbert(lng_deg, lat_deg, n=25):
    isFloat = isinstance(lng_deg, float)

    cell = lnglat2cell(lng_deg, lat_deg, n)
    hilbert = cell2hilbert(cell, n)

    #If input was float, return int not array
    if isFloat:
        hilbert = hilbert[0]
    return hilbert


def lnglat2cell(lng_deg, lat_deg, n=25):

    assert n < BIT_WIDTH, "Max bit depth exceeded"

    cast = np.atleast_1d
    dtype = DTYPE

    x = cast( (lng_deg + 180) / 360.)
    y = cast( (lat_deg + 90) / 180.)
    assert len(x) == len(y)

    scale = 2**n
    cell = np.empty((len(x), 2), dtype=dtype)
    cell[:,0] = (x * scale).astype(dtype)
    cell[:,1] = (y * scale).astype(dtype)
    return cell


def cell2hilbert(cell, n):
    """
    Inputs
    -----
    cell
        (2d np array) columns are `i` and `j` (see Glossary)

    n
        (int) level of cell


    Returns
    --------
    Array of 32 bit unsigned integers
    """
    assert cell.ndim == 2
    assert cell.shape[1] == 2

    i = cell[:,0]
    j = cell[:,1]

    max_val = 2**n
    assert np.all(i >=0) and np.all(i <= max_val)
    assert np.all(i >=0) and np.all(j <= max_val)

    d = np.zeros(len(i), dtype=DTYPE)
    # d = 0
    #Check n is a power of 2
    sarr = 2**np.arange(n-1, -1, -1, dtype=DTYPE)

    # debug()
    const = DTYPE(3)
    for s in sarr:
        ri = (i & s) > 0
        rj = (j & s) > 0
        d += s * s * ((const * ri) ^ rj);
        i, j = rot(n, i, j, ri, rj);
    return d << BIT_WIDTH


def rot(n, i, j, ri, rj):

    idx1 = (rj == 0)
    idx2 = idx1 & (ri == 1)

    i[idx2] = (n-1) - i[idx2]
    j[idx2] = (n-1) - j[idx2]

    t = i[idx1]
    i[idx1] = j[idx1]
    j[idx1] = t

    return i, j


########################################3


def convertEnvelopeToFourHilberts(env_deg, maxLevel):
    """Compute the hilbert number for the 4 smallest cells that completely
    cover an envelope. All pings inside the envelope have the first `level`
    bits of their hilbert number agree with one of the hilbert numbers
    returned by this function.

    For example, to find all the pings close to an evelope look for::

        hilbertNumberForData & (1 << level) == hilbert[0] or
        hilbertNumberForData & (1 << level) == hilbert[0] ...

    where hilbert[i] is one of the 4 hilbert numbers returned by this
    function, and hilbertNumberForData is an array of hilbert numbers
    for the input data to be intersected with the geometry

    Inputs
    -------
    env_deg
        list of 4 floats representing min and max longitudes and latitudes
        of the bounding box of a geometry. Format is [lng1, lng2, lat1, lat2]

    maxLevel
        (int) The highest level at which the cells might possibly be
        adjacent. See `findAdjacentParentCells`

    Returns
    -----------
    An array of 4 hilbert numbers.
    The level of the cells represented by those numbers.
    """
    #TODO, these should raise valueerrors?
    assert len(env_deg) == 4
    assert env_deg[0] < env_deg[1]   #Longitudes are in order
    assert env_deg[2] < env_deg[3]   #Latitudes

    cells, level = getCellsForEnvelope(env_deg, maxLevel)
    hilbert = cell2hilbert(cells[:,0], cells[:,1], level)
    return hilbert, level


def getCellsForEnvelope(env_deg, maxLevel):
    """Find the 4 smallest cells that completely
    cover an envelope. All pings inside the envelope have the first `level`
    bits of their hilbert number agree with one of the hilbert numbers
    returned by this function.
    """
    lng1, lng2, lat1, lat2 = env_deg

    corners = [ [lng1, lat1],
                [lng2, lat1],
                [lng2, lat2],
                [lng1, lat2]
             ]
    corners = np.array(corners)

    cells = lnglat2cell(corners[:,0], corners[:,1], maxLevel)
    cells, level = findAdjacentParentCells(cells, maxLevel)
    return cells, level


def findAdjacentParentCells(cells, n):
    """Find the parents of the input cells that are adjacent.

    This is used to find the 4 cells that completely cover a geometry.

    Two cells are adjacent if they share a common side and two corners.
    Cells that touch at only one point are not adjacent.

    At level n=1 there are only 4 cells and each cell is adjacent to
    two others. Therefore, for any input set of 4 cells there is always
    a parent level where every parent is adjacent to 2 other parents.

    Inputs
    ---------
    cells
        (4x2 np integer array ) Each row represents a single cell.
        The zeroth column represents the i coordinate, the first the j
        coord. It is assumed that the parents of cell[0] and cell[2] are
        never adjacent, nor are the parents of cell[1] and cell[3]

    n
        (int) The highest level at which the cells might possibly be
        adjacent. The smaller this number the faster the algorithm runs,
        but you run the risk of finding larger cells than are strictly
        necessary to cover the input geometry.

    Returns
    ---------
    An array of 4 cells, and the level of those cells.
    """

    assert cells.ndim == 2 and cells.shape == (4,2)

    while not ( cellsAreAdjacent(cells[0], cells[1], n) and \
                cellsAreAdjacent(cells[0], cells[3], n) and \
                cellsAreAdjacent(cells[1], cells[2], n)
              ):
        cells = cells >> 1  #Integer-divide by two
        n -= 1

        #This should never happend
        assert n > 0
    return cells, n


def cellsAreAdjacent(cell1, cell2, level):
    """Are two cells side by side?

    Assumes cells are at the same level. Results are meaningless
    if this is not true

    Cells are not adjacent if they touch at only one point
    A cell *is* adjacent to itself.

    Note
    --------
    The level argument is not used. It's just there to remind you that
    both cells must be at the same level

    TODO. This is probaly ripe for optimisation. Removing the if statements
    would help
    """

    # debug()
    i1, j1 = cell1
    i2, j2 = cell2

    if i1 == i2 and j1 == j2:
        return True

    #Because cell indices are unsigned, you can't blindly check i2-i1 == ...
    #Casting feels like a slow operation.
    if i2 > i1:
        horz = i2 - i1 == 1
    else:
        horz = i1 - i2 == 1

    if j2 > j1:
        vert  = j2 - j1 == 1
    else:
        vert = j1 - j2 == 1

    return horz ^ vert

# def lnglat2d(lng_deg, lat_deg, n=25):
#     """
#     Default value of n makes squares of 2.4m on a side at
#     the equator
#     """

#     lng_deg = (lng_deg + 180) / 360.
#     lat_deg = (lat_deg + 90) / 180

#     return normfloat2d(lng_deg, lat_deg, n)

# def d2lnglat(d, n=25):
#     x, y = d2normfloat(d, n)
#     lng_deg = (x * 360) - 180
#     lat_deg = (y * 180) - 90

#     return lng_deg, lat_deg

# def normfloat2d(x, y, n):
#     assert np.all(x >=0) and np.all(x <= 1)
#     assert np.all(y >=0) and np.all(y <= 1)

#     scale = 2**n
#     x = (x * scale).astype(np.uint32)
#     y = (y * scale).astype(np.uint32)

#     return xy2d(x, y, n)

# def d2normfloat(d, n):
#     x, y = d2xy(d, n)

#     scale = float(2**n)
#     x /= scale
#     y /= scale
#     return x, y


# #convert (x,y) to d
# def xy2d (x, y, n):
#     """
#     Inputs
#     -----
#     x, y
#         (np integer arrays) 0 <= x, y <= 2**n
#     """
#     max_val = 2**n
#     assert np.all(x >=0) and np.all(x <= max_val)
#     assert np.all(y >=0) and np.all(y <= max_val)

#     d = 0
#     #Check n is a power of 2
#     sarr = 2**np.arange(n-1, -1, -1)

#     for s in sarr:
#         rx = (x & s) > 0
#         ry = (y & s) > 0
#         d += s * s * ((3 * rx) ^ ry);
#         x, y = rot(n, x, y, rx, ry);
#     return d


# def d2xy(d, n):
#     t = d
#     sarr = 2**np.arange(n-1, -1, -1)

#     x = np.zeros_like(d)
#     y = np.zeros_like(d)

#     for s in sarr:
#         rx = 1 & (t // 2)
#         ry = 1 & (t ^ rx)
#         x, y = rot(s, x, y, rx, ry)
#         x += s * rx
#         y += s * ry
#         t /= 4

#     return x, y
# """
# //convert d to (x,y)
# void d2xy(int n, int d, int *x, int *y) {
#     int rx, ry, s, t=d;
#     *x = *y = 0;
#     for (s=1; s<n; s*=2) {
#         rx = 1 & (t/2);
#         ry = 1 & (t ^ rx);
#         rot(s, x, y, rx, ry);
#         *x += s * rx;
#         *y += s * ry;
#         t /= 4;
#     }
# }
# """

# #rotate/flip a quadrant appropriately

# //rotate/flip a quadrant appropriately
# void rot(int n, int *x, int *y, int rx, int ry) {
#     if (ry == 0) {
#         if (rx == 1) {
#             *x = n-1 - *x;
#             *y = n-1 - *y;
#         }

#         //Swap x and y
#         int t  = *x;
#         *x = *y;
#         *y = t;
#     }
# }
# """