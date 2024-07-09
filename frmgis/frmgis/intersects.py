"""
Code to quickly identify which input points are inside a polygon.
This a reimplemntation of ogrs' Polygon.Contains() method in numpy.

See intersects.md in the source code for implementation details.
"""

import numpy as np


def select_points_in_geom(geom, lng, lat, dtype=np.float32):
    """Find points interior to a geometry.

    Args:
        geom: (ogr Polygon, Multipolygon, or GeometryCollection)
        x0, y0: (1d np arrays) x and y coordinates (or lng/lats) of points to test
        dtype: datatype to use internally). Use np.float64 to get exactly the
            same results as GDAL's Polygon.Contains(). Use np.float32 to
            reduce the run time by a factor of two at the expense of some small
            number of points at the edge of the polygon being mis-identified
            due to rounding error.

    Returns
    ------------
    A 1d numpy array of booleans. True means the point is inside
    the envelope.
    """

    name = geom.GetGeometryName()
    if name == "POLYGON":
        return select_points_in_polygon(geom, lng, lat, dtype)
    elif name in "MULTIPOLYGON GEOMETRYCOLLECTION".split():
        idx = np.zeros(len(lng), dtype=bool)

        for i in range(geom.GetGeometryCount()):
            poly = geom.GetGeometryRef(i)
            idx |= select_points_in_polygon(poly, lng, lat, dtype)

        return idx
    else:
        raise ValueError("Geometry type %s not recognised as a area geometrty" % (name))


def select_points_in_polygon(geom, lng, lat, dtype):
    """Select points in a single polygon, possibly with holes"""
    name = geom.GetGeometryName()
    assert name == "POLYGON"

    nElt = geom.GetGeometryCount()
    assert nElt > 0
    ring = geom.GetGeometryRef(0)

    idx = select_points_in_simple_poly(ring, lng, lat, dtype)

    for i in range(1, nElt):
        hole = geom.GetGeometryRef(i)
        idx &= ~select_points_in_simple_poly(hole, lng, lat)
    return idx


def select_points_in_simple_poly(geom, x0, y0, dtype=np.float32):
    """Find points interior to a simple polygon

    The implementation of this function is described in more
    detail in `intersects.md` in this directory

    """
    len_in = len(x0)
    x0 = x0.astype(dtype)
    y0 = y0.astype(dtype)

    assert geom.GetGeometryName() == "LINEARRING"
    edges = get_edges(geom).astype(dtype)

    in_env = filter_by_env(geom, x0, y0)
    x0 = np.atleast_2d(x0[in_env]).transpose()
    y0 = np.atleast_2d(y0[in_env]).transpose()

    # Work around. Memory requirements of algorithm as described
    # scale as (num points) * (num edges). For very complicated geoms
    # this can dominate over other memory usages. We restrict the
    # max number of points processed at a time.
    # This limit sacrifices a bit of speed for improved stability.
    num_points = len(x0)
    max_elts = 8_000_000
    num_elts = num_points * len(edges)
    num_step = int(np.ceil(num_elts / (max_elts)))

    # Compute step size, guarding against edge case of zero points to process
    eps = 1e-10  # Prevent division by zero
    step_size = int(np.ceil(num_points / (num_step + eps)))
    step_size = max(step_size, 1)

    # print("%i points, %i edges, %i chunks, stepsize is %i" %(num_points, len(edges), num_step, step_size))
    isInside = np.zeros(len(x0), dtype=bool)
    for i in range(0, num_points, step_size):
        upr = min(i + step_size, num_points)
        chunk_x = x0[i:upr]
        chunk_y = y0[i:upr]

        dy = edges[:, 3] - edges[:, 1]
        dx = edges[:, 2] - edges[:, 0]
        x1 = edges[:, 0] - chunk_x
        y1 = edges[:, 1] - chunk_y
        y2 = edges[:, 3] - chunk_y

        # Which line segments straddle the point?
        delta = (y1 > 0) ^ (y2 > 0)
        # Which edges are to the right of the test point
        alpha = (dx * y1) > (dy * x1)
        alpha ^= dy < 0

        nCrossing = (delta & alpha).sum(axis=1)
        isInside[i:upr] = np.fmod(nCrossing, 2) == 1

    outidx = np.zeros(len_in, dtype=bool)
    outidx[in_env] = isInside
    return outidx


def get_edges(geom):
    """Return the lines that connect the vertices of a simple polygon

    Args:
        geom (ogr Geometry): A geometry representing a single simple polygon

    Returns
    ----------
    An (n x 4) numpy array represnting the start and end points of the
    lines connecting those vertices in the form (x1, y1, x2, y2).
    Assumes that consectutive points in the input are connected to each
    other, and to no others.
    """
    vertices = geom.GetPoints()

    # Last point repeats the first. Remove the 3rd col, if it exists
    vertices = np.atleast_2d(vertices)[:-1, :2]
    num = len(vertices)

    out = np.zeros((num, 4))
    out[:, :2] = vertices
    out[:, 2:] = np.roll(vertices, -1, axis=0)
    return out


def filter_by_env(geom, lng, lat):
    """Identify points interior to the envelope of a geometry.

    This routine is very fast way to filter points that are definitely
    not in the geometry, allowing slower, but more accurate, methods
    to focus on a subset of the points

    Args:
        geom (ogr Geometry): A polygon or multi-polygon
        lng (1d np array): Longitudes of points to check
        lat (1d np array): Longitudes of points to check


    Returns:
        A 1d numpy array of booleans. True means the point is inside
        the envelope.
    """
    envelope = geom.GetEnvelope()

    # Identify objects outside the envelope
    lng1, lng2, lat1, lat2 = envelope
    idx = (lng1 <= lng) & (lng <= lng2)
    idx &= (lat1 <= lat) & (lat <= lat2)
    return idx
