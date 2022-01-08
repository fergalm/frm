"""
Created on Thu Jan 19 11:13:04 2017

@author: fergal
"""
#from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


from frm.anygeom import AnyGeom
#import frm.geomconvertor as gc
from frm.support import npmap
import frm.projection
import frm.plots as fplots

"""A tool to format AOIs to meet the requirements of Digital Globe
and other providers
"""

def format_for_digital_globe(geoms, plot=False):
    """
    Take a colleciton of AOIs, and format them for a Digital Globe Order

    Inputs
    ---------
    geoms
        A list of ``frm.AnyGeom`` objects. Each object must have an attribute
        called ``name``

    Optional Inputs
    ---------------
    plots
        If **True**, Plot up the input and output AOIs
    """


    if plot:
        map( lambda x: fplots.plot_shape(x.as_geometry(), 'k-'), geoms)

    boxes  = npmap( create_buffered_bbox_for_dg, geoms)
    if plot:
        map( lambda x: fplots.plot_shape(x.as_geometry(), 'b-'), boxes)

    merged = merge_overlapping_boxes(boxes, plot=plot)
    return merged


def merge_overlapping_boxes(boxes, plot=False):
    """

    Inputs
    ------------
    boxes
        List of gc.GeomObject objects. Each object must have the metadata
        **name**


    Returns
    ------------
    Same.     Overlapping geometries are merged.
    The metadata value **area** equal to the sum of the areas of
    the merged boxes is added to each object.
    """

    num = len(boxes)
    keep = np.ones(num, dtype=bool)

    names = npmap( lambda x: x.meta['name'], boxes)
    area = npmap( lambda x: x.meta['area'], boxes)

    gdals = npmap( lambda x: x.as_geometry(), boxes)
    contains = names.copy().astype(object)

    for g in gdals:
        assert g is not None

    repeat = True
    while repeat:
        repeat = False
        for i in range(num):
            if keep[i] == False:
                continue

            for j in range(i+1, num):
                if keep[j] == False:
                    continue

                if gdals[i].Intersects(gdals[j]):
                    print("Merging box %i (%s)" %(j, names[j]))
                    gdals[i] = merge(gdals[i], gdals[j])
                    area[i] = area[i] + area[j]
                    contains[i] = "%s %s" %(contains[i], contains[j])

                    keep[j] = False
                    gdals[j] = None
                    area[j] = -1

                    repeat = True

    names = np.array(names)[keep]
    gdals = np.array(gdals)[keep]
    contains = contains[keep]
    area = area[keep]

    if plot:
        map( lambda x: fplots.plot_shape(x, 'r--'), gdals)

    boxes = npmap(AnyGeom, gdals)
    for i in range(len(boxes)):
        boxes[i].add_metadata(name=names[i], area=area[i], contains=contains[i])

    return boxes


def merge(gdal1, gdal2):
    """Merge two gdal geometries.

    Inputs
    -----------
    gdal1, gdal1
        ogr.Geometry objects

    Returns
    ----------
    ogr.Geometry object. New object is a bounding box encompassing both
    input bounding boxes
    """
    a1, a2, b1, b2 = gdal1.GetEnvelope()
    c1, c2, d1, d2 = gdal2.GetEnvelope()

    x1 = min(a1, c1)
    x2 = max(a2, c2)
    y1 = min(b1, d1)
    y2 = max(b2, d2)

    bbox = np.array([ [x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1] ])
    g = AnyGeom(bbox, "Polygon").as_geometry()
    return g



def create_buffered_bbox_for_dg(geom):
    """
    Input
    ---------
    geom
        An ``AnyGeom()`` object. The attribute ``name`` must exist

    Returns
    ---------
    Same
    """

    #Min and max ranges from
    #https://orbitalinsight.atlassian.net/wiki/display/PO/Ordering+Imagery
    min_area_sqm = 5e6
    max_area_sqm = 1e9
    min_side_m = 1000.   #Each side must be at least 1 km

    Rearth = 6378137    #Radius of earth in metres
    min_side_deg = 180 * min_side_m / (np.pi * Rearth)

#    import pdb; pdb.set_trace()
    #Extract coords of bounding box
#    import pdb; pdb.set_trace()
    gdal = geom.as_geometry()
    assert gdal is not None, geom.meta['name']
    name = geom.meta['name']

    x1, x2, y1, y2 = gdal.GetEnvelope()
    yc = .5 * (y1 + y2)

    #Ensure width is at least 1 km
    dx = x2-x1
    if dx < min_side_deg * np.cos( np.radians(yc) ):
        xc = .5 * (x1+x2)
        x1 = xc - .5 * min_side_deg
        x2 = xc + .5 * min_side_deg

    #Ensure height is at least 1km
    dy = y2 - y1
    if dy < min_side_deg:

        y1 = yc - .5 * min_side_deg
        y2 = yc + .5 * min_side_deg


    bbox = np.array([ [x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1] ])

    #Convert to equal area projection
    proj = frm.projection.SineProjection(Rearth)
    ea_coords = np.array( proj.w2p(bbox[:,0], bbox[:,1]) ).transpose()
    area = area_of_polygon(ea_coords)

    if area > max_area_sqm:
        raise ValueError("BBox is larger than max area")

    if area > min_area_sqm:
        ratio = 1.
    else:
        #Scale coords. ratio chosen so area of new shape meets requirements
        ratio = 1.05 * np.sqrt( min_area_sqm / area)

    scaled_coords = scale_shape(ea_coords, ratio)

    #Convert back to lng/lat coords
    new_lng, new_lat = proj.p2w(scaled_coords[:,0], scaled_coords[:,1])
    new_coords = np.array([new_lat, new_lng]).transpose()

    #Convert back to an OGR
    new_geom = AnyGeom(new_coords, "Polygon", {'name':name})
    new_geom.add_metadata(area=area_of_polygon(scaled_coords))
    return new_geom



def scale_shape(coords, ratio):
    """Scale shape so distance between points is (1+r) times bigger than it was"""

    x0 = np.mean(coords[:-1, 0])
    y0 = np.mean(coords[:-1, 1])

    new_coords = coords.copy()
    #Translate to origin
    new_coords[:,0] -= x0
    new_coords[:,1] -= y0

#    print x0, y0
#    print new_coords
    #Scale
    new_coords *= ratio

    #Translate back to orig pos
    new_coords[:,0] += x0
    new_coords[:,1] += y0

    return new_coords


def simple_area(lnglat):
    dlng, dlat = np.max(lnglat, axis=0) - np.min(lnglat, axis=0)
    lat0 = np.mean(lnglat[:-1, 1])

    area = 111**2 * dlat * dlng * np.cos( np.radians(lat0) )
    return area


def area_of_polygon(data):
    """
    Compute area of polygon assuming Euclidean geometry. Do not
    use this on lng/lat pairs

    Inputs
    ----------
    data
        (numpy 2d array) 2 columns (x points and y points)
        data[-1] == data[0].


    Returns
    ----------
    float
    """

    assert np.all(np.fabs(data[-1] - data[0]) < 1e-6), data

    x = data[:,0]
    y = data[:,1]

    val = x * np.roll(y, -1) - np.roll(x,-1) * y
    return .5 * np.fabs( np.sum(val[:-1]) )



def test_area_of_polygon():

    poly = np.array([ [0,0], [0,1], [1,1], [1,0], [0,0] ])
    assert area_of_polygon(poly) == 1

    poly = np.array([ [0,0], [0,2], [2,2], [2,0], [0,0] ])
    assert area_of_polygon(poly) == 4

    poly = np.array([ [0,0], [0,2], [1,2], [1,0], [0,0] ])
    assert area_of_polygon(poly) == 2


    poly = np.array([ [0,0], [2,0], [2,1], [0,1], [0,0] ])
    assert area_of_polygon(poly) == 2

    #Test offset from origin
    poly = np.array([ [10,10], [10,12], [12,12], [12,10], [10,10] ])
    assert area_of_polygon(poly) == 4

    #Test triangle
    poly = np.array([ [10,10], [10,14], [14,10], [10,10] ])
    assert area_of_polygon(poly) == 8
#