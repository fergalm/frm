"""
Created on Fri Mar 31 13:52:57 2017

@author: fergal
"""
from __future__ import print_function
from __future__ import division


import shapely.wkt
import shapely
import numpy as np
import json
import osgeo.ogr as ogr

class ConversionError(Exception):
    """Thrown when trying to convert the wrong kind of geometry"""
    pass



class AnyGeom(object):
    """A class to convert between the bewildering variety of shape types with a uniform interface

    Examples
    -----------
    ::

        geom1 = Anygeom( gdal_object )
        geom2 = Anygeom( wkt_string )

        geom1.as_wkt()
        geom2.as_kml()

    Regardless of the input and output types, the interface is the same.
    This simplifies code resuse where the only thing that changes in
    the source of the input.

    This table summarises the types of shape objects that are accepted and/or
    generated. For some types (like matplotlib patches) only one way conversions
    make sense (you never want to convert a patch into a wkt, for example.

    ======== ======== ========
    Type     Accepted Returned
    ======== ======== ========
    ogr      yes       yes
    wkt      yes       yes
    numpy    yes       yes
    geojson  yes       yes
    shapely  yes       yes
    pyshp    yes        no
    patch    no        yes
    kml      no        yes
    wkb      no        no
    tiles    no        yes
    ======== ======== ========

    Notes
    ------

    * A patch is a matplotlib.patch object useful for creating cloropleth plots
    * pyshp is a tool for reading ESRI shapefiles. It doesn't have a stand alone
      class to describe shapes that I can instantiate
    """

    def __init__(self, geom, gtype=None, metadata=None):
        """
        Inputs
        --------
        geom
            Object to convert. Can be any accepted type. Refer to optional inputs
            below if input data type is a numpy array


        Optional Inputs
        ----------------
        gtype
            (string) If ``geom`` is a numpy array, use this variable to specify shape type,
            eg. point or line. gtype is a WKT keyword, e.g POINT, LINESTRING etc.

        metadata
            (dict) Dictionary of metadata associated with a geometry. No checks are made
            that the keys of values of this dictionary can be converted into the
            requested data type later.
        """
        #Each function converts a single type of shape
        convlist = [conv_anygeom,
                    conv_gdal,
                    conv_wkt,
                    conv_points,
                    conv_geojson,
                    conv_pyshp,
                    conv_shapely,]

        if geom is None:
            return None

        if metadata is None:
            metadata = dict()

        self.obj = None
        for conv in convlist:
            try:
                self.obj, self.meta  = conv(geom, gtype, metadata)
                break
            except ConversionError:
                pass

        if self.obj is None:
            raise ConversionError("Input was not of recognised type")


        self.isAnyGeom = True


    def __str__(self):
        return self.as_wkt()

    def __repr__(self):
        geom_type, data = self.as_array()

        name = "None"
        for k in "name Name NAME".split():
            if k in self.meta.keys():
                name = self.meta[k]

        strr  = "<AnyGeom Name: '%s' of type %s (%i elements)>" %(name, geom_type, len(data))
        return strr


    def add_metadata(self, **kwargs):
        for k in kwargs.keys():
            self.meta[k] = kwargs[k]


    def as_geometry(self):
        """Return a GDAL/OGR object"""
        return self.obj


    def as_wkt(self):
        return self.obj.ExportToWkt()


    def as_kml(self):
        text = ["<Placemark>"]
        meta = self.meta
        if "name" in meta.keys():
            text.append("  <name>%s</name>" %(meta['name']))

        if len(meta) > 0:
            text.append("  <ExtendedData>")
            for k in meta.keys():
                if k == "name":
                    continue
                text.append('    <Data name="%s"><value>%s</value></Data>' %(k, str(meta[k])))
            text.append("  </ExtendedData>")

        text.append(self.obj.ExportToKML())
        text.append("</Placemark>")
        return "\n".join(text)

    def as_json(self):
        return self.obj.ExportToJson()


    def as_array(self):
        """Return object as a (possibly nested list) of numpy arrays"""

        geom_type = self.obj.GetGeometryName()
        geom_count = self.obj.GetGeometryCount()

        if geom_type in 'MULTIPOINT GEOMETRYCOLLECTION'.split() and geom_count == 0:
            return geom_type, np.atleast_2d(np.array([]))

        data = ogrGeometryToArray(self.obj)

        #Convenience. ogrGeometryToArray sometimes returns things
        #in a shape that makes sense, but it'd be easier if it just
        #gave us a 2d array.
        if geom_type in ['POINT', 'POLYGON'] and len(data) == 1:
            data = data[0]

        if geom_type == 'MULTIPOINT':
            data = np.array(data)[:, 0, :]

        return geom_type, data

    def as_patch(self, **kwargs):
        """Convert to a matplotlib patch for drawing"""

        try:
            import matplotlib.patches as mpatch
        except ImportError:
            raise ImportError("Failed to import matplotlib.patches")

        def foo(elt, **kwargs):
            patches = []
            if hasattr(elt, 'ndim'):
                #elt = np.atleast_2d(elt)
                assert elt.ndim == 2
                patches.append(mpatch.Polygon(elt[:,:2], closed=True, **kwargs))
            else:
                for e in elt:
                    patches.extend(foo(e, **kwargs))
            return patches

        xy_list = ogrGeometryToArray(self.obj)
        patches = foo(xy_list, **kwargs)


        # for elt in xy_list:
        #     if hasattr(elt, 'ndim'):
        #         assert elt.ndim == 2
        #         patches.append(mpatch.Polygon(elt, closed=True, **kwargs))
        #     else:
        #         outer, inner = elt
        #         patches.append(mpatch.Polygon(outer, closed=True, **kwargs))
        #         patches.append(mpatch.Polygon(inner, closed=True, **kwargs))

        return patches


    def as_shapely(self):
        return shapely.wkt.loads( self.obj.ExportToWkt() )



#########################################################################
def conv_anygeom(geom, gtype, metadata):
    if hasattr(geom, 'isAnyGeom'):
        return geom.as_geometry(), geom.meta


    raise ConversionError("Not an AnyGeom")


def conv_gdal(geom, gtype, metadata):
    if isinstance(geom, ogr.Geometry):
        return geom, metadata

    raise ConversionError("Not a gdal")


def conv_geojson(geom, gtype, metadata):
    #Call to gdal can fail in tow ways, by throwing exception or returning None
    try:
        obj = ogr.CreateGeometryFromJson(geom)
    except TypeError:
        obj = None

    if obj is None:
        raise ConversionError("Not a Wkt or not a valid Wkt")

    return obj, metadata


def conv_points(geom, gtype, metadata):

    if gtype is None:
        raise ConversionError("Sets of points require a gtype")

    legal_types = [list, tuple, np.ndarray]
    if type(geom) not in legal_types:
        raise ConversionError("Not a set of points")

    geom = np.array(geom)
    gtype = gtype.lower()
    if gtype == 'point':
        obj = ogr.Geometry(ogr.wkbPoint)
#        import pdb; pdb.set_trace()
        p = list(geom)
        obj.AddPoint(*p)

    elif gtype == 'mpoint':
        obj = ogr.Geometry(ogr.wkbMultiPoint)
        for point in geom:
            coords = list(point)
            pt = ogr.Geometry(ogr.wkbPoint)
            pt.AddPoint(*coords)
            obj.AddGeometry(pt)

    elif gtype == 'linestring':
        obj = ogr.Geometry(ogr.wkbLineString)
        for point in geom:
            p = list(point)
            obj.AddPoint(*p)

    elif gtype == 'polygon':
        obj = createPolygonFromPoints(geom)

    elif gtype == 'multipolygon':
        obj = ogr.Geometry(ogr.wkbMultiPolygon)
        for poly in geom:
            obj.AddGeometry(createPolygonFromPoints(poly))
    else:
        raise ValueError("Unrecognised value for gtype")

    return obj, metadata


def createPolygonFromPoints(geom):
    obj = ogr.Geometry(ogr.wkbPolygon)

    try:
        #Turn an array into a list with one array in it
        if geom.ndim == 2:
            geom = [geom]
    except AttributeError:
        pass  #Input is a list

    for sub_geom in geom:
        ring = ogr.Geometry( ogr.wkbLinearRing)
        for point in sub_geom:
            p = list(point)
            ring.AddPoint(*p)
        obj.AddGeometry(ring)

    return obj


def conv_wkt(geom, gtype, metadata):
    try:
        obj = ogr.CreateGeometryFromWkt(geom)
    except TypeError:
        obj = None

    if obj is None:
        raise ConversionError("Not a Wkt or not a valid Wkt")

    return obj, metadata


def conv_shapely(geom, gtype, metadata):
    try:
        wkt = shapely.wkt.dumps(geom)
    except (TypeError, AttributeError):
        raise ConversionError("Not a shapely object")

    return conv_wkt(wkt, gtype, metadata)


def conv_pyshp(geom, gtype, metadata):
    try:
        jsonDict = geom.__geo_interface__
    except (TypeError, AttributeError):
        raise ConversionError("Object has no geo_interface. Is not a pyshp")

    try:
        metadata = jsonDict['properties']
    except KeyError:
        metadata = {}

    text = json.dumps(jsonDict)
    obj = ogr.CreateGeometryFromJson(text)
    return obj, metadata


def ogrGeometryToArray(geometry):
    """Convert an OGR geometry object to a 2d numpy array

    The array is of shape nx2, where n is the number of points in the shape.

    This works as expected for Polygons, but does something slightly
    unexpected for multi-polygons. For a multi-polygon it returns
    a list. Each element of the list is either a 2d numpy array or
    another list. By checking each list element recursively you can
    finally arrive at a set or numpy arrays, each one repesenting a
    single polygon.

    See plot_shape
    """

    if geometry.GetGeometryCount() > 0:
        elts = []
        # import ipdb; ipdb.set_trace()
        for i in range(geometry.GetGeometryCount()):
            elts.append( ogrGeometryToArray(geometry.GetGeometryRef(i)) )

        return elts

    return  np.atleast_2d(geometry.GetPoints())






#
