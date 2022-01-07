# -*- coding: utf-8 -*-

"""
Tools for interacting with geolocation data.
For tools to plot geolocation data see geolocplot.py
@author: fergal
"""

from __future__ import print_function
from __future__ import division

from ipdb import set_trace as idebug

import oids.geoloc.utils.geohash as oi_geohash
from frm.anygeom import AnyGeom
from osgeo import ogr
import pandas as pd
import numpy as np
import frm.support


npmap = frm.support.npmap

def centroid(shape):
    """Compute the centroids of a single shape
    
    Args:
        shape: Any object accepted by AnyGeom, eg, wkt
        
    Returns:
        two floats, the longitude and latitude of the centroid
        of the input shape, in the coordinate system used by the 
        shape. Typically this means longitude and latitude in degrees.
    """
    
    geom = AnyGeom(shape).as_geometry()
    centroid = geom.Centroid()
    lng = centroid.GetX()
    lat = centroid.GetY()
    return lng, lat


def convert_adid_to_deviceid(ad_ids):
        """Convert ad_ids from Max's unsigned ints to Alex's strings

        This is useful to compare silo inputs/outputs with Alex's client
        """
        def f(x):
            return x.to_bytes(8, 'big', signed=True).hex().upper()

        return ad_ids.apply(f)


def geohashes_to_lnglat(ghlist):
    """Convert a list of geohashes (of any length) to lnglat positions

    Inputs
    ---------
    ghlist
        (list of strings) Input geohashes


    Returns
    ---------
    Numpy array of shape (n, 2), where *n* is the length of the input.
    Each row represents the lng/lat of the southwest corner of a geohash
    """

    return npmap(lambda x: oi_geohash.geohash_bounds(x)['sw'], ghlist)


def deg_to_metres(degrees):
    """Convert degrees in arc to metres on the surface of the Earth

    This works for degrees in longitude, or degrees along a great circle
    but not degrees of latitude. Distance between two degrees of lat
    is approximately deg_to_metres * np.cos(lat)

    This function is for approximate measure only. Don't use for
    calculations on the scale of a continent where accuracies of <1%
    are required.
    """

    earth_radius_m = 6378100.0  #Equatorial radius
    return degrees * np.pi * earth_radius_m / 180


def metres_to_deg(metres):
    """Convert distances in metres on the surface of the Earth to degrees of
    arc.

    Degrees of arc are the same as degrees along a line of longitude, but
    not the same as degrees along a line of latitude.

    This function is for approximate measure only. Don't use for
    calculations on the scale of a continent where accuracies of <1%
    are required.
    """

    earth_radius_m = 6378100.0  #Equatorial radius
    return 180 * metres / (np.pi * earth_radius_m)


def get_pings_in_shape(df, shape):
    """Filter a dataframe of pings by a geometry

    If you have a dataframe of pings from a large AOI, this function
    lets you filter that list down to only the pings inside a smaller
    AOI.

    It does so in an efficient manner by first filtering pings outside
    the bounding box of the AOI, in a manner similar to rtrees.

    Inputs
    ----------
    df
        (Dataframe) Ping data.
        Expected columns are
            longitude
                Longitude of ping
            latitude
                Latitude of ping

    shape
        Any geometry type acceptable to AnyGeom


    Returns a dataframe
    """
    assert frm.support.check_columns_in_df(df, 'lng_deg lat_deg'.split())
    geom = AnyGeom(shape).as_geometry()

    lng = df.lng_deg.values
    lat = df.lat_deg.values
    idx = is_inside(lng, lat, geom)
    return df[idx]


def is_inside(lng_arr, lat_arr, shape):
    """
    Compute which lng,lat pairs are inside shape

    Somewhat optimized for speed by checking points are inside
    and envelope before computing the intersection with the geometry
    This is a candidate for speeding up

    Inputs
    -------
    lng_arr, lat_arr
        (1d np array) Longitudes and latitudes of points
    shape
        Geometry to intersect with. Can be any type accepted by
        AnyGeom

    Returns
    ----------
    1d numpy boolean array. Points inside the geometry
    """
    assert len(lng_arr) == len(lat_arr)

    try:
        lng_arr = lng_arr.values
        lat_arr = lat_arr.values
    except:
        pass

    #Identify objects outside the envelope
    geom = AnyGeom(shape).as_geometry()
    envelope = geom.GetEnvelope()
    lng1, lng2, lat1, lat2 = envelope
    idx = (lng1 <= lng_arr) & (lng_arr <= lng2)
    idx &= (lat1 <= lat_arr) & (lat_arr <= lat2)

    #Only do a full intersection search for objects inside
    #the envelope
    for i in range(len(idx)):
        if not idx[i]:
            #Point is outside the envelope. Don't bother checking it
            continue

        point = ogr.Geometry( ogr.wkbPoint)
        point.AddPoint(lng_arr[i], lat_arr[i])
        idx[i] = geom.Intersects(point)

    return idx



#Deprecated 2020-01-21
def quickcull(df, geom, lngKey='longitude', latKey='latitude'):
    """Eliminate points not even close to the polygon in geom"""
    DeprecationWarning(("Use is_inside() instead"))
    envelope = geom.GetEnvelope()
    lng1, lng2, lat1, lat2 = envelope
    print(envelope)

    lng = df[lngKey]
    lat = df[latKey]

    idx = (lng1 < lng) & (lng < lng2) & \
          (lat1 < lat) & (lat < lat2)

    return df[idx]


#Deprecated 2020-01-21
def isInside(lnglat, shape):
    """Return true if lnglat is inside the polygon ``shape``

    Inputs
    -------
    lnglat
        (2-tuple, list, array of floats)  longitude and latitude
    shape
        (ogr Geometry)

    Returns
    ----------
    **bool**

    TODO
    -----
    Is this faster than making a multipoint geometry?
    """
    DeprecationWarning(("Use is_inside() instead"))

    point = ogr.Geometry( ogr.wkbPoint)
    pp = list(lnglat)
    point.AddPoint(*pp)
    return shape.Intersects(point)


def create_udc(df, interval, tz='UTC', tkey='unixtime', dkey='device_id'):
    """Create a time-series of unique device counts

    Inputs
    -------
    df
        (Dataframe) Input data
    interval
        (str)  What interval to bin data at (e.g 'D', '4H' etc.)

    Optional Inputs
    ----------------
    tz
        (str) Local timezone.
    tkey
        (str) Column name for time column. Col must be a datetime type
    dkey
        (str) column name for unique device id.

    Returns
    ----------
    A dataframe. Index is start time for each interval. The single column
    is the unique device count for that interval
    """
    df = df.copy()
    df.index = pd.to_datetime(df[tkey], unit='s')
    df.index =  df.index.tz_localize('UTC').tz_convert(tz)

    udc = df.resample(interval)[dkey].nunique()
    return udc
