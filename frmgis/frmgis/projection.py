"""
Created on Mon Dec 12 14:36:07 2016

@author: fergal
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
#import pandas as pd
import numpy as np


#Note: all inputs are in degrees

class AbstractProjection(object):

    def __call__(self, *args):
        return self.world_to_pixel(*args)

    def w2p(self, *args):
        return self.world_to_pixel(*args)


    def p2w(self, *args):
        return self.pixel_to_world(*args)


    def world_to_pixel(self, lng, lat):
        raise NotImplementedError("Do not call abstract class directly")


    def pixel_to_world(self, col, row):
        raise NotImplementedError("Do not call abstract class directly")


    def parse_inputs(self, arg1, arg2):
        """Convert inputs into a 2d array.

        This function accounts for args being numbers or iterables

        """
        try:
            if len(arg1) != len(arg2):
                raise ValueError("Inputs must have same length")
        except TypeError:
            #One of the inputs is not an iterable
            pass

        return np.array([arg1, arg2]).reshape(2,-1)


    def parse_output(self, output):
        """Returns two things, either two arrays, or two values"""
        assert output.ndim == 2
        output = np.array(output)

        if len(output[0]) == 1:
            return output[0][0], output[1][0]
        return output[0], output[1]


class PlateCaree(AbstractProjection):
    def __init__(self):
        AbstractProjection.__init__(self)


    def world_to_pixel(self, lng, lat):
        return lng, lat

    def pixel_to_world(self, col, row):
        return col, row


class LinearProjection(AbstractProjection):
    def __init__(self, lng0, lat0, degrees_per_pixel_lng, degrees_per_pixel_lat=None):
        AbstractProjection.__init__(self)

        if degrees_per_pixel_lat is None:
            degrees_per_pixel_lat = degrees_per_pixel_lng

        mat = np.array( [1/degrees_per_pixel_lng, 0, 0, 1/degrees_per_pixel_lat])

        self.M = np.matrix( mat.reshape(2,2) )
        self.Minv = self.M.I
        self.origin = np.array([lng0, lat0])


    def world_to_pixel(self, lng, lat):
        pos = self.parse_inputs(lng, lat)
        lng0, lat0 = self.origin

        pos[0, :] -=  lng0
        pos[1, :] -=  lat0

        pixels = np.dot(self.M, pos)
        return self.parse_output(pixels)


    def pixel_to_world(self, col, row):
        pos = self.parse_inputs(col, row)
        lng0, lat0 = self.origin

        rel_lnglat = np.dot(self.Minv, pos)
        lng, lat = self.parse_output(rel_lnglat)

        return lng + lng0, lat + lat0


class SineProjection(AbstractProjection):
    """Sine Equal Area projection

    See: https://en.wikipedia.org/wiki/Sinusoidal_projection
    """

    def __init__(self, radius=1):
        AbstractProjection.__init__(self)
        self.radius = float(radius)

    def world_to_pixel(self, lng, lat):
#        import pdb; pdb.set_trace()
        pos = self.parse_inputs(lng, lat)

        R = np.pi * self.radius / 180.
        col = R * np.cos( np.radians(pos[1, :]) ) * pos[0, :]
        row = R * pos[1,:]
        return col, row

    def pixel_to_world(self, col, row):
        pos = self.parse_inputs(col, row)

        R = np.pi * self.radius / 180.
        lat = pos[1,:] / R
        lng = pos[0,:] / (R * np.cos( np.radians(lat) ))
        return lat, lng


class WebMercator(AbstractProjection):
    def __init__(self, lng_left, lat_top, zoom):
        AbstractProjection.__init__(self)

        self.set_params(lng_left, lat_top, zoom)

    def set_params(self, lng_left, lat_top, zoom):
        self.zoom = zoom

        #Compute col row of "top left" of image
        pos = self.parse_inputs(lng_left, lat_top)
        self.col0, self.row0 = self.absolute_w2p(pos)

    def set_params_from_image(self, lng_centre, lat_centre, size):
        assert len(size) == 2

    def world_to_pixel(self, lng, lat):
        pos = self.parse_inputs(lng, lat)

        col, row = self.absolute_w2p(pos)
        col -= self.col0
        row -= self.row0
        return col, row


    def pixel_to_world(self, col, row):
        col += self.col0
        row += self.row0

        pos = self.parse_inputs(col, row)

        const = np.pi / 128. * 2 ** (-self.zoom)
        lng_rad = (const*pos[0, :]) *  - np.pi

        lat_rad =  np.pi + (const*pos[1,:])
        lat_rad = 2 * np.arctan( np.exp(lat_rad) ) - np.pi/2.

        return np.degrees(lng_rad), np.degrees(lat_rad)


    def absolute_w2p(self, pos):
        """Compute col,row for absolute map, where origin is at
        -180 long and 85 lat

        This is an internal function, you probably don't want to call directly
        """

        pos = np.radians(pos)

        const = 128 / np.pi * 2 ** self.zoom
        col = const * (pos[0, :] + np.pi)

        row = np.pi/4. + pos[1,:]/2.
        row = np.pi - np.log( np.tan(row) )
        row *= const

        return col, row


class Hammer(AbstractProjection):
    """Also known as Hammer-Aitoff

    See https://en.wikipedia.org/wiki/Hammer_projection
    """
    def __init__(self):
        AbstractProjection.__init__(self)


    def world_to_pixel(self, lng, lat):
        pos = self.parse_inputs(lng, lat)
        pos = np.radians(pos)

        lam = pos[0, :]
        phi = pos[1, :]

        denom = 1 + np.cos(phi) * np.cos(.5*lam)
        denom = np.sqrt(denom)

        col = 2 * np.sqrt(2) * np.cos(phi) * np.sin(.5*lam)
        col /= denom

        row = np.sqrt(2) * np.sin(phi)
        row /= denom

        return col, row


    def pixel_to_world(self, col, row):
        pass


import frmbase.support as fsupport
import frmplots.plots as pp
import frmgis.kml as fkml

def test_mercator():
    obj = fkml.load("/orbital/etc/usshapes/cb_2015_us_state_20m.kml")
    shapes = fkml.gdalListFromKml(obj)[0][0]

    proj = WebMercator(-160, 65, 3)
#    proj = Hammer()

    plt.clf()
    plt.axis('equal')

    for i in range(len(shapes)):
        shape = shapes[i]

        coords = pp.ogrGeometryToArray(shape)
        for coo in coords:
            if np.min(coo[:,0]) > 0:
                coo[:,0] -= 360

            col, row = proj.w2p(coo[:,0], coo[:,1])

            plt.plot(col, -row, 'k-')




#TODO
#Spherical Mercator
#Alber's Equal Area

def test_parse_inputs():
    proj = AbstractProjection()

    val = proj.parse_inputs(1,2)
    assert val.shape == (2,1)
    assert val[0,0] == 1
    assert val[1,0] == 2

    x = np.arange(10)
    val = proj.parse_inputs(x, 2*x)
    assert val.shape == (2,10)

    assert np.array_equal(val[0, :], x)
    assert np.array_equal(val[1, :], 2*x)

def test_parse_outputs():
    proj = AbstractProjection()
    tmp = proj.parse_inputs(1,2)
    a, b = proj.parse_output(tmp)

    assert a == 1
    assert b == 2

    x = np.arange(10)
    tmp = proj.parse_inputs(x, 2*x)
    a, b = proj.parse_output(tmp)

    assert np.array_equal(a, x)
    assert np.array_equal(b, 2*x)



def test_PlateCaree():
    size = 100
    lngs = -180 + 360*np.random.random(size)
    lats = -90 + 180*np.random.random(size)

    proj = PlateCaree()
    cols, rows = proj.w2p(lngs, lats)

    assert np.sum( np.fabs(cols - lngs)) == 0
    assert np.sum( np.fabs(rows - lats)) == 0

    lng2, lat2 = proj.p2w(cols, rows)
    assert np.sum( np.fabs(cols - lng2)) == 0
    assert np.sum( np.fabs(rows - lat2)) == 0


def test_Linear():
    degrees_per_pixel = .01
    proj = LinearProjection(-8, 54, degrees_per_pixel)

    lng = np.linspace(-10, 0, 11)
    lat = np.linspace(50,60, 11)

    c, r = proj.w2p(-8, 54)
    assert c == 0
    assert r == 0

    col, row = proj.w2p(lng, lat)
    assert np.array_equal(col, np.linspace(-200, 800, 11)), col
    assert np.array_equal(row, np.linspace(-400, 600, 11)), row
