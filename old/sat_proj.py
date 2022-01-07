# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 09:45:27 2016

@author: fergal

This was code was rediscovered after many years. It isn't 
well tested.

$Id$
$URL$
"""

__version__ = "$Id$"
__URL__ = "$URL$"



import matplotlib.pyplot as mp
import numpy as np

#Should inherit from AbstractProjection
class SatelliteProjection():

    def __init__(self, lng0, lat0, altitude_Rearth, offnadir_north=0, offnadir_east=0):
      """

        Inputs
        ----------
        lng0, lat0
            (floats) Longitude and latitude of nadir point on the Earth
            for the viewer (e.g the satellite). Units are degrees
        altitude_Rearth
            (float) Altitude of the viewer in units of earth radius


        Optional Inputs
        -------------
        offnadir_north
            (float) How many degrees north of Nadir is the centre of the
            field of view
        offnadir_east
            (float) How many degrees east of nadir is the centre of the
            field of view
      """

      #AbstractProject.__init__(self)
      theta0 = np.pi/2. - np.radians(lat0)
      phi0 = np.radians(lng0)

      self.delta = np.radians(offnadir_north)
      self.epsilon = np.radians(offnadir_east)

      #Mneumonics
      cp, sp = np.cos(phi0), np.sin(phi0)
      ct, st = np.cos(theta0), np.sin(theta0)
      h = altitude_Rearth

      self.observerVec = (1+h) * np.array([cp*st, sp*st, ct])
      self.eastingVec = np.array([-sp, cp, 0])
      self.northingVec = np.array([-cp*ct, -sp*ct, st])

      self.cos_horizon_angle = 1 / (1+h)


    def world_to_pixel(self, lng, lat):
        #parse lng, lat

        theta = np.pi/2. - np.radians(lat)
        phi = np.radians(lng)

        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)

        qVec = np.array([cp*st, sp*st, ct]).transpose()

        dqVec = qVec - self.observerVec
        norm = np.linalg.norm(dqVec, axis=1)


#        import pdb; pdb.set_trace()
        #Are the points on a visible portion of the sphere
        #A point is visible if it's angular distance from nadir is
        #less than the angular distance to the horizon
        #=> cos(angle) > cos(horizon)

        cos_nadir_angle = np.dot(qVec, self.observerVec)
        cos_nadir_angle /= np.linalg.norm(qVec, axis=1)
        cos_nadir_angle /= np.linalg.norm(self.observerVec)

        assert np.all( np.fabs(cos_nadir_angle) < 1)
        visible = cos_nadir_angle > self.cos_horizon_angle

#        import pdb; pdb.set_trace()
        col = np.dot(dqVec, self.eastingVec) / norm
        col -= self.epsilon

        row = np.dot(dqVec, self.northingVec) / norm
        row -= self.delta

        return col, row, visible

import frm.plots as fplot
import frm.kml as fkml

def load():
    fn = "/home/fergal/orbital/datascience/fergal/nktrains/rails.kml"
    obj = fkml.load(fn)
    tracks = fkml.gdalListFromKml(obj)[0][0]
    return tracks


def main(tracks, altitude=.1, latitude=40):

    proj = SatelliteProjection(127, latitude, altitude)
    for t in tracks:
        points = fplot.ogrGeometryToArray(t)[0]
        cols, rows, visible = proj.world_to_pixel(points[:,0], points[:,1])
#        print cols
#        print rows
#        return
#
#        fplot.plot_shape(t, 'k-', lw=.5)

#        print np.sum(~visible)
        mp.plot(cols[visible], rows[visible], 'k-', lw=.5)
        mp.plot(cols[~visible], rows[~visible], 'r-', lw=.5)
#        return

    mp.title("%.1f degrees" %(latitude))
