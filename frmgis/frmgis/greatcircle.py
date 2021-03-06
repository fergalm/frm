import numpy as np


"""
A set of functions to calculate properties of great circle lines on a
unit sphere, including, angular separation, bearing, etc.

Ported into Python from explanations at http://www.movable-type.co.uk/scripts/gis-faq-5.1.html
"""

__version__ = "$Id: greatcircle.py 33 2013-12-19 22:13:23Z fergalm $"
__URL__ = "$URL: svn+ssh://fergalm@svn.code.sf.net/p/keplertwowheel/code/py/greatcircle.py $"

#Equatorial radius. From Sec 9.2 of Cambrigde Handbook of Physical Formulae
EARTH_RADIUS_M = 6.37814e6

"""Routines to convert metres to/from degrees and radians on a spherical Earth"""
def deg_to_metres(deg):
    return rad_to_metres( np.radians(deg) )

def rad_to_metres(rad):
    return rad * np.pi * EARTH_RADIUS_M

def metres_to_rad(metres):
    return metres / (np.pi * EARTH_RADIUS_M)

def metres_to_deg(metres):
    return np.degrees(metres_to_rad(metres))

def sphericalAngSep(ra0, dec0, ra1, dec1, radians=False):
    """
        Compute the spherical angular separation between two
        points on the sky.

        //Taken from http://www.movable-type.co.uk/scripts/gis-faq-5.1.html

        NB: For small distances you can probably use
        sqrt( dDec**2 + cos^2(dec)*dRa)
        where dDec = dec1 - dec0 and
               dRa = ra1 - ra0
               and dec1 \approx dec \approx dec0
    """

    if radians==False:
        ra0  = np.radians(ra0)
        dec0 = np.radians(dec0)
        ra1  = np.radians(ra1)
        dec1 = np.radians(dec1)

    deltaRa= ra1-ra0
    deltaDec= dec1-dec0

    val = np.array(haversine(deltaDec))
    val += np.cos(dec0) * np.cos(dec1) * haversine(deltaRa)
    val = np.sqrt(val)
    val[val > 1] = 1  #Guard against round off error
    val = 2*np.arcsin(val)

    #Convert back to degrees if necessary
    if radians==False:
        val = np.degrees(val)

    #Scalar input gives scalar output
    if len(val) == 1:
        return val[0]
    return val


def sphericalAngSepFast(ra0, dec0, ra1, dec1, radians=False):
    """A faster (but less accurate) implementation of sphericalAngleSep

    Taken from http://www.movable-type.co.uk/scripts/latlong.html

    For additional speed, set wantSquare=True, and the return value
    is the square of the separation
    """

    if radians==False:
        ra0  = np.radians(ra0)
        dec0 = np.radians(dec0)
        ra1  = np.radians(ra1)
        dec1 = np.radians(dec1)

    deltaRa= ra1-ra0
    deltaDec= dec1-dec0
    avgDec = .5*(dec0+dec1)

    x = deltaRa*np.cos(avgDec)
    val = np.hypot(x, deltaDec)

    if radians == False:
        val = np.degrees(val)

    return val

def haversine(x):
    """Return the haversine of an angle

    haversine(x) = sin(x/2)**2, where x is an angle in radians
    """
    y = .5*x
    y = np.sin(y)
    return y*y


def sphericalAngBearing(ra0, dec0, ra1, dec1, radians=False):
    sin = np.sin
    cos = np.cos
    atan  = np.arctan2

    if radians==False:
        ra0  = np.radians(ra0)
        dec0 = np.radians(dec0)
        ra1  = np.radians(ra1)
        dec1 = np.radians(dec1)

    dLong = ra1 - ra0
    a = sin(dLong)*cos(dec1)
    b = cos(dec1)*sin(dec1) - sin(dec0)*cos(dec1)*cos(dLong)
    bearing = atan(a, b)

    if radians==False:
        bearing = np.degrees(bearing)

    return bearing


def sphericalAngDestination(ra0_deg, dec0_deg, bearing_deg, dist_deg):
    sin = np.sin
    cos = np.cos
    asin  = np.arcsin
    atan2 = np.arctan2

    phi1 = np.radians(dec0_deg)    #Latitude
    lambda1 = np.radians(ra0_deg)    #Longitude
    d = np.radians(dist_deg)      #Distance in radians
    theta = np.radians(bearing_deg)

    phi2 = sin(phi1)*cos(d)
    phi2 += cos(phi1)*sin(d)*cos(theta)
    phi2 = asin(phi2)

    a = sin(theta)*sin(d)*cos(phi1)
    b = cos(d) - sin(phi1)*sin(phi2)
    #print "A=%.7f b=%.7f" %(a,b)
    #import pdb; pdb.set_trace()
    lambda2 = lambda1 + atan2(a,b)

    ra2_deg = np.degrees(lambda2)
    dec2_deg = np.degrees(phi2)
    return ra2_deg, dec2_deg



